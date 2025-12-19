"""
Custom Trainer class with loss and FID logging.

Extends the base Trainer to save:
- Training loss at every step
- FID scores during evaluation
"""

import torch
import math
import json
from pathlib import Path
from tqdm import tqdm
from torchvision import utils
from multiprocessing import cpu_count

# Import from red_diffeq package
from red_diffeq.models.diffusion import (
    GaussianDiffusion, Unet, FIDEvaluation, Trainer,
    cycle, num_to_groups, divisible_by, has_int_squareroot,
    exists, Adam, DataLoader, EMA, Accelerator
)

# Get version from red_diffeq
import red_diffeq
__version__ = red_diffeq.__version__


class TrainerWithLogging(object):
    """Extended Trainer that logs training loss and FID scores to files."""

    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 10000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, \
            f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True,
                       pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming. "
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=str(results_folder),
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

        # Loss and FID logging
        self.loss_log_path = self.results_folder / 'training_loss.json'
        self.fid_log_path = self.results_folder / 'fid_scores.json'
        self.loss_history = []
        self.fid_history = []

        # Initialize log files
        if self.accelerator.is_main_process:
            self._init_log_files()

    def _init_log_files(self):
        """Initialize log files with metadata."""
        # Loss log
        with open(self.loss_log_path, 'w') as f:
            json.dump({
                'metadata': {
                    'train_num_steps': self.train_num_steps,
                    'batch_size': self.batch_size,
                    'gradient_accumulate_every': self.gradient_accumulate_every,
                },
                'losses': []
            }, f, indent=2)

        # FID log
        if self.calculate_fid:
            with open(self.fid_log_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'save_and_sample_every': self.save_and_sample_every,
                    },
                    'fid_scores': []
                }, f, indent=2)

    def _log_loss(self, step, loss):
        """Log training loss to file."""
        if not self.accelerator.is_main_process:
            return

        self.loss_history.append({
            'step': step,
            'loss': loss
        })

        # Write to file every 100 steps to avoid too much I/O
        if step % 100 == 0:
            with open(self.loss_log_path, 'r') as f:
                data = json.load(f)

            data['losses'].extend(self.loss_history)
            self.loss_history = []

            with open(self.loss_log_path, 'w') as f:
                json.dump(data, f, indent=2)

    def _log_fid(self, step, fid_score):
        """Log FID score to file."""
        if not self.accelerator.is_main_process:
            return

        self.fid_history.append({
            'step': step,
            'fid': fid_score,
            'milestone': step // self.save_and_sample_every
        })

        with open(self.fid_log_path, 'r') as f:
            data = json.load(f)

        data['fid_scores'].extend(self.fid_history)
        self.fid_history = []

        with open(self.fid_log_path, 'w') as f:
            json.dump(data, f, indent=2)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps,
                 disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                # Log loss
                self._log_loss(self.step, total_loss)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                       nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                            # Log FID score
                            self._log_fid(self.step, fid_score)

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        # Flush any remaining logged data
        if accelerator.is_main_process and len(self.loss_history) > 0:
            with open(self.loss_log_path, 'r') as f:
                data = json.load(f)
            data['losses'].extend(self.loss_history)
            with open(self.loss_log_path, 'w') as f:
                json.dump(data, f, indent=2)

        accelerator.print('training complete')
