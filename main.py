import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser
from datetime import datetime
import os
import torch.nn as nn
import scripts.pytorch_ssim as pytorch_ssim
from scripts.diffusion_model import *
from scripts.red_diffeq import *
from scripts.forward import FWIForward
from accelerate import Accelerator
from torch.optim import Adam
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssim_loss = pytorch_ssim.SSIM(window_size=11)
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

def main(regularization, lr, ts, sigma, loss_type, noise_std, initial_type, missing_number, method, reg_lambda):
    # Context setup
    ctx = {
        'n_grid': 70, 'nt': 1000, 'dx': 10, 'nbc': 120,
        'dt': 1e-3, 'f': 15, 'sz': 10, 'gz': 10, 'ng': 70, 'ns': 5
    }
    fwi_forward = FWIForward(ctx, device, normalize=True, v_denorm_func=data_trans.v_denormalize, s_norm_func=data_trans.s_normalize_none)

    # Initialize the model and diffusion process
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        channels=1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=72,
        timesteps=1000,    # number of steps
        sampling_timesteps=250,
        objective='pred_noise'
    ).to('cuda')

    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='fp16'
    )
    opt = Adam(diffusion.parameters(), lr=20, betas=(0.9, 0.99))
    diffusion, opt = accelerator.prepare(diffusion, opt)
    diffusion = accelerator.unwrap_model(diffusion)
    model_path = os.path.expanduser("model-4.pt")
    diffusion.load_state_dict(torch.load(model_path, map_location='cuda')['model'])
    diffusion.eval()
    
    # Update directory naming to include arguments
    dir_identifier = regularization if regularization else 'pure'
    args_str = f"{method}_{dir_identifier}_lr{lr}_ts{ts}_sigma{sigma}_loss{loss_type}_noisestd{noise_std}_missing{missing_number}"
    results_dir = f"experiment/{args_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    seismic_base_dir = "dataset/Test_Data/Seismic_Data_Test/"
    velocity_base_dir = "dataset/Test_Data/Velocity_Data_Test/"
    family_name_list = [file for file in os.listdir(seismic_base_dir) if file.endswith('.npy')]

    for family_name in family_name_list:
        family_results_dir = os.path.join(results_dir, family_name)
        os.makedirs(family_results_dir, exist_ok=True)

        seismic_dir = os.path.join(seismic_base_dir, family_name)
        velocity_dir = os.path.join(velocity_base_dir, family_name)
        mus, losses, reg_losses_raw_all = [], [], []
        matrices = {'MAE': [], 'RMSE': [], 'SSIM': []}

        seis_data = np.load(seismic_dir)
        velocity_data = np.load(velocity_dir)
        print(f"There are {seis_data.shape[0]} data in the file")
        for j in range(seis_data.shape[0]):
            seis_slice = torch.from_numpy(seis_data[j:j+1]).float().to(device)
            vel_slice = torch.from_numpy(velocity_data[j:j+1]).float()
            initial_model = data_trans.prepare_initial_model(vel_slice, initial_type, sigma=sigma)
            initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)
            red_diff = RED_DiffEq(
                diffusion_model=diffusion,
                data_trans_module=data_trans,
                data_vis_module=data_vis,
                pytorch_ssim_module=pytorch_ssim
            )
            if method == 'red_diff':
                mu, total_losses, obs_losses, reg_losses, reg_losses_raw, current_MAE, current_RMSE, current_SSIM = red_diff.sample(
                    mu=initial_model,
                    mu_true=vel_slice,
                    y=seis_slice,
                    ts=ts,
                    lr=lr,
                    fwi_forward=fwi_forward,
                    regularization=regularization,
                    plot_show=False,
                    loss_type=loss_type,
                    noise_std=noise_std,
                    missing_number=missing_number,
                    reg_lambda = reg_lambda
                )
            else:
                raise ValueError("Invalid method selected.")

            mu_tensor = mu[:, :, 1:-1, 1:-1].detach().cpu()
            mu_numpy = mu[:, :, 1:-1, 1:-1].detach().cpu().numpy()
            vm_data_norm = torch.tensor(data_trans.v_normalize(velocity_data[j:j+1]))
            mae = l1_loss(mu_tensor, vm_data_norm).item()
            mse = l2_loss(mu_tensor, vm_data_norm).item()
            rmse = np.sqrt(mse)
            ssim = ssim_loss((mu_tensor + 1) / 2, (vm_data_norm + 1) / 2).item()

            matrices['MAE'].append(mae)
            matrices['RMSE'].append(rmse)
            matrices['SSIM'].append(ssim)

            mus.append(mu_numpy)
            losses.append(obs_losses)
            reg_losses_raw_all.append(reg_losses_raw)

            print(f"MAE: {mae}, RMSE: {rmse}, SSIM: {ssim}")

        # Save collective results for the family
        mu_result = np.concatenate(mus)
        losses_result = np.concatenate(losses)
        reg_losses_raw_result = np.concatenate(reg_losses_raw_all)

        average_mae = np.mean(matrices['MAE'])
        average_rmse = np.mean(matrices['RMSE'])
        average_ssim = np.mean(matrices['SSIM'])

        np.save(f'{family_results_dir}/velocity_sample.npy', mu_result)
        np.save(f'{family_results_dir}/losses.npy', losses_result)
        np.save(f'{family_results_dir}/reg_losses_raw.npy', reg_losses_raw_result)

        print('*' * 50)
        print(f'{family_name} sampling is DONE')
        print(f'Average MAE: {average_mae}, RMSE: {average_rmse}, SSIM: {average_ssim}')

        with open(os.path.join(family_results_dir, 'metrics_summary.txt'), 'w') as f:
            f.write(f'Average MAE: {average_mae}\n')
            f.write(f'Average RMSE: {average_rmse}\n')
            f.write(f'Average SSIM: {average_ssim}\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--regularization", default=None, help="Regularization type (e.g., 'diffusion', 'tv')")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate for optimizer")
    parser.add_argument("--ts", type=int, default=300, help="Number of timesteps for training")
    parser.add_argument("--sigma", type=float, default=10, help="Sigma for Gaussian blurring in initial model preparation")
    parser.add_argument("--loss_type", type=str, default='l1', help="Type of the loss function")
    parser.add_argument("--noise_std", type=float, default=0, help="The standard deviation of the gaussian noise")
    parser.add_argument("--missing_number", type=int, default=0, help="The missing trace of the seismic data")
    parser.add_argument("--initial_type", type=str, default='smoothed', help="Type of initial velocity model")
    parser.add_argument("--method", type=str, default='red_diff', help="Choose to run the benchmark or our method")
    parser.add_argument("--reg_lambda", type=float, default='0.01', help="The regularization coefficient lambda")
    args = parser.parse_args()

    main(regularization=args.regularization, 
         lr=args.lr, 
         ts=args.ts, 
         sigma=args.sigma, 
         loss_type=args.loss_type, 
         noise_std=args.noise_std, 
         initial_type=args.initial_type,
         missing_number=args.missing_number,
         method=args.method,
         reg_lambda=args.reg_lambda)
