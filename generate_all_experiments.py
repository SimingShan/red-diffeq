#!/usr/bin/env python3
"""
Generate all experiment scripts for both bouchet and grace clusters
Following the hierarchical structure: {cluster}/{experiment_type}/{method}/
"""

import os
from pathlib import Path

# Cluster configurations
CLUSTERS = {
    'bouchet': {
        'partition': 'gpu_h200',
        'gpu': 'h200:1',
        'cpu_flag': 'cpus-per-gpu',
        'cpu_count': 2,
        'mem': '140G',
        'time': '0-06:00:00',
        'chdir': '/home/ss5235/scratch_pi_ll2247/ss5235/red-diffeq',
        'conda_env': '/home/ss5235/project_pi_ll2247/ss5235/conda_envs/fwi',
    },
    'grace': {
        'partition': 'gpu',
        'gpu': '1',
        'gpu_constraint': 'a100-80g',
        'cpu_flag': 'cpus-per-task',
        'cpu_count': 4,
        'mem': '80G',
        'time': '1-00:00:00',
        'chdir': '/vast/palmer/scratch/lu_lu/ss5235/red-diffeq',
        'conda_env': '/gpfs/gibbs/project/lu_lu/ss5235/envs/red-diffeq',
    }
}

# Method configurations
METHODS = {
    'baseline': {
        'regularization': 'none',
        'config': 'configs/openfwi/model_4.yaml',  # User requested model-4 for all methods
        'label': 'Baseline',
    },
    'tikhonov': {
        'regularization': 'l2',  # Main script uses 'l2' for Tikhonov regularization
        'reg_lambda': 0.01,
        'config': 'configs/openfwi/model_4.yaml',  # User requested model-4 for all methods
        'label': 'Tikhonov',
    },
    'tv': {
        'regularization': 'tv',
        'reg_lambda': 0.01,
        'config': 'configs/openfwi/model_4.yaml',  # User requested model-4 for all methods
        'label': 'TV',
    },
    'red_diffeq': {
        'regularization': 'diffusion',
        'reg_lambda': 0.75,
        'sigma': 10.0,
        'config': 'configs/openfwi/model_4.yaml',
        'label': 'RED-DiffEq',
    }
}

# Experiment configurations
EXPERIMENTS = {
    'missing': {
        'values': [15, 30, 45, 60],
        'batch_size': 10,
        'base_noise_std': 0.0,
    },
    'gaussian': {
        'values': [0.1, 0.2, 0.3, 0.4, 0.5],
        'batch_size': 25,
        'noise_type': 'gaussian',
    },
    'laplace': {
        'values': [0.1, 0.2, 0.3, 0.4, 0.5],
        'batch_size': 25,
        'noise_type': 'laplace',
    }
}

def get_job_name(cluster, method, exp_type, value):
    """Generate informative job names - all include method prefix"""
    method_abbr = {
        'baseline': 'base',
        'tikhonov': 'tik',
        'tv': 'tv',
        'red_diffeq': 'reddiff'
    }[method]

    if exp_type == 'missing':
        return f"{method_abbr}_miss_{value}"
    elif exp_type == 'gaussian':
        return f"{method_abbr}_gauss_{value}"
    elif exp_type == 'laplace':
        return f"{method_abbr}_lapl_{value}"

    return "experiment"

def generate_slurm_header(cluster, job_name):
    """Generate SLURM header for a cluster"""
    cfg = CLUSTERS[cluster]

    header = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={cfg['partition']}
#SBATCH --gpus={cfg['gpu']}
"""

    if 'gpu_constraint' in cfg:
        header += f"#SBATCH --constraint={cfg['gpu_constraint']}\n"

    header += f"""#SBATCH --{cfg['cpu_flag']}={cfg['cpu_count']}
#SBATCH --mem={cfg['mem']}
#SBATCH --time={cfg['time']}
#SBATCH --output=%x_%j.out
#SBATCH --chdir={cfg['chdir']}

module reset
module load miniconda
conda activate {cfg['conda_env']}
"""

    return header

def generate_missing_script(cluster, method, missing_num):
    """Generate missing traces experiment script"""
    cfg_cluster = CLUSTERS[cluster]
    cfg_method = METHODS[method]
    cfg_exp = EXPERIMENTS['missing']

    job_name = get_job_name(cluster, method, 'missing', missing_num)

    script = generate_slurm_header(cluster, job_name)

    # Add comment
    script += f"\n# {cfg_method['label']}: Missing traces {missing_num}\n"

    # Add variables
    script += f"""ITERATIONS=300
LR=0.03
BATCH_SIZE={cfg_exp['batch_size']}
REGULARIZATION="{cfg_method['regularization']}"
"""

    if 'reg_lambda' in cfg_method:
        script += f"REG_LAMBDA={cfg_method['reg_lambda']}\n"

    if 'sigma' in cfg_method:
        script += f"SIGMA={cfg_method['sigma']}\n"

    script += f"""NOISE_STD={cfg_exp['base_noise_std']}
MISSING_NUMBER={missing_num}

CONFIG="{cfg_method['config']}"
EXP_NAME="openfwi_model4_{method}_missing{missing_num}"

echo "============================================================"
echo "OpenFWI {cfg_method['label']}: Missing Traces {missing_num}"
echo "Config: ${{CONFIG}}"
echo "Batch size: ${{BATCH_SIZE}}"
echo "Regularization: {cfg_method['label']}"
echo "Families: CF, CV, FF, FV (4 × 100 samples)"
echo "============================================================"

python scripts/run_inversion.py \\
    --config ${{CONFIG}} \\
    --ts ${{ITERATIONS}} \\
    --lr ${{LR}} \\
    --batch_size ${{BATCH_SIZE}} \\
    --regularization ${{REGULARIZATION}} \\
"""

    if 'reg_lambda' in cfg_method:
        script += f"    --reg_lambda ${{REG_LAMBDA}} \\\n"

    if 'sigma' in cfg_method:
        script += f"    --sigma ${{SIGMA}} \\\n"

    script += f"""    --noise_std ${{NOISE_STD}} \\
    --noise_type gaussian \\
    --missing_number ${{MISSING_NUMBER}} \\
    --experiment_name ${{EXP_NAME}}

if [ $? -eq 0 ]; then
    echo "✓ Completed: ${{EXP_NAME}}"
else
    echo "✗ Failed: ${{EXP_NAME}}"
    exit 1
fi
"""

    return script

def generate_noisy_script(cluster, method, noise_type, noise_std):
    """Generate noisy data experiment script"""
    cfg_cluster = CLUSTERS[cluster]
    cfg_method = METHODS[method]
    cfg_exp = EXPERIMENTS[noise_type]

    job_name = get_job_name(cluster, method, noise_type, noise_std)

    script = generate_slurm_header(cluster, job_name)

    # Add comment
    noise_label = noise_type.capitalize()
    script += f"\n# {cfg_method['label']}: {noise_label} noise {noise_std}\n"

    # Add variables
    script += f"""ITERATIONS=300
LR=0.03
BATCH_SIZE={cfg_exp['batch_size']}
REGULARIZATION="{cfg_method['regularization']}"
"""

    if 'reg_lambda' in cfg_method:
        script += f"REG_LAMBDA={cfg_method['reg_lambda']}\n"

    if 'sigma' in cfg_method:
        script += f"SIGMA={cfg_method['sigma']}\n"

    script += f"""NOISE_STD={noise_std}
NOISE_TYPE="{noise_type}"

CONFIG="{cfg_method['config']}"
EXP_NAME="openfwi_model4_{method}_{noise_type}{noise_std}"

echo "============================================================"
echo "OpenFWI {cfg_method['label']}: {noise_label} Noise {noise_std}"
echo "Config: ${{CONFIG}}"
echo "Batch size: ${{BATCH_SIZE}}"
echo "Regularization: {cfg_method['label']}"
echo "Families: CF, CV, FF, FV (4 × 100 samples)"
echo "============================================================"

python scripts/run_inversion.py \\
    --config ${{CONFIG}} \\
    --ts ${{ITERATIONS}} \\
    --lr ${{LR}} \\
    --batch_size ${{BATCH_SIZE}} \\
    --regularization ${{REGULARIZATION}} \\
"""

    if 'reg_lambda' in cfg_method:
        script += f"    --reg_lambda ${{REG_LAMBDA}} \\\n"

    if 'sigma' in cfg_method:
        script += f"    --sigma ${{SIGMA}} \\\n"

    script += f"""    --noise_std ${{NOISE_STD}} \\
    --noise_type ${{NOISE_TYPE}} \\
    --experiment_name ${{EXP_NAME}}

if [ $? -eq 0 ]; then
    echo "✓ Completed: ${{EXP_NAME}}"
else
    echo "✗ Failed: ${{EXP_NAME}}"
    exit 1
fi
"""

    return script

def generate_checkpoint_script(cluster, model_num):
    """Generate checkpoint comparison script (only for red_diffeq)"""
    cfg_cluster = CLUSTERS[cluster]

    job_name = f"reddiff_model_{model_num}"

    script = generate_slurm_header(cluster, job_name)

    script += f"""
# Model-{model_num} ablation (clean)
ITERATIONS=300
LR=0.03
BATCH_SIZE=25
REGULARIZATION="diffusion"
REG_LAMBDA=0.75
SIGMA=10.0
NOISE_STD=0.0

CONFIG="configs/openfwi/model_{model_num}.yaml"
EXP_NAME="openfwi_model{model_num}_clean"

echo "============================================================"
echo "OpenFWI RED-DiffEq: Model-{model_num} Ablation (Clean)"
echo "Config: ${{CONFIG}}"
echo "Batch size: ${{BATCH_SIZE}}"
echo "Families: CF, CV, FF, FV (4 × 100 samples)"
echo "============================================================"

python scripts/run_inversion.py \\
    --config ${{CONFIG}} \\
    --ts ${{ITERATIONS}} \\
    --lr ${{LR}} \\
    --batch_size ${{BATCH_SIZE}} \\
    --regularization ${{REGULARIZATION}} \\
    --reg_lambda ${{REG_LAMBDA}} \\
    --sigma ${{SIGMA}} \\
    --noise_std ${{NOISE_STD}} \\
    --noise_type gaussian \\
    --experiment_name ${{EXP_NAME}}

if [ $? -eq 0 ]; then
    echo "✓ Completed: ${{EXP_NAME}}"
else
    echo "✗ Failed: ${{EXP_NAME}}"
    exit 1
fi
"""

    return script

def generate_time_weight_script(cluster):
    """Generate time-weighted experiment script (only for red_diffeq)"""
    cfg_cluster = CLUSTERS[cluster]

    job_name = "reddiff_time_wt"

    script = generate_slurm_header(cluster, job_name)

    script += f"""
# Time-weighted experiment (clean)
ITERATIONS=300
LR=0.03
BATCH_SIZE=25
REGULARIZATION="diffusion"
REG_LAMBDA=0.75
SIGMA=10.0
NOISE_STD=0.0
TIME_WEIGHT=1

CONFIG="configs/openfwi/model_4.yaml"
EXP_NAME="openfwi_model4_timeweight"

echo "============================================================"
echo "OpenFWI RED-DiffEq: Time-Weighted Experiment"
echo "Config: ${{CONFIG}}"
echo "Batch size: ${{BATCH_SIZE}}"
echo "Time weight: ${{TIME_WEIGHT}}"
echo "Families: CF, CV, FF, FV (4 × 100 samples)"
echo "============================================================"

python scripts/run_inversion.py \\
    --config ${{CONFIG}} \\
    --ts ${{ITERATIONS}} \\
    --lr ${{LR}} \\
    --batch_size ${{BATCH_SIZE}} \\
    --regularization ${{REGULARIZATION}} \\
    --reg_lambda ${{REG_LAMBDA}} \\
    --sigma ${{SIGMA}} \\
    --noise_std ${{NOISE_STD}} \\
    --noise_type gaussian \\
    --time_weight ${{TIME_WEIGHT}} \\
    --experiment_name ${{EXP_NAME}}

if [ $? -eq 0 ]; then
    echo "✓ Completed: ${{EXP_NAME}}"
else
    echo "✗ Failed: ${{EXP_NAME}}"
    exit 1
fi
"""

    return script

def generate_submit_all_script(cluster, exp_type, method=None):
    """Generate submit_all script for a specific experiment type"""

    if exp_type == 'checkpoints':
        scripts = [f"model_{i}.sh" for i in [1, 2, 3, 4]]
        title = "Checkpoint Comparison"
    elif exp_type == 'wt':
        scripts = ["time_weight.sh"]
        title = "Time-Weighted"
    elif exp_type == 'missing':
        scripts = [f"missing_{val}.sh" for val in EXPERIMENTS['missing']['values']]
        title = f"Missing Traces - {METHODS[method]['label']}"
    elif exp_type in ['gaussian', 'laplace']:
        scripts = [f"{exp_type}_{val}.sh" for val in EXPERIMENTS[exp_type]['values']]
        title = f"{exp_type.capitalize()} Noise - {METHODS[method]['label']}"
    else:
        return ""

    submit_script = f"""#!/bin/bash
# Submit all {title} experiments on {cluster}

echo "Submitting {title} experiments..."
echo "============================================================"

"""

    for script in scripts:
        submit_script += f"sbatch {script}\n"

    submit_script += f"""
echo "============================================================"
echo "All jobs submitted!"
echo "Check status with: squeue -u $USER"
"""

    return submit_script

def main():
    """Generate all experiment scripts"""

    base_dir = Path("experiment/exp_sh")

    for cluster in ['bouchet', 'grace']:
        print(f"\n{'='*60}")
        print(f"Generating scripts for {cluster.upper()}")
        print('='*60)

        cluster_dir = base_dir / cluster

        # 1. Missing traces experiments
        print(f"\n[{cluster}] Generating missing traces experiments...")
        for method in ['baseline', 'tikhonov', 'tv', 'red_diffeq']:
            method_dir = cluster_dir / 'missing' / method
            method_dir.mkdir(parents=True, exist_ok=True)

            for missing_num in EXPERIMENTS['missing']['values']:
                script_path = method_dir / f"missing_{missing_num}.sh"
                script = generate_missing_script(cluster, method, missing_num)
                script_path.write_text(script)
                script_path.chmod(0o755)
                print(f"  ✓ {script_path.relative_to(base_dir)}")

            # Submit all script
            submit_path = method_dir / f"submit_all_{method}.sh"
            submit_script = generate_submit_all_script(cluster, 'missing', method)
            submit_path.write_text(submit_script)
            submit_path.chmod(0o755)
            print(f"  ✓ {submit_path.relative_to(base_dir)} (submit script)")

        # 2. Gaussian noise experiments
        print(f"\n[{cluster}] Generating gaussian noise experiments...")
        for method in ['baseline', 'tikhonov', 'tv', 'red_diffeq']:
            method_dir = cluster_dir / 'noisy' / 'gaussian' / method
            method_dir.mkdir(parents=True, exist_ok=True)

            for noise_std in EXPERIMENTS['gaussian']['values']:
                script_path = method_dir / f"gaussian_{noise_std}.sh"
                script = generate_noisy_script(cluster, method, 'gaussian', noise_std)
                script_path.write_text(script)
                script_path.chmod(0o755)
                print(f"  ✓ {script_path.relative_to(base_dir)}")

            # Submit all script
            submit_path = method_dir / f"submit_all_{method}.sh"
            submit_script = generate_submit_all_script(cluster, 'gaussian', method)
            submit_path.write_text(submit_script)
            submit_path.chmod(0o755)
            print(f"  ✓ {submit_path.relative_to(base_dir)} (submit script)")

        # 3. Laplace noise experiments
        print(f"\n[{cluster}] Generating laplace noise experiments...")
        for method in ['baseline', 'tikhonov', 'tv', 'red_diffeq']:
            method_dir = cluster_dir / 'noisy' / 'laplace' / method
            method_dir.mkdir(parents=True, exist_ok=True)

            for noise_std in EXPERIMENTS['laplace']['values']:
                script_path = method_dir / f"laplace_{noise_std}.sh"
                script = generate_noisy_script(cluster, method, 'laplace', noise_std)
                script_path.write_text(script)
                script_path.chmod(0o755)
                print(f"  ✓ {script_path.relative_to(base_dir)}")

            # Submit all script
            submit_path = method_dir / f"submit_all_{method}.sh"
            submit_script = generate_submit_all_script(cluster, 'laplace', method)
            submit_path.write_text(submit_script)
            submit_path.chmod(0o755)
            print(f"  ✓ {submit_path.relative_to(base_dir)} (submit script)")

        # 4. Checkpoint comparison experiments
        print(f"\n[{cluster}] Generating checkpoint comparison experiments...")
        checkpoint_dir = cluster_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for model_num in [1, 2, 3, 4]:
            script_path = checkpoint_dir / f"model_{model_num}.sh"
            script = generate_checkpoint_script(cluster, model_num)
            script_path.write_text(script)
            script_path.chmod(0o755)
            print(f"  ✓ {script_path.relative_to(base_dir)}")

        # Submit all script for checkpoints
        submit_path = checkpoint_dir / "submit_all_checkpoints.sh"
        submit_script = generate_submit_all_script(cluster, 'checkpoints')
        submit_path.write_text(submit_script)
        submit_path.chmod(0o755)
        print(f"  ✓ {submit_path.relative_to(base_dir)} (submit script)")

        # 5. Time-weighted experiment
        print(f"\n[{cluster}] Generating time-weighted experiment...")
        wt_dir = cluster_dir / 'wt'
        wt_dir.mkdir(parents=True, exist_ok=True)

        script_path = wt_dir / "time_weight.sh"
        script = generate_time_weight_script(cluster)
        script_path.write_text(script)
        script_path.chmod(0o755)
        print(f"  ✓ {script_path.relative_to(base_dir)}")

        # Submit all script for time weight
        submit_path = wt_dir / "submit_all_wt.sh"
        submit_script = generate_submit_all_script(cluster, 'wt')
        submit_path.write_text(submit_script)
        submit_path.chmod(0o755)
        print(f"  ✓ {submit_path.relative_to(base_dir)} (submit script)")

    print(f"\n{'='*60}")
    print("✓ ALL SCRIPTS GENERATED SUCCESSFULLY!")
    print('='*60)
    print("\nSummary:")
    print("  - Missing traces: 4 methods × 4 values × 2 clusters = 32 scripts")
    print("  - Gaussian noise: 4 methods × 5 values × 2 clusters = 40 scripts")
    print("  - Laplace noise: 4 methods × 5 values × 2 clusters = 40 scripts")
    print("  - Checkpoints: 4 models × 2 clusters = 8 scripts")
    print("  - Time weight: 1 × 2 clusters = 2 scripts")
    print("  - Submit scripts: automated batch submission")
    print(f"\n  TOTAL: 122+ experiment scripts across both clusters")
    print("\nNote: All methods use model_4.yaml as requested")

if __name__ == "__main__":
    main()
