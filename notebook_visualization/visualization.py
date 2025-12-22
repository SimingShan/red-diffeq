import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json
from collections import defaultdict

def load_openfwi(folder_path, return_final_only=False):

    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return None
    
    # Load data from family directories (CF, CV, FF, FV)
    all_mae, all_rmse, all_ssim = [], [], []
    for family in ['CF', 'CV', 'FF', 'FV']:
        family_dir = folder_path / family
        if not family_dir.exists():
            continue
        
        for npz_file in family_dir.glob('*_results.npz'):
            try:
                data = np.load(npz_file)
                all_mae.append(data['mae'])
                all_rmse.append(data['rmse'])
                all_ssim.append(data['ssim'])
            except Exception as e:
                print(f"Warning: Failed to load {npz_file}: {e}")
                continue
    
    if not all_mae:
        print(f"Error: No valid npz files found in {folder_path}")
        return None
    
    if return_final_only:
        return {
            'MAE': np.mean([mae[-1] for mae in all_mae]),
            'RMSE': np.mean([rmse[-1] for rmse in all_rmse]),
            'SSIM': np.mean([ssim[-1] for ssim in all_ssim])
        }
    else:
        return {
            'mae': np.mean(all_mae, axis=0),
            'rmse': np.mean(all_rmse, axis=0),
            'ssim': np.mean(all_ssim, axis=0)
        }

def load_marmousi(folder_path, return_final_only=False):

    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return None
    
    # Look for dataset subdirectory (marmousi or overthrust)
    dataset_dirs = [d for d in folder_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.') 
                   and d.name.lower() in ['marmousi', 'overthrust']]
    
    if dataset_dirs:
        # Has dataset subdirectory: folder_path/{dataset}/X_results.npz
        dataset_dir = dataset_dirs[0]
        npz_files = list(dataset_dir.glob('*_results.npz'))
    else:
        # Check if npz files are directly in folder_path
        npz_files = list(folder_path.glob('*_results.npz'))
        if npz_files:
            dataset_dir = folder_path
        else:
            print(f"Error: Could not find npz files in {folder_path}")
            return None
    
    if not npz_files:
        print(f"Error: No npz files found in {folder_path}")
        return None
    
    all_mae, all_rmse, all_ssim = [], [], []
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            all_mae.append(data['mae'])
            all_rmse.append(data['rmse'])
            all_ssim.append(data['ssim'])
        except Exception as e:
            print(f"Warning: Failed to load {npz_file}: {e}")
            continue
    
    if not all_mae:
        print(f"Error: No valid data loaded from {folder_path}")
        return None
    
    if return_final_only:
        return {
            'MAE': np.mean([mae[-1] for mae in all_mae]),
            'RMSE': np.mean([rmse[-1] for rmse in all_rmse]),
            'SSIM': np.mean([ssim[-1] for ssim in all_ssim])
        }
    else:
        return {
            'mae': np.mean(all_mae, axis=0),
            'rmse': np.mean(all_rmse, axis=0),
            'ssim': np.mean(all_ssim, axis=0)
        }
