#!/usr/bin/env python
"""
Generate or update requirements.txt with the most recent package versions.

Usage:
    python scripts/generate_requirements.py              # Generate from current environment
    python scripts/generate_requirements.py --latest      # Check latest versions (slower)
    python scripts/generate_requirements.py --scan       # Scan imports with pipreqs
"""

import subprocess
import sys
from pathlib import Path

def get_installed_packages():
    """Get currently installed packages from pip freeze."""
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        print("Error: Could not run 'pip freeze'")
        return []

def filter_project_packages(packages, requirements_file):
    """Filter pip freeze output to only packages in requirements.txt."""
    if not requirements_file.exists():
        return packages
    
    # Read existing requirements to know which packages to keep
    existing = set()
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                existing.add(pkg_name.lower())
    
    # Also add common dependencies we know we need
    known_packages = {
        'torch', 'torchvision', 'numpy', 'scipy', 'matplotlib', 
        'tqdm', 'pandas', 'scikit-learn', 'pillow', 'einops',
        'accelerate', 'ema-pytorch', 'denoising-diffusion-pytorch',
        'tensorboard', 'tensorboardx', 'ml-collections', 'ml_collections',
        'pyyaml', 'h5py', 'zarr', 'wandb', 'safetensors', 'huggingface-hub'
    }
    
    filtered = []
    for pkg in packages:
        if not pkg or '==' not in pkg:
            continue
        pkg_name = pkg.split('==')[0].lower()
        if pkg_name in existing or any(known in pkg_name for known in known_packages):
            filtered.append(pkg)
    
    return filtered

def get_latest_version(package_name):
    """Get latest version of a package from PyPI."""
    try:
        result = subprocess.run(
            ["pip", "index", "versions", package_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse output to get latest version
            for line in result.stdout.split('\n'):
                if 'Available versions:' in line or 'LATEST:' in line:
                    # Extract version number
                    import re
                    versions = re.findall(r'\d+\.\d+\.\d+', line)
                    if versions:
                        return versions[0]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback: use pip show
    try:
        result = subprocess.run(
            ["pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
    except:
        pass
    
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate requirements.txt')
    parser.add_argument('--latest', action='store_true', 
                       help='Check latest versions from PyPI (slower)')
    parser.add_argument('--scan', action='store_true',
                       help='Use pipreqs to scan imports')
    parser.add_argument('--output', type=str, default='requirements.txt',
                       help='Output file path')
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    requirements_file = repo_root / args.output
    
    if args.scan:
        print("Scanning imports with pipreqs...")
        try:
            subprocess.run(
                ["pipreqs", str(repo_root), "--force", "--savepath", str(requirements_file)],
                check=True
            )
            print(f"✅ Generated {requirements_file} from imports")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ pipreqs not available. Install with: pip install pipreqs")
            print("Falling back to pip freeze method...")
            args.scan = False
    
    if not args.scan:
        print("Generating from current environment...")
        packages = get_installed_packages()
        
        if args.latest:
            print("Checking latest versions (this may take a while)...")
            updated = []
            for pkg_line in packages:
                if '==' in pkg_line:
                    pkg_name = pkg_line.split('==')[0]
                    latest = get_latest_version(pkg_name)
                    if latest:
                        updated.append(f"{pkg_name}=={latest}")
                        print(f"  {pkg_name}: {pkg_line.split('==')[1]} → {latest}")
                    else:
                        updated.append(pkg_line)
                else:
                    updated.append(pkg_line)
            packages = updated
        
        # Filter to project-relevant packages
        filtered = filter_project_packages(packages, requirements_file)
        
        # Write to file
        with open(requirements_file, 'w') as f:
            f.write("# Generated requirements.txt\n")
            f.write("# To update: python scripts/generate_requirements.py\n\n")
            for pkg in sorted(set(filtered)):
                f.write(f"{pkg}\n")
        
        print(f"✅ Generated {requirements_file} with {len(filtered)} packages")
        print(f"   Location: {requirements_file.absolute()}")

if __name__ == '__main__':
    main()

