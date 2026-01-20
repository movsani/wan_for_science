#!/usr/bin/env python3
"""
Script to download The Well turbulent_radiative_layer_2D dataset.

Usage:
    python scripts/download_data.py --output_dir ./datasets

This will download both training and validation splits.

NOTE: The well_download function creates a nested structure:
    {output_dir}/datasets/{dataset_name}/data/{split}/
    
So if you use --output_dir ./datasets, the actual data will be at:
    ./datasets/datasets/turbulent_radiative_layer_2D/data/train/
    
The config file (configs/default.yaml) expects base_path to be ./datasets/datasets
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_well_dataset(
    output_dir: str,
    dataset_name: str = "turbulent_radiative_layer_2D",
    splits: list = None,
    flatten: bool = False,
):
    """
    Download The Well dataset.
    
    Args:
        output_dir: Directory to store the dataset
        dataset_name: Name of the dataset to download
        splits: List of splits to download
        flatten: If True, move data to remove the nested 'datasets' folder
    """
    try:
        from the_well.utils.download import well_download
    except ImportError:
        print("Error: the_well package not installed.")
        print("Please install it with: pip install the_well[benchmark]")
        sys.exit(1)
    
    if splits is None:
        splits = ["train", "valid"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {dataset_name} dataset to {output_dir}")
    print(f"Splits: {splits}")
    print("-" * 50)
    
    for split in splits:
        print(f"\nDownloading {split} split...")
        try:
            well_download(
                base_path=output_dir,
                dataset=dataset_name,
                split=split,
            )
            print(f"Successfully downloaded {split} split")
        except Exception as e:
            print(f"Error downloading {split} split: {e}")
            continue
    
    # The well_download creates: {output_dir}/datasets/{dataset_name}/...
    # If flatten=True, move it to: {output_dir}/{dataset_name}/...
    nested_path = Path(output_dir) / "datasets" / dataset_name
    target_path = Path(output_dir) / dataset_name
    
    if flatten and nested_path.exists() and not target_path.exists():
        print(f"\nFlattening directory structure...")
        shutil.move(str(nested_path), str(target_path))
        # Remove empty datasets folder
        datasets_folder = Path(output_dir) / "datasets"
        if datasets_folder.exists() and not any(datasets_folder.iterdir()):
            datasets_folder.rmdir()
        final_location = target_path
        print(f"Moved data to: {target_path}")
    else:
        final_location = nested_path
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"Dataset location: {final_location}")
    
    # Print config hint
    print("\n" + "=" * 50)
    print("IMPORTANT - Config Setup:")
    print("-" * 50)
    if flatten:
        print(f"In configs/default.yaml, set:")
        print(f'  data.base_path: "{output_dir}"')
    else:
        print(f"In configs/default.yaml, set:")
        print(f'  data.base_path: "{output_dir}/datasets"')
    
    # Print dataset information
    print("\n" + "=" * 50)
    print("Dataset Information:")
    print("-" * 50)
    print(f"Dataset: {dataset_name}")
    print("Fields: density, pressure, velocity_x, velocity_y (4 channels)")
    print("Spatial resolution: 128 x 384")
    print("This dataset simulates turbulent radiative layer dynamics")
    print("\nFor more information, see:")
    print("https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/")


def main():
    parser = argparse.ArgumentParser(
        description="Download The Well turbulent_radiative_layer_2D dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Directory to store the dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="turbulent_radiative_layer_2D",
        help="Dataset name to download"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid"],
        help="Splits to download (train, valid, test)"
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten nested 'datasets' folder structure after download"
    )
    
    args = parser.parse_args()
    
    download_well_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        splits=args.splits,
        flatten=args.flatten,
    )


if __name__ == "__main__":
    main()
