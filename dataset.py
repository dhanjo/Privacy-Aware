"""
PyTorch Dataset class for Market-1501 person re-identification dataset.
Handles data loading, preprocessing, and dataloader creation.
"""

import os
import re
from typing import Tuple, List, Callable, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

import config


class Market1501Dataset(Dataset):
    """
    Market-1501 Dataset for Person Re-Identification.

    Market-1501 filename format: PPPP_cCsS_FFFFFF_NN.jpg
    - PPPP: Person ID (0001-1501, with 0000 and -1 being junk/distractor)
    - C: Camera ID (1-6)
    - S: Sequence ID
    - F: Frame number
    - N: Detection index within the frame
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize Market-1501 dataset.

        Args:
            root: Root directory of Market-1501 dataset
            split: One of 'train', 'query', or 'gallery'
            transform: Optional transform to be applied on images
        """
        self.root = root
        self.split = split
        self.transform = transform

        # Map split names to folder names
        split_folders = {
            "train": "bounding_box_train",
            "query": "query",
            "gallery": "bounding_box_test",
        }

        if split not in split_folders:
            raise ValueError(f"Split must be one of {list(split_folders.keys())}")

        self.data_dir = os.path.join(root, split_folders[split])

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}\n"
                f"Please run download_dataset.py first."
            )

        # Load dataset
        self.img_paths, self.person_ids, self.camera_ids = self._load_data()

        print(f"Loaded {split} split: {len(self.img_paths)} images, "
              f"{len(set(self.person_ids))} identities, "
              f"{len(set(self.camera_ids))} cameras")

    def _load_data(self) -> Tuple[List[str], List[int], List[int]]:
        """
        Load image paths and parse person IDs and camera IDs from filenames.

        Returns:
            Tuple of (image_paths, person_ids, camera_ids)
        """
        img_paths = []
        person_ids = []
        camera_ids = []

        # Pattern: PPPP_cC...
        pattern = re.compile(r'([-\d]+)_c(\d)')

        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith(('.jpg', '.png')):
                continue

            match = pattern.match(filename)
            if not match:
                continue

            person_id = int(match.group(1))
            camera_id = int(match.group(2))

            # Filter out junk images (person_id = -1 or 0)
            if person_id <= 0:
                continue

            img_paths.append(os.path.join(self.data_dir, filename))
            person_ids.append(person_id)
            camera_ids.append(camera_id)

        # Remap person IDs to start from 0
        unique_person_ids = sorted(set(person_ids))
        person_id_map = {pid: idx for idx, pid in enumerate(unique_person_ids)}
        person_ids = [person_id_map[pid] for pid in person_ids]

        return img_paths, person_ids, camera_ids

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image, person_id, camera_id)
        """
        img_path = self.img_paths[idx]
        person_id = self.person_ids[idx]
        camera_id = self.camera_ids[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, person_id, camera_id

    def get_num_classes(self) -> int:
        """Return the number of unique person identities."""
        return len(set(self.person_ids))


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Get image transforms for the specified split.

    Args:
        split: One of 'train', 'query', or 'gallery'

    Returns:
        torchvision transforms composition
    """
    height = config.IMAGE_CONFIG["height"]
    width = config.IMAGE_CONFIG["width"]
    mean = config.IMAGE_CONFIG["mean"]
    std = config.IMAGE_CONFIG["std"]

    if split == "train":
        # Training transforms with data augmentation
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Test transforms without augmentation
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def create_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, query, and gallery dataloaders.

    Args:
        data_root: Root directory of Market-1501 dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, query_loader, gallery_loader, num_classes)
    """
    # Create datasets
    train_dataset = Market1501Dataset(
        root=data_root,
        split="train",
        transform=get_transforms("train")
    )

    query_dataset = Market1501Dataset(
        root=data_root,
        split="query",
        transform=get_transforms("test")
    )

    gallery_dataset = Market1501Dataset(
        root=data_root,
        split="gallery",
        transform=get_transforms("test")
    )

    num_classes = train_dataset.get_num_classes()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for batch normalization
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, query_loader, gallery_loader, num_classes


def get_dataset_stats(data_root: str) -> dict:
    """
    Get statistics about the Market-1501 dataset.

    Args:
        data_root: Root directory of Market-1501 dataset

    Returns:
        Dictionary with dataset statistics
    """
    train_dataset = Market1501Dataset(
        root=data_root,
        split="train",
        transform=None
    )

    query_dataset = Market1501Dataset(
        root=data_root,
        split="query",
        transform=None
    )

    gallery_dataset = Market1501Dataset(
        root=data_root,
        split="gallery",
        transform=None
    )

    stats = {
        "train_images": len(train_dataset),
        "train_identities": train_dataset.get_num_classes(),
        "train_cameras": len(set(train_dataset.camera_ids)),
        "query_images": len(query_dataset),
        "query_identities": query_dataset.get_num_classes(),
        "gallery_images": len(gallery_dataset),
        "gallery_identities": gallery_dataset.get_num_classes(),
    }

    return stats


if __name__ == "__main__":
    """Test the dataset loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Market-1501 dataset loader")
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.DATA_ROOT,
        help="Path to Market-1501 dataset"
    )
    args = parser.parse_args()

    print("Testing Market-1501 dataset loader...")
    print(f"Data root: {args.data_root}\n")

    # Get dataset statistics
    try:
        stats = get_dataset_stats(args.data_root)
        print("Dataset Statistics:")
        print("-" * 50)
        for key, value in stats.items():
            print(f"  {key:20s}: {value:6d}")
        print("-" * 50)

        # Test dataloader creation
        print("\nCreating dataloaders...")
        train_loader, query_loader, gallery_loader, num_classes = create_dataloaders(
            data_root=args.data_root,
            batch_size=32,
            num_workers=0,
        )

        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Query loader: {len(query_loader)} batches")
        print(f"✓ Gallery loader: {len(gallery_loader)} batches")
        print(f"✓ Number of classes: {num_classes}")

        # Test loading a batch
        print("\nLoading a test batch...")
        images, person_ids, camera_ids = next(iter(train_loader))
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Person IDs: {person_ids[:5].tolist()}...")
        print(f"✓ Camera IDs: {camera_ids[:5].tolist()}...")

        print("\n✓ All tests passed!")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run: python download_dataset.py")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
