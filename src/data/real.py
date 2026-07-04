from __future__ import annotations

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from pathlib import Path


def build_real_loaders(
    train_dir: str,
    test_dir: str,
    image_size: int,
    batch_size: int,
    valid_dir: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders for real datasets using torchvision's ImageFolder.
    
    Assumes standard directory structure:
    dir/
      real/
        img1.jpg...
      fake/
        img1.jpg...
    """
    # Standard transformation: resize and to tensor [0, 1]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    # If valid_dir is provided, we can concatenate it with train_dataset for more data
    if valid_dir and Path(valid_dir).exists():
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
        train_dataset = ConcatDataset([train_dataset, valid_dataset])
        print(f"[+] Combined train and valid sets. Total train samples: {len(train_dataset)}")
    else:
        print(f"[+] Train set samples: {len(train_dataset)}")

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    print(f"[+] Test set samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    return train_loader, test_loader
