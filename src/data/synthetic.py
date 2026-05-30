from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_synthetic_loaders(
    num_samples: int,
    image_size: int,
    batch_size: int,
    train_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_samples) % 2
    images = torch.rand((num_samples, 3, image_size, image_size), generator=generator)

    fake_mask = labels.view(-1, 1, 1, 1).float()
    images = (images + fake_mask * _checkerboard(num_samples, image_size) * 0.25).clamp(0, 1)

    dataset = TensorDataset(images, labels.long())
    train_size = int(num_samples * train_split)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def _checkerboard(num_samples: int, image_size: int) -> torch.Tensor:
    y = torch.arange(image_size).view(1, 1, image_size, 1)
    x = torch.arange(image_size).view(1, 1, 1, image_size)
    pattern = ((x + y) % 2).float()
    return pattern.repeat(num_samples, 3, 1, 1)
