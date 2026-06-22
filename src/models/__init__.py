"""Deepfake detector model definitions."""
from .tiny_cnn import TinyCNN
from .dino_mac import DinoMACForDeepfakeDetection

__all__ = ["TinyCNN", "DinoMACForDeepfakeDetection"]
