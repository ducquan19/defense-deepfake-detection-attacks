"""Script to generate sample adversarial images for all attacks.

Usage:
    python scripts/generate_samples.py \
        --image-path /kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test/fake/0KD0J8VYU3.jpg \
        --model-ckpt models/tiny_cnn_robust_42_1783137099.pt \
        --config configs/experiment_full_robustness.yaml \
        --output-dir reports/samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from config import load_config
from attacks import ATTACK_REGISTRY
from models.tiny_cnn import TinyCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial samples for a single image.")
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the source image (e.g., from test/fake/).",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Ground truth label of the image (0 for fake, 1 for real).",
    )
    parser.add_argument(
        "--model-ckpt",
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_full_robustness.yaml",
        help="Path to the config file (to load attack hyperparameters).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/samples",
        help="Directory to save the generated samples.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load TinyCNN detector from a checkpoint."""
    model = TinyCNN().to(device)
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"[+] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[!] Checkpoint not found: {ckpt_path}. Using random weights.")
    
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    attack_cfg = config.values.get("attack", {})
    eval_attacks = attack_cfg.get("eval_attacks", [])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # Load model
    model = load_model(args.model_ckpt, device)

    # Load and transform image
    image_size = config.values.get("data", {}).get("image_size", 128)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    print(f"[+] Loading image: {img_path}")
    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device) # Shape: (1, 3, 128, 128)
    label_tensor = torch.tensor([args.label], dtype=torch.long).to(device)

    # Save original image
    save_image(img_tensor, output_dir / "original.jpg")
    print(f"[+] Saved original image to {output_dir / 'original.jpg'}")

    # Build and run attacks
    print("\n[+] Generating adversarial samples...")
    for atk_def in eval_attacks:
        name = atk_def.get("name")
        if name in ATTACK_REGISTRY:
            kwargs = {k: v for k, v in atk_def.items() if k != "name"}
            attack_instance = ATTACK_REGISTRY[name](**kwargs)
            
            print(f"    - Running {name} attack...")
            # For DeepFool, C&W, AutoAttack, it can be slightly slow even for 1 image, but acceptable.
            try:
                result = attack_instance.perturb(model, img_tensor, label_tensor)
                adv_img = result.adversarial_images
                perturbation = result.perturbations
                
                # Enhance perturbation visibility (normalize to 0-1 for visualization)
                # Perturbations are small, so we scale them up to be visible
                pert_vis = perturbation.abs()
                if pert_vis.max() > 0:
                    pert_vis = pert_vis / pert_vis.max()
                
                save_image(adv_img, output_dir / f"adv_{name}.jpg")
                save_image(pert_vis, output_dir / f"noise_{name}.jpg")
                print(f"      Saved adv_{name}.jpg and noise_{name}.jpg")
            except Exception as e:
                print(f"      [!] Error running {name}: {e}")
        else:
            print(f"    [!] Unknown attack '{name}' in config. Skipping.")

    print("\n[+] Done! All samples generated successfully.")


if __name__ == "__main__":
    main()
