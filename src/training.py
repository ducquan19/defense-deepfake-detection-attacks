from __future__ import annotations

import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from torchvision import transforms
from tqdm import tqdm

# DinoMACForDeepfakeDetection is imported lazily inside run_dino_training()
# to avoid triggering a HuggingFace model download at import time.

# =========================================================================
# 1. Hàm huấn luyện cơ bản (Dành cho pipeline hiện tại để không bị lỗi)
# =========================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(len(loader), 1)

# =========================================================================
# 2. Các khối Augmentation và Dataset cho DINO-MAC
# =========================================================================
class GaussianNoiseInjection(nn.Module):
    """Mô-đun tiêm nhiễu phân phối chuẩn vào không gian Tensor"""
    def __init__(self, p=0.5, variance=0.05):
        super().__init__()
        self.p = p
        self.variance = variance

    def forward(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.variance
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

class RealWorldDegradationTransform:
    def __init__(self, target_size=224):
        assert target_size % 14 == 0, "Target size must perfectly align with patch scale (mod 14)."
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            transforms.ToTensor(),
            GaussianNoiseInjection(p=0.2, variance=0.03),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transform(img)

class DeepfakeForensicsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Ảnh tensor dummy để test
        img = torch.rand(3, 224, 224) 
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# =========================================================================
# 3. Phân hệ Tối ưu hóa và Đánh giá nâng cao (Advanced Training Loops)
# =========================================================================
def run_training_epoch_advanced(model, loader, criterion, optimizer, scaler, scheduler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) 
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            logits = model(inputs) 
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    return epoch_loss, epoch_auc

@torch.no_grad()
def run_validation_epoch_advanced(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in tqdm(loader, desc="Validation"):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            logits = model(inputs)
            loss = criterion(logits, labels)
            
        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    epoch_acc = accuracy_score(all_labels, binary_preds)
    return epoch_loss, epoch_auc, epoch_acc

# =========================================================================
# 4. Hàm main thực thi huấn luyện DINO-MAC 
# =========================================================================
def run_dino_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lazy import: only download/load the HuggingFace model when this function is called
    from models.dino_mac import DinoMACForDeepfakeDetection

    # Định nghĩa thư mục lưu theo yêu cầu là thư mục models/ ở root
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = DinoMACForDeepfakeDetection(
        model_name="facebook/dinov2-with-registers-large",
        freeze_backbone=False,
        use_stochastic_depth=True
    ).to(device)
    
    train_dataset = DeepfakeForensicsDataset(["train.jpg"]*1000, [0, 1]*500, transform=RealWorldDegradationTransform())
    val_dataset = DeepfakeForensicsDataset(["val.jpg"]*200, [0, 1]*100, transform=RealWorldDegradationTransform())
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.mac_head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ])
    
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*epochs, eta_min=1e-6)
    scaler = GradScaler()
    best_auc = 0.0
    
    print(">>> KHỞI ĐỘNG CHU TRÌNH HUẤN LUYỆN DINO-MAC <<<")
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        t_loss, t_auc = run_training_epoch_advanced(model, train_loader, criterion, optimizer, scaler, scheduler, device)
        v_loss, v_auc, v_acc = run_validation_epoch_advanced(model, val_loader, criterion, device)
        
        print(f"Train | Loss: {t_loss:.4f} - AUC: {t_auc:.4f}")
        print(f"Val   | Loss: {v_loss:.4f} - AUC: {v_auc:.4f} - Acc: {v_acc:.4f}")
        
        if v_auc > best_auc:
            best_auc = v_auc
            chkpt_path = os.path.join(checkpoint_dir, f"dinomac_best_ep{epoch}_auc{best_auc:.4f}.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
            }, chkpt_path)
            print(f"[*] Checkpoint mới được kết xuất tại: {chkpt_path}")

if __name__ == "__main__":
    run_dino_training()
