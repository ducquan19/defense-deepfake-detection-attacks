import torch
import torch.nn as nn
from transformers import AutoModel, Dinov2Config
from core.base import BaseDetector
class MultiAspectFusionHead(nn.Module):
    """
    Đầu giải mã phân loại đa khía cạnh (Multi-Aspect Classification Head).
    Thiết kế theo chuẩn hội nghị CVPR 2026, khai thác triệt để 
    các biểu diễn không gian được chưng cất từ Register Tokens.
    """
    def __init__(self, embed_dim: int, projection_dim: int = 512, dropout_rate: float = 0.3):
        super(MultiAspectFusionHead, self).__init__()
        
        self.cls_proj = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU()
        )
        
        self.reg_proj = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU()
        )
        
        self.patch_proj = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU()
        )
        
        fusion_dim = projection_dim * 3
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # THAY ĐỔI 1: Trả về 2 classes thay vì 1 class để khớp với BaseDetector
            nn.Linear(1024, 2) 
        )

    def forward(self, cls_token, reg_tokens, patch_tokens):
        cls_feat = self.cls_proj(cls_token)
        reg_feat = self.reg_proj(reg_tokens.mean(dim=1))
        patch_feat = self.patch_proj(patch_tokens.mean(dim=1))
        
        fused_features = torch.cat([cls_feat, reg_feat, patch_feat], dim=-1)
        logits = self.classifier(fused_features)
        return logits

# THAY ĐỔI 2: Kế thừa BaseDetector
class DinoMACForDeepfakeDetection(BaseDetector):
    """
    Mô hình nhận diện ảnh Deepfake toàn diện kết hợp xương sống DINOv2-Large 
    (tích hợp Registers) và hệ thống dự báo Multi-Aspect.
    """
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-with-registers-large", 
        freeze_backbone: bool = False,
        use_stochastic_depth: bool = True
    ):
        super().__init__()
        
        config = Dinov2Config.from_pretrained(model_name)
        if use_stochastic_depth:
            config.drop_path_rate = 0.1 
            
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.embed_dim = config.hidden_size
        self.num_registers = 4 
        
        self.mac_head = MultiAspectFusionHead(embed_dim=self.embed_dim)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Tên tham số đổi từ pixel_values thành images để khớp với BaseDetector
        outputs = self.backbone(pixel_values=images)
        hidden_states = outputs.last_hidden_state
        
        cls_token = hidden_states[:, 0, :]
        reg_tokens = hidden_states[:, 1:1+self.num_registers, :]
        patch_tokens = hidden_states[:, 1+self.num_registers:, :]
        
        logits = self.mac_head(cls_token, reg_tokens, patch_tokens)
        return logits
