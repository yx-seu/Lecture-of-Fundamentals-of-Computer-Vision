"""
DINOv2-based Semantic Segmentation Model with Multi-Head ADBA-Head.

Architecture:
    DINOv2 ViT-B/14 (registers) backbone
    ├── Hook @ Layer 3  → shallow geometric features
    └── Hook @ Layer 11 → self-attention matrix + coarse features
        (end-to-end gradient flow through ALL paths)

    Multi-Head ADBA-Head:
        - Multi-head attention diffusion (NO head averaging)
        - Group Convolution aligned with attention heads
        - Per-head independent subspace evolution before fusion
        - Deep fusion (2 stacked conv layers)
        - Exact 2× PixelShuffle → 7× bilinear = 14× upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"
EMBED_DIM = 768
NUM_HEADS = 12
NUM_REGISTERS = 4
NUM_PREFIX = 1 + NUM_REGISTERS  # CLS + 4 registers = 5


# ---------------------------------------------------------------------------
# Non-intrusive hook helpers
# ---------------------------------------------------------------------------

class AttentionCapture:
    """
    Capture self-attention matrix via forward hook on ``attn_drop``.

    Hooks onto ``attn_drop`` because its input is the post-softmax attention
    matrix — avoids monkey-patching and is robust to timm version changes.

    ``fused_attn`` is temporarily disabled so timm takes the explicit path
    that materializes the attention matrix.
    """

    def __init__(self, attn_module: nn.Module):
        self.attn_matrix: Optional[torch.Tensor] = None
        self._original_fused = getattr(attn_module, "fused_attn", True)
        attn_module.fused_attn = False
        self.hook = attn_module.attn_drop.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, args, output):
        # args[0] = softmax(QK^T / sqrt(d)) — keep gradient graph intact
        self.attn_matrix = args[0]  # (B, heads, N, N)

    def remove(self):
        self.hook.remove()


class FeatureCapture:
    """Forward hook to capture block output features."""

    def __init__(self, block: nn.Module):
        self.features: Optional[torch.Tensor] = None
        self.hook = block.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, args, output):
        self.features = output  # (B, N, embed_dim) — keep grad

    def remove(self):
        self.hook.remove()


# ---------------------------------------------------------------------------
# Multi-Head ADBA Seg Head
# ---------------------------------------------------------------------------

class MultiHeadADBAHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_classes: int = 51,  # 1 background + 50 ImageNet-S categories
        shallow_dim: int = 120,
        coarse_dim: int = 384,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coarse_dim = coarse_dim
        self.head_dim = coarse_dim // num_heads  # 384 / 12 = 32

        # ---------------------------------------------------------
        # 1. 扩散前的空间特征提取
        # Attention 最后的 proj 线性层已经混合了各 head 的信息，
        # 用全通道卷积让特征自由交互，不做分组限制。
        # ---------------------------------------------------------
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(embed_dim, coarse_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(coarse_dim),
            nn.ReLU(inplace=True)
        )

        self.shallow_conv = nn.Sequential(
            nn.Conv2d(embed_dim, shallow_dim, kernel_size=1),
            nn.BatchNorm2d(shallow_dim),
            nn.ReLU(inplace=True)
        )

        # ---------------------------------------------------------
        # 2. 扩散后的融合与渐进式上采样 (3×2× PixelShuffle = 8×, + bilinear)
        # ---------------------------------------------------------
        fusion_dim = coarse_dim + shallow_dim  # 384 + 120 = 504
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32×32 → 64×64  (256 → 256)
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 256 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Stage 2: 64×64 → 128×128  (256 → 128)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Stage 3: 128×128 → 256×256  (128 → 64)
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Final projection + light refinement at full resolution
        self.cls_proj = nn.Conv2d(64, num_classes, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        coarse_features: torch.Tensor,   # (B, N_total, embed_dim)
        attn_matrix: torch.Tensor,       # (B, heads, N_total, N_total)
        shallow_features: torch.Tensor,  # (B, N_total, embed_dim)
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        
        B = coarse_features.shape[0]
        P = patch_h * patch_w
        NUM_PREFIX = 5 # 1 CLS + 4 Registers

        # =========================================================
        # 步骤 A：提取纯 Patch 的注意力矩阵
        # =========================================================
        # A 维度: (B, H, P, P)
        A = attn_matrix[:, :, NUM_PREFIX:, NUM_PREFIX:]

        # =========================================================
        # 步骤 B：按你的思路 -> 序列转空间 -> 分组卷积
        # =========================================================
        x_coarse = coarse_features[:, NUM_PREFIX:, :] # (B, P, embed_dim)
        # 转换为二维特征图 (B, embed_dim, patch_h, patch_w)
        x_coarse_spatial = x_coarse.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)
        
        # 经过分组卷积提取 Head 专属的局部空间先验
        # M_spatial 维度: (B, coarse_dim, patch_h, patch_w)
        M_spatial = self.coarse_conv(x_coarse_spatial)

        # =========================================================
        # 步骤 C：数学重组为多头序列以进行扩散
        # =========================================================
        # 1. 展平空间维度: (B, coarse_dim, P)
        M_flat = M_spatial.reshape(B, self.coarse_dim, P)
        
        # 2. 将通道拆分给各个 Head，并正确转置到最后
        # (B, coarse_dim, P) -> (B, H, C/H, P) -> (B, H, P, C/H)
        M = M_flat.reshape(B, self.num_heads, self.head_dim, P).transpose(2, 3)

        # =========================================================
        # 步骤 D：多头空间内的独立注意力扩散
        # =========================================================
        # (B, H, P, P) @ (B, H, P, C/H) -> (B, H, P, C/H)
        M_diffused = torch.matmul(A, M)

        # =========================================================
        # 步骤 E：扩散完毕，完美还原为空间特征图
        # =========================================================
        # (B, H, P, C/H) -> (B, H, C/H, P)
        M_diffused = M_diffused.transpose(2, 3)
        # (B, H, C/H, P) -> (B, coarse_dim, P) -> (B, coarse_dim, patch_h, patch_w)
        M_diffused = M_diffused.reshape(B, self.coarse_dim, patch_h, patch_w)

        # =========================================================
        # 步骤 F：浅层桥接与深度融合
        # =========================================================
        x_shallow = shallow_features[:, NUM_PREFIX:, :].transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)
        shallow_map = self.shallow_conv(x_shallow)

        fused = torch.cat([M_diffused, shallow_map], dim=1)
        fused = self.fusion(fused)

        # =========================================================
        # 步骤 G：渐进式上采样 3×2× PixelShuffle + bilinear → refine
        # =========================================================
        # fused: (B, 256, Hp, Wp)  e.g. (B, 256, 32, 32)
        target_h = patch_h * 14
        target_w = patch_w * 14

        out = self.up1(fused)                # 2× : 32 → 64   (256ch → 256ch)
        out = self.up2(out)                  # 4× : 64 → 128  (256ch → 128ch)
        out = self.up3(out)                  # 8× : 128 → 256 (128ch → 64ch)
        out = F.interpolate(out, size=(target_h, target_w),
                           mode="bilinear", align_corners=False)  # → 448
        out = self.cls_proj(out)             # (B, 64, H, W) → (B, num_classes, H, W)
        out = self.refine(out)
        return out


# ---------------------------------------------------------------------------
# Full segmentation model
# ---------------------------------------------------------------------------

class DINOv2Seg(nn.Module):
    """
    DINOv2-backed semantic segmentation with Multi-Head ADBA-Head.

    Backbone  DINOv2 ViT-B/14 + 4 register tokens
    Head      MultiHeadADBAHead

    End-to-end trainable: gradients flow from segmentation loss through
    the multi-head attention-diffusion path back to Layer 11 Q,K,V.
    """

    def __init__(
        self,
        num_classes: int = 51,
        img_size: int = 448,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        assert img_size % 14 == 0, f"img_size must be multiple of 14, got {img_size}"
        self.img_size = img_size
        self.patch_h = img_size // 14
        self.patch_w = img_size // 14

        # ---- Backbone ----
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        assert hasattr(self.backbone.blocks[0].attn, "attn_drop"), \
            "Expected timm Attention with attn_drop"

        # ---- Hooks ----
        self.attn_capture = AttentionCapture(self.backbone.blocks[10].attn)
        self.feat_capture = FeatureCapture(self.backbone.blocks[2])

        # ---- Freeze ----
        if freeze_backbone:
            self._freeze_backbone()

        # ---- Multi-Head ADBA Head ----
        self.seg_head = MultiHeadADBAHead(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_classes=num_classes,
        )

    def _freeze_backbone(self):
        total_blocks = len(self.backbone.blocks)
        freeze_until = int(total_blocks * 0.8)

        for i, block in enumerate(self.backbone.blocks):
            for param in block.parameters():
                param.requires_grad = i >= freeze_until

        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        for attr in ("cls_token", "pos_embed", "reg_token"):
            if hasattr(self.backbone, attr):
                t = getattr(self.backbone, attr)
                if t is not None:
                    t.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict:
        coarse_feat = self.backbone.forward_features(x)
        attn = self.attn_capture.attn_matrix
        shallow = self.feat_capture.features

        logits = self.seg_head(
            coarse_features=coarse_feat,
            attn_matrix=attn,
            shallow_features=shallow,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
        )

        return {
            "logits": logits,
            "attn": attn,
            "shallow": shallow,
            "coarse": coarse_feat,
        }

    def remove_hooks(self):
        self.attn_capture.remove()
        self.feat_capture.remove()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    pretrained = "--pretrained" in sys.argv

    print("=" * 60)
    print("DINOv2 + MultiHead-ADBA-Head  Model Test")
    print("=" * 60)

    print(f"\n[1] Building model (pretrained={pretrained})...")
    model = DINOv2Seg(num_classes=51, img_size=448,
                      freeze_backbone=True, pretrained=pretrained)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total params:     {total / 1e6:.2f}M")
    print(f"    Trainable params: {trainable / 1e6:.2f}M")

    print("\n[2] Forward pass  (1, 3, 448, 448)...")
    x = torch.randn(1, 3, 448, 448)
    model.eval()
    with torch.no_grad():
        out = model(x)

    for k in ("logits", "attn", "shallow", "coarse"):
        print(f"    {k:10s}  {list(out[k].shape)}")

    assert out["logits"].shape == (1, 51, 448, 448), \
        f"Expected (1,51,448,448), got {out['logits'].shape}"
    print("\n[3] ✓ Output shape correct: (1, 51, 448, 448)")

    print(f"\n[4] Attention matrix: {list(out['attn'].shape)}")
    row_sums = out['attn'].sum(-1)
    print(f"    Σ per query: [{row_sums.min():.4f}, {row_sums.max():.4f}]  (≈1)")

    print("\n[5] Gradient flow (end-to-end)...")
    model.train()
    x = torch.randn(2, 3, 448, 448)
    out = model(x)
    loss = out["logits"].mean()
    loss.backward()

    grad_params = [n for n, p in model.named_parameters() if p.grad is not None]
    no_grad_params = [n for n, p in model.named_parameters()
                      if p.requires_grad and p.grad is None]
    print(f"    Params with grads:  {len(grad_params)}")
    print(f"    Trainable, no grad: {len(no_grad_params)}")
    l11_attn_grads = [n for n in grad_params if "blocks.10.attn" in n]
    print(f"    Layer-11 attn grads: {len(l11_attn_grads)}  (multi-head Q,K,V)")

    model.remove_hooks()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
