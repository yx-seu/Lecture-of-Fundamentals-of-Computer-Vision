import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive


IMAGENETTE_160_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


class PatchEmbedding(nn.Module):
    """把图像切成 patch，并把每个 patch 投影成一个 token。"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size 必须能被 patch_size 整除")

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)          # (B, embed_dim, 14, 14)
        x = x.flatten(2)          # (B, embed_dim, 196)
        x = x.transpose(1, 2)     # (B, 196, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """从零实现的多头自注意力。"""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """Transformer block 中的两层前馈网络。"""

    def __init__(self, embed_dim: int = 256, mlp_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropPath(nn.Module):
    """随机丢弃整条残差路径 (Stochastic Depth)，防止模型过度依赖特定层。"""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor_()
        return x / keep_prob * random_tensor


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer encoder block。"""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """课程作业要求的 ViT 分类模型。"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # 每层 DropPath 概率线性递增：从 0 到 drop_path
        drop_path_rates = [
            drop_path * (i / (depth - 1)) if depth > 1 else drop_path
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, drop_path_rates[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_transforms(
    img_size: int = 224,
    rand_augment: bool = False,
    random_erasing: float = 0.0,
) -> Tuple[transforms.Compose, transforms.Compose]:
    train_steps = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if rand_augment:
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    if random_erasing > 0.0:
        train_steps.append(transforms.RandomErasing(p=random_erasing))

    train_transform = transforms.Compose(train_steps)
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return train_transform, val_transform


def create_imagefolder_loaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    rand_augment: bool = False,
    random_erasing: float = 0.0,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"没有找到 ImageFolder 数据目录。需要: {train_dir} 和 {val_dir}"
        )

    train_transform, val_transform = build_transforms(
        img_size,
        rand_augment=rand_augment,
        random_erasing=random_erasing,
    )
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("train 和 val 的类别目录不一致")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset.classes


def download_imagenette160(data_dir: str = "./data") -> str:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    dataset_root = data_path / "imagenette2-160"
    if dataset_root.is_dir():
        print(f"已找到数据集: {dataset_root}")
        return str(dataset_root)

    print("开始下载 Imagenette 160px 数据集（10 个 ImageNet 类）...")
    download_and_extract_archive(
        IMAGENETTE_160_URL,
        download_root=str(data_path),
        filename="imagenette2-160.tgz",
        remove_finished=False,
    )
    return str(dataset_root)


def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100.0


def topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> List[float]:
    max_k = min(max(topk), logits.size(1))
    predictions = logits.topk(max_k, dim=1).indices
    correct = predictions.eq(labels.view(-1, 1))
    results = []
    for k in topk:
        effective_k = min(k, logits.size(1))
        accuracy = correct[:, :effective_k].any(dim=1).float().mean().item() * 100.0
        results.append(accuracy)
    return results


def validate_class_count(class_names: List[str], expected_num_classes: int) -> None:
    if expected_num_classes > 0 and len(class_names) != expected_num_classes:
        raise ValueError(
            f"数据集包含 {len(class_names)} 类，但要求 {expected_num_classes} 类。"
        )


def create_criterion(label_smoothing: float = 0.0) -> nn.CrossEntropyLoss:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing 必须在 [0, 1) 范围内")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    min_lr: float = 1e-5,
    warmup_epochs: int = 0,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )
        if warmup_epochs > 0:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-3,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    base_scheduler,
                ],
                milestones=[warmup_epochs],
            )
        return base_scheduler
    raise ValueError(f"不支持的 scheduler: {scheduler_name}")


def load_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: str,
    expected_class_names: List[str],
    device: torch.device,
) -> List[str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_class_names = list(checkpoint["class_names"])
    if checkpoint_class_names != list(expected_class_names):
        raise ValueError(
            "checkpoint 类别顺序与当前数据集不一致，不能安全继续训练"
        )
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint_class_names


def save_training_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    best_acc: float,
    class_names: List[str],
    args: dict,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_acc": best_acc,
            "class_names": class_names,
            "args": args,
        },
        checkpoint_path,
    )


def load_training_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    expected_class_names: List[str],
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_class_names = list(checkpoint["class_names"])
    if checkpoint_class_names != list(expected_class_names):
        raise ValueError("resume checkpoint 类别顺序与当前数据集不一致")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_acc", 0.0))


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对一批图片做 MixUp：按随机比例混合两张图及其标签。

    强制模型不做死记硬背，而是理解更底层的视觉特征。
    """
    if alpha <= 0.0:
        return images, torch.nn.functional.one_hot(labels, num_classes).float()

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    lam = max(lam, 1.0 - lam)  # 保证 lam >= 0.5，混合以主图为主

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    index_onehot = torch.nn.functional.one_hot(labels[index], num_classes).float()
    mixed_labels = lam * labels_onehot + (1.0 - lam) * index_onehot

    return mixed_images, mixed_labels


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对一批图片做 CutMix：从另一张图切一块贴到当前图上。

    与 MixUp 互补 —— MixUp 全局混合，CutMix 局部替换。
    """
    if alpha <= 0.0:
        return images, torch.nn.functional.one_hot(labels, num_classes).float()

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())

    batch_size, _, H, W = images.shape
    index = torch.randperm(batch_size, device=images.device)

    # 根据 lam 决定切块大小
    cut_rat = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # 随机切块位置
    cy = torch.randint(0, H - cut_h + 1, (1,), device=images.device).item()
    cx = torch.randint(0, W - cut_w + 1, (1,), device=images.device).item()

    # 粘贴
    mixed_images = images.clone()
    mixed_images[:, :, cy:cy + cut_h, cx:cx + cut_w] = images[index, :, cy:cy + cut_h, cx:cx + cut_w]

    # 实际混合比例（按面积算）
    lam_actual = 1.0 - (cut_h * cut_w) / (H * W)

    labels_onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    index_onehot = torch.nn.functional.one_hot(labels[index], num_classes).float()
    mixed_labels = lam_actual * labels_onehot + (1.0 - lam_actual) * index_onehot

    return mixed_images, mixed_labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_classes: int = 10,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    max_batches: Optional[int] = None,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp / CutMix：每批随机选一种，防止死记硬背
        targets = labels
        if use_mixup and use_cutmix:
            if torch.rand(1).item() < 0.5:
                images, targets = mixup_batch(images, labels, alpha=mixup_alpha, num_classes=num_classes)
            else:
                images, targets = cutmix_batch(images, labels, alpha=cutmix_alpha, num_classes=num_classes)
        elif use_mixup:
            images, targets = mixup_batch(images, labels, alpha=mixup_alpha, num_classes=num_classes)
        elif use_cutmix:
            images, targets = cutmix_batch(images, labels, alpha=cutmix_alpha, num_classes=num_classes)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            if use_mixup or use_cutmix:
                # 混合模式：用 soft label 算交叉熵
                loss = - (targets * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx}/{len(loader)} "
                f"Loss {loss.item():.4f}"
            )
        if max_batches is not None and batch_idx >= max_batches:
            break

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if max_batches is not None and batch_idx >= max_batches:
            break

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def run_forward_demo(device: torch.device) -> None:
    model = VisionTransformer(num_classes=1000).to(device)
    model.eval()
    print(f"1000 类 ViT 参数量: {count_parameters(model) / 1e6:.2f} M")

    with torch.no_grad():
        dummy_img = torch.randn(2, 3, 224, 224, device=device)
        logits = model(dummy_img)
    print(f"1000 类前向输出 shape: {tuple(logits.shape)}")


def run_training(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = args.data_root
    if args.download_imagenette:
        data_root = download_imagenette160(args.data_dir)

    train_loader, val_loader, class_names = create_imagefolder_loaders(
        data_root=data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rand_augment=args.rand_augment,
        random_erasing=args.random_erasing,
    )
    print(f"类别数: {len(class_names)}")
    print(f"类别目录: {class_names}")
    print(f"训练图片数: {len(train_loader.dataset)}, 验证图片数: {len(val_loader.dataset)}")

    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=len(class_names),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        drop_path=args.drop_path,
    ).to(device)
    print(f"训练模型参数量: {count_parameters(model) / 1e6:.2f} M")
    if args.init_checkpoint:
        loaded_classes = load_checkpoint_weights(
            model,
            args.init_checkpoint,
            class_names,
            device,
        )
        print(f"已加载微调初始权重: {args.init_checkpoint}")
        print(f"checkpoint 类别: {loaded_classes}")

    criterion = create_criterion(args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_csv) if args.metrics_csv else None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"],
            )
            writer.writeheader()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            num_classes=len(class_names),
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            max_batches=args.max_train_batches,
            use_amp=not args.no_amp,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_val_batches,
            use_amp=not args.no_amp,
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR {optimizer.param_groups[0]['lr']:.6f} "
            f"Train Loss {train_loss:.4f} Train Acc {train_acc:.2f}% "
            f"Val Loss {val_loss:.4f} Val Acc {val_acc:.2f}%"
        )
        if metrics_path is not None:
            with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"],
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )

        if val_acc >= best_acc:
            best_acc = val_acc
            checkpoint_path = output_dir / "vit_imagenette10_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                    "best_acc": best_acc,
                },
                checkpoint_path,
            )
            print(f"保存最佳 checkpoint: {checkpoint_path} (Val Acc {best_acc:.2f}%)")

        if scheduler is not None:
            scheduler.step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从零实现 ViT，并在 Imagenette/ImageNet-10 上训练")
    parser.add_argument("--data-root", default="./data/imagenette2-160", help="ImageFolder 根目录，需包含 train/ 和 val/")
    parser.add_argument("--data-dir", default="./data", help="下载 Imagenette 时使用的数据目录")
    parser.add_argument("--download-imagenette", action="store_true", help="下载并解压 Imagenette 160px 数据集")
    parser.add_argument("--output-dir", default="./outputs", help="checkpoint 输出目录")
    parser.add_argument("--metrics-csv", default="", help="保存每个 epoch 指标的 CSV 路径")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="学习率线性预热的 epoch 数，0=不使用")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp Beta 分布参数，0=不使用")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0, help="CutMix Beta 分布参数，0=不使用。与 MixUp 配合时每批随机选一种")
    parser.add_argument("--drop-path", type=float, default=0.1, help="Stochastic Depth 的最大丢弃概率")
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--rand-augment", action="store_true", help="训练时启用 RandAugment 数据增强")
    parser.add_argument("--random-erasing", type=float, default=0.0, help="训练时 RandomErasing 概率，例如 0.25")
    parser.add_argument("--init-checkpoint", default=None, help="加载已有 checkpoint 作为微调起点")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="例如 cuda 或 cpu；默认自动选择")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA AMP 混合精度")
    parser.add_argument("--forward-only", action="store_true", help="只运行 1000 类前向 shape demo")
    parser.add_argument("--max-train-batches", type=int, default=None, help="调试时限制每个 epoch 的训练 batch 数")
    parser.add_argument("--max-val-batches", type=int, default=None, help="调试时限制每个 epoch 的验证 batch 数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_forward_demo(device)
    if not args.forward_only:
        run_training(args)


if __name__ == "__main__":
    main()
