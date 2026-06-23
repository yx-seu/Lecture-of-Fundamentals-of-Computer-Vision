import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class BasicImageProcessor:
    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean or [0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std or [0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __call__(self, images, return_tensors="pt"):
        image = images.convert("RGB")
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        if return_tensors == "pt":
            tensor = tensor.unsqueeze(0)
        return {"pixel_values": tensor}

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "processor_type": "BasicImageProcessor",
            "mean": self.mean.flatten().tolist(),
            "std": self.std.flatten().tolist(),
        }
        with (save_dir / "preprocessor_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_dir):
        path = Path(model_dir) / "preprocessor_config.json"
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(mean=cfg.get("mean"), std=cfg.get("std"))


class OwnSegformerOverlapPatchEmbeddings(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)
        return x, height, width


class OwnSegformerEfficientSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, sr_ratio):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sr_ratio = sr_ratio
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio)
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, height, width):
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        key_value_states = hidden_states
        if self.sr_ratio > 1:
            batch_size, seq_len, channels = hidden_states.shape
            key_value_states = hidden_states.permute(0, 2, 1).reshape(batch_size, channels, height, width)
            key_value_states = self.sr(key_value_states)
            key_value_states = key_value_states.reshape(batch_size, channels, -1).permute(0, 2, 1)
            key_value_states = self.layer_norm(key_value_states)

        key_layer = self.transpose_for_scores(self.key(key_value_states))
        value_layer = self.transpose_for_scores(self.value(key_value_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(new_context_shape)


class OwnSegformerSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class OwnSegformerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, sr_ratio):
        super().__init__()
        self.self = OwnSegformerEfficientSelfAttention(hidden_size, num_heads, sr_ratio)
        self.output = OwnSegformerSelfOutput(hidden_size)

    def forward(self, hidden_states, height, width):
        return self.output(self.self(hidden_states, height, width))


class OwnSegformerDWConv(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dwconv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, channels = hidden_states.shape
        x = hidden_states.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class OwnSegformerMixFFN(nn.Module):
    def __init__(self, hidden_size, mlp_ratio):
        super().__init__()
        intermediate_size = hidden_size * mlp_ratio
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dwconv = OwnSegformerDWConv(intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class OwnSegformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, sr_ratio, mlp_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = OwnSegformerAttention(hidden_size, num_heads, sr_ratio)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.mlp = OwnSegformerMixFFN(hidden_size, mlp_ratio)

    def forward(self, hidden_states, height, width):
        hidden_states = hidden_states + self.attention(self.layer_norm_1(hidden_states), height, width)
        hidden_states = hidden_states + self.mlp(self.layer_norm_2(hidden_states), height, width)
        return hidden_states


class OwnSegformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_sizes = cfg["hidden_sizes"]
        patch_sizes = cfg["patch_sizes"]
        strides = cfg["strides"]
        depths = cfg["depths"]
        num_heads = cfg["num_attention_heads"]
        sr_ratios = cfg["sr_ratios"]
        mlp_ratios = cfg["mlp_ratios"]

        in_channels = [cfg.get("num_channels", 3)] + hidden_sizes[:-1]
        self.patch_embeddings = nn.ModuleList(
            [
                OwnSegformerOverlapPatchEmbeddings(in_ch, hidden, patch, stride)
                for in_ch, hidden, patch, stride in zip(in_channels, hidden_sizes, patch_sizes, strides)
            ]
        )
        self.block = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        OwnSegformerLayer(hidden_sizes[i], num_heads[i], sr_ratios[i], mlp_ratios[i])
                        for _ in range(depths[i])
                    ]
                )
                for i in range(len(hidden_sizes))
            ]
        )
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in hidden_sizes])

    def forward(self, pixel_values):
        hidden_states = pixel_values
        all_hidden_states = []
        for idx, patch_embed in enumerate(self.patch_embeddings):
            hidden_states, height, width = patch_embed(hidden_states)
            for layer in self.block[idx]:
                hidden_states = layer(hidden_states, height, width)
            hidden_states = self.layer_norm[idx](hidden_states)
            batch_size, seq_len, channels = hidden_states.shape
            feature_map = hidden_states.transpose(1, 2).reshape(batch_size, channels, height, width)
            all_hidden_states.append(feature_map)
            hidden_states = feature_map
        return all_hidden_states


class OwnSegformerBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = OwnSegformerEncoder(cfg)

    def forward(self, pixel_values):
        return self.encoder(pixel_values)


class OwnSegformerLinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states):
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states.transpose(1, 2).reshape(batch_size, -1, height, width)


class OwnSegformerDecodeHead(nn.Module):
    def __init__(self, hidden_sizes, decoder_hidden_size, num_labels, dropout):
        super().__init__()
        self.linear_c = nn.ModuleList(
            [OwnSegformerLinearProjection(hidden_size, decoder_hidden_size) for hidden_size in hidden_sizes]
        )
        self.linear_fuse = nn.Conv2d(decoder_hidden_size * len(hidden_sizes), decoder_hidden_size, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(decoder_hidden_size, num_labels, kernel_size=1)

    def forward(self, hidden_states):
        target_size = hidden_states[0].shape[-2:]
        projected = []
        for feature, linear in zip(hidden_states, self.linear_c):
            feature = linear(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        fused = torch.cat(projected, dim=1)
        fused = self.linear_fuse(fused)
        fused = self.batch_norm(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)


class OwnSegformerForImageNetS(nn.Module):
    def __init__(self, cfg, num_labels, id2label, label2id, ignore_index=255):
        super().__init__()
        self.cfg = cfg
        self.num_labels = int(num_labels)
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = label2id
        self.ignore_index = int(ignore_index)
        self.segformer = OwnSegformerBackbone(cfg)
        self.decode_head = OwnSegformerDecodeHead(
            hidden_sizes=cfg["hidden_sizes"],
            decoder_hidden_size=cfg.get("decoder_hidden_size", 256),
            num_labels=self.num_labels,
            dropout=cfg.get("classifier_dropout_prob", 0.1),
        )

    @classmethod
    def from_ade_checkpoint(cls, model_dir, config):
        model_dir = Path(model_dir)
        with (model_dir / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = cls(
            cfg=cfg,
            num_labels=config["model"]["num_labels"],
            id2label=config["model"]["id2label"],
            label2id=config["model"]["label2id"],
            ignore_index=config["data"]["ignore_index"],
        )
        state_path = model_dir / "model.safetensors"
        state = load_file(str(state_path)) if state_path.exists() else torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        own_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in own_state and own_state[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(
            f"[OwnSegFormer] loaded {len(filtered)} pretrained tensors; "
            f"skipped {len(state) - len(filtered)} tensors; missing {len(missing)} tensors."
        )
        return model

    @classmethod
    def from_pretrained(cls, model_dir, device=None):
        model_dir = Path(model_dir)
        with (model_dir / "own_segformer_config.json").open("r", encoding="utf-8") as f:
            saved = json.load(f)
        model = cls(
            cfg=saved["segformer_config"],
            num_labels=saved["num_labels"],
            id2label={int(k): v for k, v in saved["id2label"].items()},
            label2id=saved["label2id"],
            ignore_index=saved["ignore_index"],
        )
        state = torch.load(model_dir / "pytorch_model.bin", map_location=device or "cpu")
        model.load_state_dict(state)
        return model

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
        saved = {
            "architecture": "OwnSegformerForImageNetS",
            "segformer_config": self.cfg,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "ignore_index": self.ignore_index,
        }
        with (save_dir / "own_segformer_config.json").open("w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)

    def forward(self, pixel_values, labels=None):
        hidden_states = self.segformer(pixel_values)
        logits = self.decode_head(hidden_states)
        loss = None
        if labels is not None:
            resized_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(resized_logits, labels, ignore_index=self.ignore_index)
        return SimpleNamespace(loss=loss, logits=logits)


def build_own_segformer_processor():
    return BasicImageProcessor()
