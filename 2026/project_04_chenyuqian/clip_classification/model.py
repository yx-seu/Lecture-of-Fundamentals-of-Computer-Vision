"""CLIPZeroShotClassifier — wraps CLIPModel for zero-shot classification.

Key design: manually extract image & text features, L2-normalize,
compute cosine similarity, argmax — making the two-tower architecture
and shared latent space explicit and explainable.
"""

import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPZeroShotClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        if self._device == "cpu":
            print("[model] CUDA not available, falling back to CPU.")

        print(f"[model] Loading {model_name} ...")
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)
        print(f"[model] Loaded on {self._device}.")

    @property
    def device(self):
        return self._device

    @property
    def feature_dim(self):
        return self._model.config.projection_dim

    @torch.no_grad()
    def encode_images(self, images):
        """Encode a list of PIL images into L2-normalized feature vectors.

        Args:
            images: list of PIL.Image.

        Returns:
            Tensor of shape (len(images), feature_dim), L2-normalized.
        """
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        features = self._model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def encode_texts(self, texts):
        """Encode a list of text prompts into L2-normalized feature vectors.

        Args:
            texts: list of str.

        Returns:
            Tensor of shape (len(texts), feature_dim), L2-normalized.
        """
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        features = self._model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def build_text_features(self, class_names, prompts, ensembling=True):
        """Build text feature vectors for each class.

        Args:
            class_names: list of class name strings.
            prompts: list of (class_idx, template, prompt_text) tuples
                     from prompt_templates.generate_prompts().
            ensembling: if True, average features across prompts for each class.

        Returns:
            Tensor of shape (num_classes, feature_dim), L2-normalized.
        """
        texts = [p[2] for p in prompts]
        text_features = self.encode_texts(texts)  # (total_prompts, dim)

        if not ensembling:
            return text_features

        num_classes = len(class_names)
        # Group features by class_idx and average
        class_features = []
        for c in range(num_classes):
            mask = torch.tensor(
                [p[0] == c for p in prompts], device=self._device
            )
            c_feats = text_features[mask]  # prompts for this class
            c_mean = c_feats.mean(dim=0)   # average
            c_mean = c_mean / c_mean.norm(dim=-1, keepdim=True)  # re-normalize
            class_features.append(c_mean)

        return torch.stack(class_features, dim=0)  # (num_classes, dim)

    @torch.no_grad()
    def predict(self, images, text_features):
        """Predict class labels for a batch of images.

        Args:
            images: list of PIL.Image.
            text_features: (num_classes, feature_dim) L2-normalized tensor.

        Returns:
            Tensor of shape (len(images),) with predicted class indices.
        """
        image_features = self.encode_images(images)   # (batch, dim)
        similarity = image_features @ text_features.T  # (batch, num_classes)
        predictions = similarity.argmax(dim=-1)
        return predictions

    @torch.no_grad()
    def predict_all(self, data_loader, text_features):
        """Predict class labels for the entire data loader.

        Args:
            data_loader: DataLoader yielding (PIL_image, label) tuples.
            text_features: (num_classes, feature_dim) L2-normalized tensor.

        Returns:
            all_preds: (N,) tensor of predicted class indices.
            all_labels: (N,) tensor of ground-truth labels.
        """
        all_preds = []
        all_labels = []
        for images, labels in data_loader:
            # images from CIFAR-10 DataLoader are tensors (C,H,W), need PIL
            pil_images = []
            for img_tensor in images:
                if isinstance(img_tensor, torch.Tensor):
                    # CIFAR-10 returns tensors; convert to PIL
                    from torchvision.transforms import ToPILImage
                    pil_images.append(ToPILImage()(img_tensor))
                else:
                    pil_images.append(img_tensor)

            preds = self.predict(pil_images, text_features)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

        return torch.cat(all_preds), torch.cat(all_labels)
