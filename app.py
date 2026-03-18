import os

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        s = x.mean(dim=1)  # [B, C]
        w = self.fc(s).unsqueeze(1)  # [B, 1, C]
        return x * w

class ViTWithSE(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        embed_dim = self.vit.embed_dim

        self.patch_embed_16 = self.vit.patch_embed
        self.patch_embed_32 = timm.layers.PatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=embed_dim)

        self.cls_fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.se = SEBlock(embed_dim, reduction=32)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(embed_dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 16×16 PATCH PATH (WITH POSITIONAL EMBEDDING)
        x16 = self.patch_embed_16(x)  # [B, N16, C] (no CLS yet)
        x16 = self.vit._pos_embed(x16)  # adds CLS + pos embed => [B, 1+N16, C]

        # 32×32 PATCH PATH (NO POSITIONAL EMBEDDING)
        # IMPORTANT: match your training script exactly.
        # PatchEmbed returns only patch tokens [B, N32, C]; you fused using x32[:, 0]
        # (i.e., the first patch token), not an explicit CLS token.
        x32 = self.patch_embed_32(x)  # [B, N32, C]

        # CLS TOKEN FUSION
        cls16 = x16[:, 0]  # [B, C]
        cls32 = x32[:, 0]  # [B, C]
        cls_fused = self.cls_fusion(torch.cat([cls16, cls32], dim=-1))  # [B, C]

        # Replace CLS token early (keep 16×16 patch tokens)
        x = torch.cat([cls_fused.unsqueeze(1), x16[:, 1:]], dim=1)  # [B, 1+N16, C]

        # SE ON PATCH TOKENS
        cls_tok = x[:, :1]
        patch_tok = self.se(x[:, 1:])
        x = torch.cat([cls_tok, patch_tok], dim=1)

        # TRANSFORMER ENCODER
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return self.head(x[:, 0])


def build_preprocess() -> transforms.Compose:
    # As requested:
    # - Resize 224x224
    # - Match training val_transforms: PILToTensor -> float32 -> ImageNet normalize
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights/model file not found at '{weights_path}'. Place it next to app.py "
            "or set the WEIGHTS_PATH environment variable."
        )

    model = ViTWithSE().to(device)
    model.eval()

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    if not isinstance(state, dict):
        raise TypeError(
            "Expected a state_dict/checkpoint dict in your .pth. "
            f"Got type: {type(state)}. If you saved a TorchScript model, use a TorchScript loader instead."
        )

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=True)
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "vit_final_full_good_SE+DUAL_SCALE_4_epochs.pth")
PREPROCESS = build_preprocess()
MODEL = load_model(WEIGHTS_PATH, DEVICE)


@torch.inference_mode()
def predict(image: Image.Image):
    if image is None:
        return {"Real": 0.0, "Fake": 0.0}

    image = image.convert("RGB")
    x = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    logits = MODEL(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().float().tolist()
    return {"Real": float(probs[0]), "Fake": float(probs[1])}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="DeepFake Detector",
    description="Upload a face image to detect whether it's real or fake.",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()

