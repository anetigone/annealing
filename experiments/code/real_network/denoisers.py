"""Local-only grayscale DRUNet/FFDNet. Architecture credits: see NOTICE.md."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.res(x)


class DRUNet(nn.Module):
    """Original bias-free gray UNetRes; original KAIR/DPIR state-dict names."""
    def __init__(self):
        super().__init__()
        self.m_head = nn.Conv2d(2, 64, 3, padding=1, bias=False)
        channels = (64, 128, 256, 512)
        for i, (left, right) in enumerate(zip(channels[:-1], channels[1:]), 1):
            down = [ResBlock(left) for _ in range(4)]
            down.append(nn.Conv2d(left, right, 2, stride=2, bias=False))
            setattr(self, f"m_down{i}", nn.Sequential(*down))
            up = [nn.ConvTranspose2d(right, left, 2, stride=2, bias=False)]
            up.extend(ResBlock(left) for _ in range(4))
            setattr(self, f"m_up{i}", nn.Sequential(*up))
        self.m_body = nn.Sequential(*(ResBlock(512) for _ in range(4)))
        self.m_tail = nn.Conv2d(64, 1, 3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_up3(self.m_body(x4) + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


class FFDNet(nn.Module):
    """Support original KAIR and the explicitly converted DeepInv HF layout."""
    def __init__(self, layout="deepinv"):
        super().__init__()
        self.layout = layout
        if layout == "kair":
            layers = []
            for i in range(15):
                layers.append(nn.Conv2d(5 if i == 0 else 64,
                                        4 if i == 14 else 64, 3, padding=1))
                if i != 14:
                    layers.append(nn.ReLU(inplace=True))
            self.model = nn.Sequential(*layers)
        elif layout == "deepinv":
            blocks = [nn.Sequential(nn.Conv2d(8, 64, 3, padding=1), nn.ReLU(inplace=True))]
            blocks.extend(nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                         nn.Identity(), nn.ReLU(inplace=True)) for _ in range(13))
            blocks.append(nn.Conv2d(64, 4, 3, padding=1))
            self.blocks = nn.Sequential(*blocks)
        else:
            raise ValueError(f"Unknown FFDNet layout: {layout}")

    def forward(self, x, sigma):
        if self.layout == "deepinv":
            noise = x.new_full((x.shape[0], 1, *x.shape[-2:]), sigma)
            x = self.blocks(F.pixel_unshuffle(torch.cat((x, noise), dim=1), 2))
        else:
            x = F.pixel_unshuffle(x, 2)
            noise = x.new_full((x.shape[0], 1, *x.shape[-2:]), sigma)
            x = self.model(torch.cat((x, noise), dim=1))
        return F.pixel_shuffle(x, 2)


class Denoiser:
    """One full-image forward per call; fixed replicate pad, no clipping or x8."""
    def __init__(self, model, name):
        self.model = model.eval().requires_grad_(False)
        self.name = name
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, x, sigma):
        if x.ndim != 4 or x.shape[1] != 1 or not 0 < sigma <= 50 / 255:
            raise ValueError("Expected NCHW grayscale and 0 < sigma <= 50/255")
        h, w = x.shape[-2:]
        multiple = 8 if self.name == "drunet" else 2
        padded = F.pad(x, (0, (-w) % multiple, 0, (-h) % multiple), mode="replicate")
        self.calls += 1
        if self.name == "drunet":
            noise = padded.new_full((x.shape[0], 1, *padded.shape[-2:]), sigma)
            result = self.model(torch.cat((padded, noise), dim=1))
        else:
            result = self.model(padded, sigma)
        return result[..., :h, :w]


def load_denoiser(spec, device):
    path = Path(spec["weights"])
    if not path.is_file():
        raise FileNotFoundError(f"Local weights missing: {path}. See experiments/real_network.md")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "params"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty PyTorch state dictionary")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    name = spec["name"]
    if name == "drunet":
        model = DRUNet()
        layout = "original_bias_free"
    elif name == "ffdnet":
        layout = "deepinv" if "blocks.0.0.weight" in state else "kair"
        model = FFDNet(layout)
    else:
        raise ValueError(f"Unsupported network: {name}")
    model.load_state_dict(state, strict=True)
    denoiser = Denoiser(model.to(device=device, dtype=torch.float32), name)
    denoiser.layout = layout
    return denoiser
