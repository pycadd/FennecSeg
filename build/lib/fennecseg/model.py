"""Model package: a folder with model.json (manifest) + weights.safetensors.

The manifest carries everything inference needs (architecture, label table, spacing, intensity
normalisation, default window settings), so the code never hard-codes organs.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

FORMAT = 1


def build_net(arch: dict) -> nn.Module:
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    if arch["type"] != "ResidualEncoderUNet":
        raise ValueError(f"unknown architecture {arch['type']}")
    n = len(arch["features"])
    return ResidualEncoderUNet(
        input_channels=1, n_stages=n, features_per_stage=arch["features"], conv_op=nn.Conv3d,
        kernel_sizes=[3] * n, strides=[1] + [2] * (n - 1), n_blocks_per_stage=arch["blocks"],
        num_classes=arch["num_classes"], n_conv_per_stage_decoder=[arch["decoder_convs"]] * (n - 1),
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=False)


@dataclass
class Model:
    net: nn.Module
    manifest: dict
    path: Path

    @property
    def labels(self) -> dict:
        """{label id (int): organ name}; 0 is background."""
        return {int(k): v for k, v in self.manifest["labels"].items()}

    @property
    def spacing(self):
        return tuple(self.manifest["input"]["spacing_mm"])

    def describe(self) -> str:
        m = self.manifest
        s = [f"{m['name']} {m['version']}", f"  {len(self.labels)} structures: {', '.join(self.labels.values())}",
             f"  input: {m['input']['modality']} at {m['input']['spacing_mm']} mm"]
        for k, v in m.get("metrics", {}).items():
            s.append(f"  {k}: " + ", ".join(f"{a}={b}" for a, b in v.items()))
        return "\n".join(s)


HUB_REPO = os.environ.get("FENNECSEG_HUB_REPO", "amine0110/fennecseg")


def resolve(spec) -> Path:
    """A local package folder, or 'family' (latest) / 'family@x.y.z' (exact release) from the model hub.
    Downloads are cached by huggingface_hub; a private hub repo needs `hf auth login` once."""
    p = Path(spec)
    if p.exists():
        return p
    fam, _, ver = str(spec).partition("@")
    fam = fam.removeprefix("fennecseg-")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise FileNotFoundError(f"{spec} is not a folder, and huggingface_hub is not installed to download it") from e
    rev = f"{fam}-v{ver.removeprefix('v')}" if ver else "main"
    from huggingface_hub.utils import RevisionNotFoundError
    try:
        root = snapshot_download(HUB_REPO, revision=rev, allow_patterns=[f"{fam}/*"])
    except RevisionNotFoundError as e:
        raise FileNotFoundError(f"{fam} version {ver} is not released on {HUB_REPO} (no tag {rev})") from e
    p = Path(root) / fam
    if not (p / "model.json").exists():
        raise FileNotFoundError(f"no model '{fam}' at revision {rev} of {HUB_REPO}")
    return p


def load(path, device="cuda") -> Model:
    from safetensors.torch import load_model
    path = resolve(path)
    manifest = json.loads((path / "model.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"model format {manifest.get('format')} not supported (expected {FORMAT})")
    net = build_net(manifest["architecture"])
    load_model(net, str(path / "weights.safetensors"), device="cpu")
    net.to(device).eval()
    return Model(net, manifest, path)
