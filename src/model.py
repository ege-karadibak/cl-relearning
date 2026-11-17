"""Backbone (ResNet-18 or a small ViT) with swappable classifier heads.

One backbone (the part that learns features) is shared across all stages.
Each dataset gets its own small head (the part that picks the class),
because the labels mean different things in each dataset.

    image (3×32×32) ──► backbone ──► features ──► active head ──► class scores

Backbones ("arch"):
    "resnet18"    ResNet-18 adapted to 32×32 images: 512 features (the default)
    "vit"         small Vision Transformer, 4×4 patches: 384 features (the class token)

Heads used in the experiment:
    "cifar10"     A's head from stage 1 (kept to measure the old-head accuracy)
    "cifar100"    C's head from stage 1
    "svhn"        every arm's head in stage 2
    "relearn"     a fresh CIFAR-10 head for every arm in stage 3
"""

import torch                                 # tensors
from torch import nn                         # layers
from torchvision.models import resnet18      # the standard ResNet-18
from torchvision.models.vision_transformer import VisionTransformer   # the standard ViT, sized below

# ---------------------------------------------------------------- settings
FEATURE_DIM = {                              # features per image, for each backbone
    "resnet18": 512,
    "vit": 384,
}
ARCHS = tuple(FEATURE_DIM)                   # the backbones the experiment can use
VIT_SIZE = {                                 # ~10.7M parameters, close to ResNet-18's 11.2M
    "image_size": 32,                        # CIFAR / SVHN images, no resizing
    "patch_size": 4,                         # 32×32 → 8×8 = 64 patches
    "num_layers": 6,                         # transformer blocks
    "num_heads": 6,                          # attention heads (6 × 64 = 384)
    "hidden_dim": 384,                       # numbers per token
    "mlp_dim": 1536,                         # 4 × hidden_dim, the usual ratio
}
N_CLASSES = {                                # how many outputs each head needs
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "relearn": 10,                           # CIFAR-10 again, but a new head
}


# ---------------------------------------------------------------- backbone
def make_backbone(arch="resnet18"):
    """The backbone for `arch`, without its classifier. Random weights, no pretraining."""
    if arch == "resnet18":
        return make_resnet18()
    if arch == "vit":
        return make_vit()
    raise ValueError(f"unknown arch {arch!r}, use one of {list(ARCHS)}")


def make_resnet18():
    """ResNet-18 adapted to 32×32 images, without its classifier.

    Returns a module: images (N, 3, 32, 32) → features (N, 512).
    """
    net = resnet18(weights=None)                         # random weights, no ImageNet pretraining
    net.conv1 = nn.Conv2d(                               # small first layer for small images
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    net.maxpool = nn.Identity()                          # no early shrinking
    net.fc = nn.Identity()                               # no classifier; the heads do that
    return net


def make_vit():
    """Small Vision Transformer for 32×32 images, without its classifier.

    The image is cut into 64 patches of 4×4 pixels; each becomes a token of 384 numbers.
    A learned class token is added, 6 transformer blocks mix all tokens, and the class
    token at the end is the image's features.
    Returns a module: images (N, 3, 32, 32) → features (N, 384).
    """
    net = VisionTransformer(**VIT_SIZE)                  # no dropout (the default): augmentation regularizes
    net.heads = nn.Identity()                            # no classifier; the heads do that
    return net


# ---------------------------------------------------------------- the model
class Net(nn.Module):
    """Backbone + a set of named heads; one head is active at a time."""

    def __init__(self, arch="resnet18"):
        super().__init__()
        self.arch = arch                                 # saved with every checkpoint
        self.backbone = make_backbone(arch)
        self.feature_dim = FEATURE_DIM[arch]             # what every head reads
        self.heads = nn.ModuleDict()                     # PyTorch saves and moves these with the model
        self.active = None                               # name of the head in use

    def add_head(self, name):
        """Add a new, randomly initialised head and make it active.

        name: a key of N_CLASSES, e.g. "svhn"
        """
        if name not in N_CLASSES:
            raise ValueError(f"unknown head {name!r}, use one of {list(N_CLASSES)}")
        if name in self.heads:
            raise ValueError(f"head {name!r} already exists; a trained head is never replaced")
        device = next(self.backbone.parameters()).device  # where the model lives (cuda, mps or cpu)
        self.heads[name] = nn.Linear(self.feature_dim, N_CLASSES[name]).to(device)
        self.active = name

    def set_head(self, name):
        """Switch to an existing head (e.g. back to A's old "cifar10" head)."""
        if name not in self.heads:
            raise ValueError(f"no head {name!r}; existing heads: {list(self.heads)}")
        self.active = name

    def features(self, x):
        """Images → features (512 for ResNet-18, 384 for the ViT). Used by the linear probe."""
        return self.backbone(x)

    def forward(self, x):
        """Images → class scores from the active head."""
        if self.active is None:
            raise ValueError("no head yet; call add_head first")
        return self.heads[self.active](self.features(x))


# ---------------------------------------------------------------- helpers
def count_params(module):
    """Number of trainable numbers in a module."""
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- quick check
if __name__ == "__main__":                   # run: python src/model.py
    device = "cuda:0" if torch.cuda.is_available() else "mps"
    net = Net().to(device)
    x = torch.randn(4, 3, 32, 32, device=device)         # fake batch of 4 images

    net.add_head("cifar10")
    print("cifar10 head: ", tuple(net(x).shape))         # expect (4, 10)
    print("features:     ", tuple(net.features(x).shape))  # expect (4, 512)

    net.add_head("cifar100")
    print("cifar100 head:", tuple(net(x).shape))         # expect (4, 100)

    net.set_head("cifar10")
    print("back to cifar10:", tuple(net(x).shape))       # expect (4, 10)

    try:
        net.add_head("cifar10")                          # must refuse: would wipe a trained head
    except ValueError as e:
        print("refused:", e)

    print(f"backbone parameters: {count_params(net.backbone):,}")   # about 11.2 million
    print(f"heads: {list(net.heads)} on {device}")

    vit = Net("vit").to(device)
    vit.add_head("cifar10")
    print("vit cifar10 head:", tuple(vit(x).shape))      # expect (4, 10)
    print("vit features:    ", tuple(vit.features(x).shape))  # expect (4, 384)
    print(f"vit backbone parameters: {count_params(vit.backbone):,}")   # about 10.7 million
