"""Datasets, transforms and DataLoaders for every stage.

Everything that touches images goes through this file, so all stages
load data the same way.

What each stage uses:
    stage 1   A: CIFAR-10 pretrain pool (40k)     C: CIFAR-100 pretrain (40k)
    stage 2   all arms: SVHN train
    stage 3   all arms: A-seen 5k (one per seed) or never-seen 5k
    probe     never-seen 5k (train the probe), CIFAR-10 test (score it)
    tuning    CIFAR-10 validation 5k
Test sets are only for reporting results.
"""

from pathlib import Path                     # file paths
import numpy as np                           # to read splits.npz
import torch                                 # random generator for shuffling
from torchvision import datasets, transforms as T   # datasets and image transforms
from torch.utils.data import DataLoader, Subset     # batching, and picking images by index

# ---------------------------------------------------------------- settings
DATA = Path(__file__).resolve().parent.parent / "data"   # data folder (works from any folder)
SPLITS = DATA / "splits.npz"                 # made by make_splits.py

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)      # measured in notebooks/01_datasets.ipynb
CIFAR10_STD = (0.2470, 0.2435, 0.2616)       # used for every dataset and stage

BATCH_SIZE = 128                             # same in every stage
NUM_WORKERS = 4                              # background processes preparing batches (run scripts set it from the config)
PIN_MEMORY = False                           # faster copies to a CUDA GPU; run scripts turn it on for CUDA
AUGMENT = "basic"                            # training augmentation: "basic" (ResNet-18) or "randaugment" (ViT)
RANDAUGMENT = (2, 9)                         # RandAugment (number of operations, magnitude), if used
SVHN_MAGNITUDE = None                        # RandAugment magnitude for SVHN only (None = the same as the others)


def configure(num_workers, pin_memory, augment="basic", randaugment=(2, 9), svhn_magnitude=None):
    """Set loader options before making loaders (run scripts call this from the config and device).

    augment:        "basic" = random crop + flip (no flip for SVHN)
                    "randaugment" = the same, then RandAugment(*randaugment)
    svhn_magnitude: a weaker RandAugment magnitude for SVHN only (None = the same as the others)
    """
    global NUM_WORKERS, PIN_MEMORY, AUGMENT, RANDAUGMENT, SVHN_MAGNITUDE
    if augment not in ("basic", "randaugment"):
        raise ValueError(f"unknown augment {augment!r}, use 'basic' or 'randaugment'")
    if svhn_magnitude is not None and not (isinstance(svhn_magnitude, int) and 0 <= svhn_magnitude <= 30):
        raise ValueError(f"svhn_magnitude must be a whole number from 0 to 30 or None, not {svhn_magnitude!r}")
    NUM_WORKERS, PIN_MEMORY, AUGMENT, RANDAUGMENT = num_workers, pin_memory, augment, tuple(randaugment)
    SVHN_MAGNITUDE = svhn_magnitude


def augment_from(cfg):
    """A config's augmentation settings, as keywords for configure. No setting = "basic" (ResNet-18)."""
    return {"augment": cfg.get("augment", "basic"),
            "randaugment": (cfg.get("randaugment_ops", 2), cfg.get("randaugment_magnitude", 9)),
            "svhn_magnitude": cfg.get("randaugment_magnitude_svhn")}   # no setting = same as the others


# ---------------------------------------------------------------- splits
def load_splits():
    """Return the saved splits as a dict (name → index array)."""
   
    if not SPLITS.exists(): 
        raise FileNotFoundError(f"{SPLITS} not found. Run: python src/make_splits.py")
    with np.load(SPLITS) as data:                                        # open; closes automatically at the end
        return {name: data[name] for name in data.files} 


# ---------------------------------------------------------------- transforms
def make_transform(name, augment):
    """Build the image transform for one dataset.

    name:    "cifar10", "cifar100" or "svhn"
    augment: True for training, False for evaluation and the probe
             (which augmentation: AUGMENT, set by configure)
    """
    steps = []
    if augment:
        steps.append(T.RandomCrop(32, padding=4))
        if name != "svhn":
            steps.append(T.RandomHorizontalFlip())
        if AUGMENT == "randaugment":
            num_ops, magnitude = RANDAUGMENT
            if name == "svhn" and SVHN_MAGNITUDE is not None:
                magnitude = SVHN_MAGNITUDE           # the SVHN-only fallback (configs/vit/base.yaml)
            steps.append(T.RandAugment(num_ops=num_ops, magnitude=magnitude))   # on the image, before ToTensor
    steps.append(T.ToTensor())
    steps.append(T.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return T.Compose(steps)
        




# ---------------------------------------------------------------- datasets
def load_dataset(name, train, augment):
    """Load a full dataset with the right transform.

    name:    "cifar10", "cifar100" or "svhn"
    train:   True for the training set, False for the test set
    augment: whether to use random crop/flip
    """
    transform = make_transform(name, augment)
    match name:
        case "cifar10":
            return datasets.CIFAR10(DATA, train=train, download=False, transform=transform)
        case "cifar100":
            return datasets.CIFAR100(DATA, train=train, download=False, transform=transform)
        case "svhn":
            return datasets.SVHN(DATA / "svhn", split="train" if train else "test", download=False, transform=transform)
        case _:
            raise ValueError
        

def load_split(name, split_name, augment):
    """Load only the images of one saved split, e.g. ("cifar10", "c10_never_seen").

    Splits are parts of a TRAINING set, so this always loads train=True.
    Use augment=False for validation and the probe.
    """
    full = load_dataset(name,train=True, augment=augment )
    idx = load_splits()[split_name]
    return Subset(full, idx)
    


# ---------------------------------------------------------------- loaders
def make_loader(dataset, shuffle, seed=0):
    """Wrap a dataset in a DataLoader.

    shuffle: True for training (new order every epoch), False for evaluation
    seed:    fixes the shuffle order so a run can be repeated
    """
 
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      persistent_workers=NUM_WORKERS > 0, pin_memory=PIN_MEMORY, generator=generator)




# ---------------------------------------------------------------- one function per stage
def stage1_data(arm, seed):
    """Stage 1 loaders: (train, test).

    arm "A": CIFAR-10 pretrain pool, scored on CIFAR-10 test
    arm "C": CIFAR-100 pretrain set, scored on CIFAR-100 test
    arm "B": no stage 1, so raise an error
    """

    match arm:
        case "A":
            train = make_loader(load_split("cifar10", "c10_pretrain", augment=True), shuffle=True, seed=seed)
            test = make_loader(load_dataset("cifar10", train=False, augment=False), shuffle=False, seed=seed)
            return train, test
        case "C":
            train = make_loader(load_split("cifar100", "c100_pretrain", augment=True), shuffle=True, seed=seed)
            test = make_loader(load_dataset("cifar100", train=False, augment=False),shuffle=False, seed=seed)
            return train, test
        case "B":
            raise ValueError()
        case _:
            raise ValueError()
            


def stage2_data(seed):
    """Stage 2 loaders: (train, test). All arms: the full SVHN train set and SVHN test."""

    train = make_loader(load_dataset("svhn", train=True,augment=True),shuffle=True, seed=seed)
    test = make_loader(load_dataset("svhn", train=False, augment=False), shuffle=False, seed=seed)
    return train, test


def stage3_data(subset, seed):
    """Stage 3 loaders: (train, test).

    subset "a_seen":     the A-seen 5k for this seed ("a_seen_seed{seed}")
    subset "never_seen": the never-seen 5k (same for every seed)
    Scored on CIFAR-10 test.
    """

    match subset:
        case "a_seen":
            split_name = f"a_seen_seed{seed}"        # seed 1 → "a_seen_seed1", seed 2 → "a_seen_seed2", ...
        case "never_seen":
            split_name = "c10_never_seen"            # the same 5k for every seed
        case _:
            raise ValueError(f"unknown subset {subset!r}, use 'a_seen' or 'never_seen'")

    train = make_loader(load_split("cifar10", split_name, augment=True), shuffle=True, seed=seed)       # 5k, augmented, shuffled
    test = make_loader(load_dataset("cifar10", train=False, augment=False), shuffle=False, seed=seed)  # CIFAR-10 test, plain, in order
    return train, test

    


def probe_data():
    """Probe loaders: (train, test). No augmentation and no shuffling.

    train: never-seen 5k (the probe is never trained on A-seen images)
    test:  CIFAR-10 test
    """
    train = make_loader(load_split("cifar10", "c10_never_seen", augment=False), shuffle=False)
    test = make_loader(load_dataset("cifar10", train=False, augment=False), shuffle=False)  # CIFAR-10 test, plain, in order
    return train, test


def val_data():
    """Validation loader: CIFAR-10 validation 5k, no augmentation, not shuffled."""
    return make_loader(load_split("cifar10", "c10_val", augment=False), shuffle=False)

# ---------------------------------------------------------------- quick check
if __name__ == "__main__":                   # run: python src/data.py
    train, test = stage1_data("A", 1)                          # loaders for arm A
    images, labels = next(iter(train))                         # first batch
    print(images.shape, labels.shape, labels.min().item(), labels.max().item())  # expect (128, 3, 32, 32) (128,) 0 9
    print(len(train.dataset), len(test.dataset))               # expect 40000 10000
    print(make_transform("svhn", augment=True))                # crop only (basic)

    configure(NUM_WORKERS, PIN_MEMORY, augment="randaugment")  # the ViT's setting
    print(make_transform("cifar10", augment=True))             # crop, flip, RandAugment
    print(make_transform("cifar10", augment=False))            # evaluation: unchanged, no augmentation
    images, _ = next(iter(stage2_data(1)[0]))                  # an augmented SVHN batch
    print(images.shape)                                        # expect (128, 3, 32, 32)


