"""Make the data splits once and save them to data/splits.npz.

Run from the project root:  python src/make_splits.py

CIFAR-10 train (50k)
├── pretrain pool   40k   A trains on this in stage 1
│   └── A-seen      5k    picked from the pool, a new one for each seed
└── held out        10k
    ├── validation  5k    for choosing settings
    └── never-seen  5k    no model ever trains on these before stage 3
CIFAR-100 train (50k)
└── pretrain        40k   C trains on this in stage 1

Every split has the same number of images per class.
We only save image numbers (indices), not the images.
"""

from pathlib import Path                     # file paths
import numpy as np                           # arrays and random numbers
from torchvision import datasets             # to read the labels

# ---------------------------------------------------------------- settings
DATA = Path(__file__).resolve().parent.parent / "data"   # data folder (works from any folder)
OUT = DATA / "splits.npz"                    # output file
SPLIT_SEED = 0                               # seed for the pools; never change it
RUN_SEEDS = [1, 2, 3, 4, 5]                  # one A-seen subset per seed

C10_PRETRAIN_PER_CLASS = 4000                # 4,000 × 10 = 40k
C10_VAL_PER_CLASS = 500                      # 500 × 10 = 5k
C10_NEVER_SEEN_PER_CLASS = 500               # 500 × 10 = 5k
A_SEEN_PER_CLASS = 500                       # 500 × 10 = 5k
C100_PRETRAIN_PER_CLASS = 400                # 400 × 100 = 40k, same size as A's pool


# ---------------------------------------------------------------- helpers
def load_labels():
    """Return the CIFAR-10 and CIFAR-100 train labels."""
    c10 = datasets.CIFAR10(DATA, train=True, download=False)
    c100 = datasets.CIFAR100(DATA, train=True, download=False)
    return np.array(c10.targets), np.array(c100.targets)


def take_per_class(indices, labels, n_per_class, rng):
    """Pick n_per_class random indices of each class from `indices`.

    Returns (picked, rest).
    """
    picked = []
    classes = np.unique(labels[indices])
    for k in classes:
        of_class = indices[labels[indices] == k]
        of_class = rng.permutation(of_class)
        picked.append(of_class[:n_per_class])
    picked = np.concatenate(picked)
    rest = np.setdiff1d(indices, picked)
    return np.sort(picked), np.sort(rest)


# ---------------------------------------------------------------- the splits
def make_cifar10_pools(c10_labels, rng):
    """Split CIFAR-10 train into pretrain pool, validation and never-seen."""
    pretrain, held_out = take_per_class(np.arange(len(c10_labels)), c10_labels, C10_PRETRAIN_PER_CLASS, rng)
    val, never_seen = take_per_class(held_out, c10_labels, C10_VAL_PER_CLASS, rng)
    return pretrain, val, never_seen


def make_a_seen(pretrain, c10_labels, run_seed):
    """Pick the 5k A-seen subset for one seed from the pretrain pool."""
    rng = np.random.default_rng(run_seed)
    picked, _ = take_per_class(pretrain, c10_labels, A_SEEN_PER_CLASS, rng)
    return picked


def make_cifar100_pretrain(c100_labels, rng):
    """Pick the 40k CIFAR-100 images for C."""
    picked, _ = take_per_class(np.arange(len(c100_labels)), c100_labels, C100_PRETRAIN_PER_CLASS, rng)
    return picked


def build_splits():
    """Make every split and return them as a dict (name → indices), plus the labels."""
    c10_labels, c100_labels = load_labels()
    rng = np.random.default_rng(SPLIT_SEED)
    c10_pretrain, c10_val, c10_never_seen = make_cifar10_pools(c10_labels, rng)
    c100_pretrain = make_cifar100_pretrain(c100_labels, rng)
    splits = {
        "c10_pretrain": c10_pretrain,
        "c10_val": c10_val,
        "c10_never_seen": c10_never_seen,
        "c100_pretrain": c100_pretrain,
    }
    for s in RUN_SEEDS:
        splits[f"a_seen_seed{s}"] = make_a_seen(c10_pretrain, c10_labels, s)
    return splits, c10_labels, c100_labels


# ---------------------------------------------------------------- checks
def check_splits(splits, c10_labels, c100_labels):
    """Stop with an error if any split is wrong."""
    pretrain = splits["c10_pretrain"]                                   # short names for the CIFAR-10 groups
    val = splits["c10_val"]
    never_seen = splits["c10_never_seen"]
    a_seen_names = [f"a_seen_seed{s}" for s in RUN_SEEDS]               # "a_seen_seed1", ..., "a_seen_seed5"

    # which labels and how many per class each split should have
    expected = {
        "c10_pretrain": (c10_labels, C10_PRETRAIN_PER_CLASS),
        "c10_val": (c10_labels, C10_VAL_PER_CLASS),
        "c10_never_seen": (c10_labels, C10_NEVER_SEEN_PER_CLASS),
        "c100_pretrain": (c100_labels, C100_PRETRAIN_PER_CLASS),
    }
    for name in a_seen_names:
        expected[name] = (c10_labels, A_SEEN_PER_CLASS)

    # 1. every split is there, and nothing extra
    assert set(splits) == set(expected), f"wrong split names: {sorted(splits)}"

    for name, (labels, per_class) in expected.items():
        idx = splits[name]
        n_classes = len(np.unique(labels))                              # 10 for CIFAR-10, 100 for CIFAR-100

        # 2. right size
        assert len(idx) == per_class * n_classes, f"{name}: {len(idx)} images, expected {per_class * n_classes}"
        # 3. no image twice
        assert len(np.unique(idx)) == len(idx), f"{name}: an image appears twice"
        # 4. every class has exactly per_class images
        counts = np.bincount(labels[idx], minlength=n_classes)
        assert (counts == per_class).all(), f"{name}: unequal class counts {counts}"

    # 5. the three CIFAR-10 groups don't overlap
    assert np.intersect1d(pretrain, val).size == 0, "pretrain and validation overlap"
    assert np.intersect1d(pretrain, never_seen).size == 0, "pretrain and never-seen overlap"
    assert np.intersect1d(val, never_seen).size == 0, "validation and never-seen overlap"

    # 6. together they are all 50k CIFAR-10 images
    all_c10 = np.sort(np.concatenate([pretrain, val, never_seen]))
    assert np.array_equal(all_c10, np.arange(len(c10_labels))), "the CIFAR-10 groups don't cover all images"

    # 7. every A-seen group is inside the pretrain pool, and the seeds differ
    for name in a_seen_names:
        assert np.isin(splits[name], pretrain).all(), f"{name}: has images outside the pretrain pool"
    for i, a in enumerate(a_seen_names):
        for b in a_seen_names[i + 1:]:
            assert not np.array_equal(splits[a], splits[b]), f"{a} and {b} are identical"

    print("All split checks passed.")


# ---------------------------------------------------------------- main
def main():
    """Make the splits, check them, and save."""
    # 1. never replace splits that already exist
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists. The splits must never change, so nothing was done.\n"
                         f"Only delete it if no real run has used it yet.")

    # 2. build and check
    splits, c10_labels, c100_labels = build_splits()
    check_splits(splits, c10_labels, c100_labels)

    # 3. build again: the result must be exactly the same
    again, _, _ = build_splits()
    for name in splits:
        assert np.array_equal(splits[name], again[name]), f"{name} changed on the second build"
    print("Second build is identical.")

    # 4. save and show what was saved
    np.savez(OUT, **splits)
    print(f"\nSaved to {OUT}")
    for name, idx in splits.items():
        print(f"  {name:<16} {len(idx):>6} images")


if __name__ == "__main__":                   # run only when started as a script
    main()
