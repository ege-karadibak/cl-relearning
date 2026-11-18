"""BatchNorm check (ResNet-18 only): how much of the "forgetting" is only BatchNorm statistics?

ResNet-18's BatchNorm layers keep a running mean and variance of the data they see. These are
averages, not learned weights, and during stage 2 they drift to SVHN. This script takes a saved
model, recomputes ONLY those statistics from images of a chosen dataset (forward passes, no
gradients, no labels, no weight changes), and scores the old head again.

    python src/bn_check.py                      seeds 1-3, arms A and C (lab GPU, like the real runs)
    python src/bn_check.py --seeds 1 --arms A   a part of it
    python src/bn_check.py --smoke --device mps test on the smoke checkpoints (writes apart)

For every (seed, arm, stage) it evaluates one fresh copy of the checkpoint per variant:
    as_saved     statistics as saved (control: must reproduce the logged old-head accuracy)
    own          recomputed on the arm's own stage-1 images (A: c10_pretrain, C: c100_pretrain)
    never_seen   recomputed on the never-seen CIFAR-10 5k (A only: does it depend on the images?)
    svhn         recomputed on SVHN train (control: re-estimating by itself should not help)
Stage-2 checkpoints get all variants; stage-1 checkpoints get as_saved and own (control: on a
model that was never overwritten, the recompute should change almost nothing).

Measured per variant: the old head on its own test set (A: cifar10, C: cifar100), the svhn head
on SVHN test (stage 2 only), and the CIFAR-10 linear probe (probe settings of configs/probe.yaml).

Safety: models are only loaded and changed in memory; nothing is ever saved as a checkpoint.
Results go to logs/analysis/bn_check/seed{N}/ (smoke: logs/analysis/bn_check_smoke/); the Logger
refuses to write over an existing log, so a finished result cannot be overwritten.
"""

import argparse                              # command-line options
import json                                  # reading the logged values back

import torch                                 # no_grad
from torch import nn                         # to find the BatchNorm layers

import data
from data import load_dataset, load_split, make_loader
from evaluate import evaluate, evaluate_head
from probe import Probe
from utils import (CHECKPOINTS, LOGS, Logger, checkpoint_path, get_device, load_checkpoint,
                   load_config, set_seed)

OLD_HEAD = {"A": ("cifar10", "cifar10"), "C": ("cifar100", "cifar100")}   # arm → (old head, its dataset)
OWN_IMAGES = {"A": ("cifar10", "c10_pretrain"), "C": ("cifar100", "c100_pretrain")}  # arm → its stage-1 images


def parse_args():
    parser = argparse.ArgumentParser(description="BatchNorm check on saved ResNet-18 models.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--arms", nargs="+", default=["A", "C"], choices=["A", "C"])
    parser.add_argument("--smoke", action="store_true", help="use the smoke checkpoints; write to logs/analysis/bn_check_smoke/")
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1 or mps (default: configs/base.yaml)")
    return parser.parse_args()


# ---------------------------------------------------------------- the one new operation
def recompute_bn_stats(net, loader, device):
    """Replace every BatchNorm layer's running mean/variance by the plain average over `loader`.

    Only the statistics change: the weights, biases and heads are untouched, no gradients are
    computed and no labels are used. momentum=None makes BatchNorm keep an equal-weight average
    of all batches instead of a moving average.
    """
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise RuntimeError("this model has no BatchNorm layers (the check is for ResNet-18 only)")
    for m in bns:
        m.reset_running_stats()              # mean 0, variance 1, batch count 0
        m.momentum = None                    # cumulative average over all batches
    net.train()                              # BatchNorm updates its statistics only in train mode (ResNet-18 has no dropout)
    with torch.no_grad():
        for images, _ in loader:             # labels are not used
            net.features(images.to(device, non_blocking=True))
    net.eval()
    return len(bns)


# ---------------------------------------------------------------- helpers
def logged_value(path, key):
    """The last logged value of `key` in a .jsonl log (the end of that run), or None."""
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if key in row and row[key] is not None:
            last = row[key]
    return last


def stats_loader(variant, arm):
    """Images the statistics are recomputed on: no augmentation, fixed order."""
    if variant == "own":
        name, split = OWN_IMAGES[arm]
        return make_loader(load_split(name, split, augment=False), shuffle=False)
    if variant == "never_seen":
        return make_loader(load_split("cifar10", "c10_never_seen", augment=False), shuffle=False)
    if variant == "svhn":
        return make_loader(load_dataset("svhn", train=True, augment=False), shuffle=False)
    raise ValueError(variant)


def main():
    args = parse_args()
    cfg = load_config("probe", "resnet18")   # device, workers, and the probe settings (C = 0.1)
    device = get_device(args.device or cfg["device"], cfg["gpu_memory_fraction"])
    cfg["device"] = str(device)
    data.configure(cfg["num_workers"], pin_memory=device.type == "cuda")   # no augmentation is used anywhere here
    out = LOGS / "analysis" / ("bn_check_smoke" if args.smoke else "bn_check")
    log_root = LOGS / "smoke" if args.smoke else LOGS
    probe = Probe(device, cfg)

    test = {name: make_loader(load_dataset(name, train=False, augment=False), shuffle=False)
            for name in ("cifar10", "cifar100", "svhn")}

    for seed in args.seeds:
        for arm in args.arms:
            old_head, old_data = OLD_HEAD[arm]
            for stage in (1, 2):
                path = (CHECKPOINTS / "smoke" / f"seed{seed}" / f"{arm}_stage{stage}.pt" if args.smoke
                        else checkpoint_path(arm, stage, seed, "resnet18"))
                if not path.exists():
                    print(f"skip: {path} not found")
                    continue
                # the value this run must reproduce with the saved statistics
                log = log_root / f"seed{seed}" / f"{arm}_stage{stage}.jsonl"
                logged = logged_value(log, "old_acc" if stage == 2 else "test_acc")
                variants = ["as_saved", "own"] + (["never_seen"] if arm == "A" and stage == 2 else []) \
                           + (["svhn"] if stage == 2 else [])
                logger = Logger(f"{arm}_stage{stage}_bn", seed, folder=out, device=device,
                                config={**cfg, "checkpoint": str(path.relative_to(CHECKPOINTS.parent)),
                                        "variants": variants, "logged_old_acc": logged})
                for variant in variants:
                    set_seed(seed)
                    net, _ = load_checkpoint(path, device, arch="resnet18")   # a fresh copy for every variant
                    n_bn = 0 if variant == "as_saved" else recompute_bn_stats(net, stats_loader(variant, arm), device)
                    net.eval()
                    row = {"seed": seed, "arm": arm, "stage": stage, "variant": variant, "bn_layers_recomputed": n_bn}
                    old = evaluate_head(net, test[old_data], device, old_head)
                    row.update(old_acc=old["acc"], old_loss=old["loss"])
                    if stage == 2:
                        sv = evaluate_head(net, test["svhn"], device, "svhn")
                        row.update(svhn_acc=sv["acc"], svhn_loss=sv["loss"])
                    row.update(probe(net))   # probe_acc, probe_loss (CIFAR-10, frozen backbone)
                    if variant == "as_saved":
                        row["logged_old_acc"] = logged
                        row["matches_log"] = logged is not None and abs(old["acc"] - logged) < 1e-6
                        if not row["matches_log"]:
                            print(f"WARNING: seed {seed} {arm} stage {stage}: old head {old['acc']:.4f} "
                                  f"but the run logged {logged}; the numbers below are not comparable")
                    logger.log(**row)
                    extra = f" · svhn {row['svhn_acc']:.4f}" if stage == 2 else ""
                    print(f"seed {seed} {arm} stage {stage} {variant:<10} old head {row['old_acc']:.4f}{extra} "
                          f"· probe {row['probe_acc']:.4f}")
                    del net
    print(f"\nresults → {out}")


if __name__ == "__main__":                   # needed: the data loaders start worker processes
    main()
