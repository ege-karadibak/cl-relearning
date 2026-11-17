"""Linear probe: how much CIFAR-10 knowledge is inside a backbone, without changing it.

    1. freeze the backbone: eval mode, no gradients, no augmentation
    2. turn every image into its features, once (512 for ResNet-18, 384 for the ViT)
    3. train a linear classifier (logistic regression) on the features of the never-seen 5k
    4. score it on CIFAR-10 test (or on the validation split, during calibration)

The probe is trained on never-seen images only: A trained on the A-seen ones, so a probe
trained on them would favour A. The same probe settings are used for every arm.

Used two ways:
    inside a run:  probe = Probe(device, cfg); probe(net) → {"probe_acc": ..., "probe_loss": ...}
    on a saved model:
        python src/probe.py --arm A --stage 1 --seed 1            probe A after stage 1
        python src/probe.py --arm B --stage 0 --seed 1            B at its random start
        python src/probe.py --arm A --stage 1 --seed 1 --choose-c pick the regularization on validation
        add --arch vit for the ViT experiment (configs/vit/probe.yaml, checkpoints/vit/, logs/vit/)
"""

import argparse                              # command-line options
import numpy as np                           # features as arrays
import torch                                 # no_grad
from sklearn.linear_model import LogisticRegression   # the linear classifier
from sklearn.metrics import log_loss                  # probe loss
from sklearn.preprocessing import StandardScaler      # put every feature on the same scale

from data import probe_data, val_data


def extract_features(net, loader, device):
    """Features + label for every image in `loader`. The model is not changed."""
    train_mode = net.training
    net.eval()                               # BatchNorm uses its saved statistics and doesn't update them (ViT: no dropout either way)
    feats, labels = [], []
    try:
        with torch.no_grad():
            for images, y in loader:
                feats.append(net.features(images.to(device, non_blocking=True)).float().cpu())
                labels.append(y)
    finally:
        net.train(train_mode)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def fit_and_score(train_x, train_y, score_x, score_y, C, max_iter):
    """Standardize with the training features, fit logistic regression, score. Returns (acc, loss)."""
    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(C=C, max_iter=max_iter)   # lbfgs, multinomial: deterministic
    clf.fit(scaler.transform(train_x), train_y)
    proba = clf.predict_proba(scaler.transform(score_x))
    acc = float((proba.argmax(axis=1) == score_y).mean())
    loss = float(log_loss(score_y, proba, labels=np.arange(proba.shape[1])))
    return acc, loss


class Probe:
    """Reusable probe: loaders are made once, so probing every few hundred steps stays cheap.

    cfg keys (configs/probe.yaml, ViT: configs/vit/probe.yaml): C (regularization; smaller = stronger), max_iter
    score_on: "test" for real runs, "val" during calibration
    """

    def __init__(self, device, cfg, score_on="test"):
        if score_on not in ("test", "val"):
            raise ValueError(f"score_on must be 'test' or 'val', not {score_on!r}")
        self.device, self.C, self.max_iter = device, cfg["C"], cfg["max_iter"]
        self.train_loader, test_loader = probe_data()             # never-seen 5k, CIFAR-10 test
        self.score_loader = test_loader if score_on == "test" else val_data()
        self.prefix = "probe" if score_on == "test" else "probe_val"

    def __call__(self, net):
        train_x, train_y = extract_features(net, self.train_loader, self.device)
        score_x, score_y = extract_features(net, self.score_loader, self.device)
        acc, loss = fit_and_score(train_x, train_y, score_x, score_y, self.C, self.max_iter)
        return {f"{self.prefix}_acc": acc, f"{self.prefix}_loss": loss}


def choose_C(net, device, Cs, max_iter):
    """Try several C values: fit on never-seen, score on validation. Returns {C: val_acc}."""
    train_x, train_y = extract_features(net, probe_data()[0], device)
    val_x, val_y = extract_features(net, val_data(), device)
    return {C: fit_and_score(train_x, train_y, val_x, val_y, C, max_iter)[0] for C in Cs}


# ---------------------------------------------------------------- probe a saved model
if __name__ == "__main__":                   # needed: the data loaders start worker processes
    import data
    from model import ARCHS
    from utils import Logger, checkpoint_path, get_device, load_checkpoint, load_config, roots

    parser = argparse.ArgumentParser(description="Probe a saved model.")
    parser.add_argument("--arm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--stage", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1 or mps (default: configs/base.yaml)")
    parser.add_argument("--smoke", action="store_true", help="use the smoke-test checkpoints")
    parser.add_argument("--choose-c", action="store_true", help="compare C values on validation instead")
    parser.add_argument("--arch", default="resnet18", choices=ARCHS, help="backbone (default: resnet18)")
    args = parser.parse_args()

    cfg = load_config("probe", args.arch)
    cfg["arch"] = args.arch                  # recorded with the log
    ckpt_root, log_root = roots(args.arch)
    device = get_device(args.device or cfg["device"], cfg["gpu_memory_fraction"])
    data.configure(cfg["num_workers"], pin_memory=device.type == "cuda")   # the probe never augments
    path = (ckpt_root / "smoke" / f"seed{args.seed}" / f"{args.arm}_stage{args.stage}.pt" if args.smoke
            else checkpoint_path(args.arm, args.stage, args.seed, args.arch))
    net, meta = load_checkpoint(path, device, arch=args.arch)

    if args.choose_c:
        for C, acc in choose_C(net, device, cfg["C_choices"], cfg["max_iter"]).items():
            print(f"C = {C:<8} validation acc = {acc:.4f}")
    else:
        result = Probe(device, cfg)(net)
        folder = log_root / "smoke" if args.smoke else log_root
        Logger(f"{args.arm}_stage{args.stage}_probe", args.seed, config=cfg, folder=folder, device=device) \
            .log(arm=args.arm, stage=args.stage, seed=args.seed, **result)
