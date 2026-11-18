"""Per-layer linear probes and CKA (analysis only). Q3: where is the "lost" knowledge kept?

    python src/layer_probe.py --arch resnet18 --choose-c          choose C per layer (A, stage 1, seed 1, validation)
    python src/layer_probe.py --arch resnet18                     probes + CKA, seeds 1-3, every checkpoint
    python src/layer_probe.py --arch vit --seeds 1                a part of it
    python src/layer_probe.py --arch resnet18 --smoke DIR --device mps   test on a smoke-chain copy (DIR/checkpoints, DIR/logs)

Reads every layer of a saved model with forward hooks (read-only taps: the model and model.py are not
changed), turns each layer into a feature vector (see configs/analysis/layer_probe.yaml), and fits the
same linear probe as src/probe.py on each one: logistic regression trained on the never-seen 5k, scored
on CIFAR-10 test (validation for --choose-c). The "final" readout is exactly the existing probe and must
reproduce the logged probe value (matches_log).

Checkpoints per seed: A and C after stage 1, B at its start (stage 0), and A, B, C after stage 2.
CKA (linear, on the never-seen 5k): each arm before vs after stage 2, and A vs C at stage 1 and stage 2.

Safety: checkpoints are only loaded; nothing is ever saved as a model. Results go to
logs/analysis/layer_probe/{arch}/ (smoke: logs/analysis/layer_probe_smoke/{arch}/); the Logger refuses to
write over an existing log.
"""

import argparse                              # command-line options
import json                                  # reading the logged probe values back
from pathlib import Path                     # file paths

import numpy as np                           # features and CKA
import torch                                 # no_grad
import torch.nn.functional as F              # pooling
import yaml                                  # the analysis config

import data
from data import probe_data, val_data
from probe import fit_and_score
from utils import (LOGS, ROOT, Logger, checkpoint_path, get_device, load_checkpoint, load_config, roots,
                   set_seed)

CHECKPOINTS_PER_SEED = [("A", 1), ("C", 1), ("B", 0), ("A", 2), ("B", 2), ("C", 2)]
CONFIG = ROOT / "configs" / "analysis" / "layer_probe.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Per-layer linear probes and CKA on saved models.")
    parser.add_argument("--arch", required=True, choices=["resnet18", "vit"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--choose-c", action="store_true", help="compare C_choices per layer on validation (A, stage 1, seed 1)")
    parser.add_argument("--tag", default="try1", help="name of a --choose-c attempt (each attempt gets its own log)")
    parser.add_argument("--smoke", default=None, metavar="DIR", help="a smoke-chain copy with checkpoints/ and logs/")
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1 or mps (default: configs/base.yaml)")
    return parser.parse_args()


# ---------------------------------------------------------------- reading layers
def readout_names(arch):
    if arch == "resnet18":
        return ["stem", "layer1", "layer2", "layer3", "layer4", "final"]
    return ["embed_patch"] + [f"block{i}_{k}" for i in range(1, 7) for k in ("cls", "patch")] + ["final"]


def taps(net, arch):
    """(module, name) pairs whose outputs are read. Forward hooks only read, never change, the output."""
    b = net.backbone
    if arch == "resnet18":
        # maxpool is an Identity in our ResNet-18, so its output is the stem's output (conv1 → bn1 → relu)
        return [(b.maxpool, "stem"), (b.layer1, "layer1"), (b.layer2, "layer2"), (b.layer3, "layer3"), (b.layer4, "layer4")]
    # the encoder adds the position embedding, then dropout (zero here); its output is the block input
    return [(b.encoder.dropout, "embed")] + [(layer, f"block{i + 1}") for i, layer in enumerate(b.encoder.layers)]


def to_vectors(name, out, arch):
    """One tapped output → {readout name: (N, d) tensor}."""
    if arch == "resnet18":
        return {name: F.adaptive_avg_pool2d(out, 2).flatten(1)}       # (N, C, H, W) → (N, 4C)
    patch = out[:, 1:].mean(dim=1)                                     # (N, 65, 384): token 0 is the class token
    if name == "embed":
        return {"embed_patch": patch}                                  # the class token is the same for every image here
    return {f"{name}_cls": out[:, 0], f"{name}_patch": patch}


def extract_layers(net, loader, device, arch):
    """Every readout's features + the labels, for every image in `loader`. The model is not changed."""
    store, hooks = {}, []
    for module, name in taps(net, arch):
        def hook(_m, _inp, out, name=name):
            for k, v in to_vectors(name, out, arch).items():
                store.setdefault(k, []).append(v.float().cpu())
        hooks.append(module.register_forward_hook(hook))
    finals, labels = [], []
    net.eval()                               # BatchNorm uses its saved statistics; nothing is updated
    try:
        with torch.no_grad():
            for images, y in loader:
                finals.append(net.features(images.to(device, non_blocking=True)).float().cpu())
                labels.append(y)
    finally:
        for h in hooks:
            h.remove()                       # the model is left exactly as it was loaded
    feats = {k: torch.cat(v).numpy() for k, v in store.items()}
    feats["final"] = torch.cat(finals).numpy()
    return feats, torch.cat(labels).numpy()


# ---------------------------------------------------------------- CKA
def linear_cka(x, y):
    """Linear CKA between two feature matrices of the same images (rows). 1 = same up to rotation and scale."""
    x = x.astype(np.float64) - x.mean(axis=0)
    y = y.astype(np.float64) - y.mean(axis=0)
    cross = np.linalg.norm(y.T @ x) ** 2
    return float(cross / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


# ---------------------------------------------------------------- helpers
def logged_probe(log_root, seed, arm, stage):
    """The probe value the real run logged for this checkpoint (end of stage 2, or the stage-1/0 probe file)."""
    path = (log_root / f"seed{seed}" / f"{arm}_stage2.jsonl" if stage == 2
            else log_root / f"seed{seed}" / f"{arm}_stage{stage}_probe.jsonl")
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("probe_acc") is not None:
            last = row["probe_acc"]
    return last


def main():
    args = parse_args()
    arch = args.arch
    cfg = load_config("probe", arch)                      # device, workers, and C for "final"
    lp = yaml.safe_load(CONFIG.read_text())       # C per readout, C_choices, max_iter
    device = get_device(args.device or cfg["device"], cfg["gpu_memory_fraction"])
    data.configure(cfg["num_workers"], pin_memory=device.type == "cuda")   # no augmentation anywhere here
    ckpt_root, log_root = roots(arch)
    sub = "layer_probe_smoke" if args.smoke else "layer_probe"
    out = LOGS / "analysis" / sub / arch
    names = readout_names(arch)
    settings = {"arch": arch, "device": str(device), "C_final": cfg["C"], "max_iter": lp["max_iter"],
                "C_choices": lp["C_choices"], "C_layers": lp["C"][arch]}

    def path_of(arm, stage, seed):
        if args.smoke:
            return Path(args.smoke) / "checkpoints" / f"{arm}_stage{stage}.pt"
        return checkpoint_path(arm, stage, seed, arch)

    train_loader, test_loader = probe_data()             # never-seen 5k, CIFAR-10 test

    # ---------- choose C per layer
    if args.choose_c:
        seed = 1
        net, _ = load_checkpoint(path_of("A", 1, seed), device, arch=arch)
        tr, ytr = extract_layers(net, train_loader, device, arch)
        va, yva = extract_layers(net, val_data(), device, arch)
        chosen_file = out / "chosen_c.json"
        if chosen_file.exists():
            raise SystemExit(f"{chosen_file} already exists: C was already chosen for {arch}.")
        logger = Logger(f"choose_c_{args.tag}", seed, config=settings, folder=out, device=device)
        chosen = {}
        for name in names:
            if name == "final":
                continue                                 # fixed by configs/probe.yaml
            # the rule (configs/analysis/layer_probe.yaml): the best C not at the edge of the values tried;
            # if the best is at the edge, add the next value beyond it (×10 or ÷10) and look again, at most 3 times
            grid = list(lp["C_choices"])
            accs = {C: fit_and_score(tr[name], ytr, va[name], yva, C, lp["max_iter"])[0] for C in grid}
            for _ in range(3):
                best = max(grid, key=lambda C: accs[C])
                if best not in (grid[0], grid[-1]):
                    break
                new_C = float(f"{best / 10:g}") if best == grid[0] else float(f"{best * 10:g}")
                accs[new_C] = fit_and_score(tr[name], ytr, va[name], yva, new_C, lp["max_iter"])[0]
                grid = sorted(accs)
            best = max(grid, key=lambda C: accs[C])
            at_edge = best in (grid[0], grid[-1])
            chosen[name] = best
            logger.log(readout=name, dims=int(tr[name].shape[1]), C_tried=grid,
                       val_acc=[accs[C] for C in grid], best_C=best, best_at_edge=at_edge)
            print(f"{name:<13} {tr[name].shape[1]:>5} dims  " + "  ".join(f"C={C:g}: {accs[C]:.4f}" for C in grid)
                  + f"  → C = {best:g}{'  (STILL AT THE EDGE after 3 extensions)' if at_edge else ''}")
        chosen_file.write_text(json.dumps({"arch": arch, "rule": "best not at the edge, extended up to 3 times",
                                           "chosen_on": "A, stage 1, seed 1, validation", "C": chosen}, indent=2))
        print(f"\nchosen C → {chosen_file}\nresults → {logger.path}")
        return

    # ---------- probes + CKA
    Cs = dict(lp["C"][arch])                           # values written into the config take precedence
    chosen_file = out / "chosen_c.json"                 # else: the values --choose-c chose by the rule
    if chosen_file.exists():
        for n, C in json.loads(chosen_file.read_text())["C"].items():
            if Cs.get(n) is None:
                Cs[n] = C
    missing = [n for n in names if n != "final" and Cs.get(n) is None]
    if missing and not args.smoke:
        raise SystemExit(f"no C yet for {missing}: run --choose-c first (or set them in {CONFIG.relative_to(ROOT)}).")
    for n in missing:                                    # smoke test only: any C, the numbers are throwaway
        Cs[n] = 0.1
    Cs["final"] = cfg["C"]

    for seed in args.seeds:
        set_seed(seed)
        train_feats = {}                                 # never-seen features per checkpoint, kept for CKA
        for arm, stage in CHECKPOINTS_PER_SEED:
            path = path_of(arm, stage, seed)
            if not path.exists():
                print(f"skip: {path} not found")
                continue
            net, _ = load_checkpoint(path, device, arch=arch)
            tr, ytr = extract_layers(net, train_loader, device, arch)
            te, yte = extract_layers(net, test_loader, device, arch)
            del net
            logged = (logged_probe_flat(Path(args.smoke) / "logs", arm, stage) if args.smoke
                      else logged_probe(log_root, seed, arm, stage))
            logger = Logger(f"{arm}_stage{stage}", seed, folder=out, device=device,
                            config={**settings, "checkpoint": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                                    "C_used": Cs, "logged_final_probe": logged})
            accs = {}
            for name in names:
                acc, loss = fit_and_score(tr[name], ytr, te[name], yte, Cs[name], lp["max_iter"])
                row = {"seed": seed, "arm": arm, "stage": stage, "readout": name, "dims": int(tr[name].shape[1]),
                       "C": Cs[name], "probe_acc": acc, "probe_loss": loss}
                if name == "final":
                    row["logged_probe_acc"] = logged
                    row["matches_log"] = logged is not None and abs(acc - logged) < 1e-6
                    if not row["matches_log"]:
                        print(f"WARNING: seed {seed} {arm} stage {stage}: final probe {acc:.4f} but the run logged "
                              f"{logged}; the per-layer numbers of this checkpoint are not comparable")
                logger.log(**row)
                accs[name] = acc
            print(f"seed {seed} {arm} stage {stage}: " + "  ".join(f"{n} {a:.3f}" for n, a in accs.items()))
            train_feats[(arm, stage)] = tr

        # CKA on the never-seen 5k (same images, same order for every checkpoint)
        pairs = [(("A", 1), ("A", 2)), (("C", 1), ("C", 2)), (("B", 0), ("B", 2)),
                 (("A", 1), ("C", 1)), (("A", 2), ("C", 2)), (("A", 1), ("A", 1))]   # last: control, must be 1
        pairs = [(p, q) for p, q in pairs if p in train_feats and q in train_feats]
        if pairs:
            logger = Logger("cka", seed, folder=out, device=device, config=settings)
            for p, q in pairs:
                for name in names:
                    value = linear_cka(train_feats[p][name], train_feats[q][name])
                    logger.log(seed=seed, a=f"{p[0]}_stage{p[1]}", b=f"{q[0]}_stage{q[1]}", readout=name, cka=value)
                print(f"seed {seed} CKA {p[0]}{p[1]} vs {q[0]}{q[1]}: " +
                      "  ".join(f"{n} {linear_cka(train_feats[p][n], train_feats[q][n]):.3f}" for n in names))
    print(f"\nresults → {out}")


def logged_probe_flat(log_dir, arm, stage):
    """Smoke copies keep the logs of one seed directly in DIR/logs."""
    path = log_dir / (f"{arm}_stage2.jsonl" if stage == 2 else f"{arm}_stage{stage}_probe.jsonl")
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("probe_acc") is not None:
            last = row["probe_acc"]
    return last


if __name__ == "__main__":                   # needed: the data loaders start worker processes
    main()
