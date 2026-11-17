"""Stage 2 for one arm and one seed: train on SVHN with a new head (the "overwrite").

    python src/run_stage2.py --arm A --seed 1                  real run (needs steps in stage2.yaml)
    python src/run_stage2.py --arm A --seed 1 --calibrate      calibration at the lr in stage2.yaml
    python src/run_stage2.py --arm A --seed 1 --calibrate --lr 0.01    calibration at another lr
    python src/run_stage2.py --arm B --seed 1 --calibrate --steps 2292  trial of one exact length
    python src/run_stage2.py --arm A --seed 1 --smoke          short test on the smoke checkpoints
    add --device cuda:1 (or mps) to use another device than the one in configs/base.yaml
    add --arch vit for the ViT experiment (configs/vit/, checkpoints/vit/, logs/vit/)

Starts from:  A, C → {arm}_stage1.pt      B → B_stage0.pt (random start, no stage 1)
Measures, every measure_every_steps:
    svhn_acc / svhn_loss            SVHN test
    old_acc / old_loss              the old head on its own data: A's "cifar10" head on CIFAR-10,
                                    C's "cifar100" head on CIFAR-100 (B has no old head)
and every probe_every_steps (plus start and end):
    probe_acc / probe_loss          CIFAR-10 linear probe (trained on never-seen, scored on test)
Writes checkpoints/seed{seed}/{arm}_stage2.pt and logs/seed{seed}/{arm}_stage2.jsonl.

Calibration (--calibrate) runs calibration_steps and measures A's old head and the probe on the
VALIDATION split instead of test (logged as old_val_acc, probe_val_acc). Choose `steps` where A's old head has dropped to about chance
(~10%) and SVHN accuracy has levelled off; write it in configs/stage2.yaml. Nothing is saved
as a checkpoint; the log goes to logs/calibration/.
"""

import argparse                              # command-line options

import data
from data import load_dataset, make_loader, stage2_data, val_data
from evaluate import evaluate, evaluate_head
from probe import Probe
from model import ARCHS
from train import train
from utils import (ROOT, Logger, checkpoint_path, config_file, get_device, load_checkpoint, load_config,
                   roots, save_checkpoint, set_seed)

OLD = {"A": ("cifar10", "cifar10"), "C": ("cifar100", "cifar100")}   # arm → (old head, its dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 (SVHN) for one arm and seed.")
    parser.add_argument("--arm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1 or mps (default: configs/base.yaml)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true", help="choose the stage-2 length on validation")
    mode.add_argument("--smoke", action="store_true", help="short test on the smoke checkpoints")
    parser.add_argument("--lr", type=float, default=None, help="learning rate to try (calibration only)")
    parser.add_argument("--steps", type=int, default=None,
                        help="length to try, with a full schedule like a real run (calibration only)")
    parser.add_argument("--arch", default="resnet18", choices=ARCHS, help="backbone (default: resnet18)")
    return parser.parse_args()


def main():
    args = parse_args()
    arm, seed, arch = args.arm, args.seed, args.arch
    cfg = load_config("stage2", arch)
    probe_cfg = load_config("probe", arch)
    stage_file = config_file("stage2", arch).relative_to(ROOT)   # e.g. configs/stage2.yaml, for messages
    if (args.lr is not None or args.steps is not None) and not args.calibrate:
        raise SystemExit(f"--lr and --steps are only for calibration; real runs use {stage_file}.")
    cfg["arch"] = arch                       # recorded with the log and the checkpoint
    ckpt_root, log_root = roots(arch)
    run_name = f"{arm}_stage2"

    # how long, and where things go
    if args.smoke:
        cfg.update(steps=200, measure_every_steps=100, probe_every_steps=100, resume_every_steps=100)
        ckpt_dir, log_folder = ckpt_root / "smoke" / f"seed{seed}", log_root / "smoke"
        start_path = ckpt_dir / ("B_stage0.pt" if arm == "B" else f"{arm}_stage1.pt")
        final = ckpt_dir / f"{arm}_stage2.pt"
        for old_file in (final, log_folder / f"seed{seed}" / f"{arm}_stage2.jsonl",
                         log_folder / f"seed{seed}" / f"{arm}_stage2.meta.json"):
            old_file.unlink(missing_ok=True)     # smoke outputs are throwaway: start clean
    elif args.calibrate:
        cfg["steps"] = cfg["calibration_steps"]
        if args.lr is not None:              # each learning rate tried gets its own log
            cfg["lr"] = args.lr
            run_name += f"_lr{args.lr:g}"
        if args.steps is not None:           # a trial of one exact length
            cfg["steps"] = args.steps
            run_name += f"_steps{args.steps}"
        start_path = checkpoint_path(arm, 0 if arm == "B" else 1, seed, arch)
        final, log_folder = None, log_root / "calibration"
    else:
        if cfg["steps"] is None:
            raise SystemExit(f"{stage_file} has no `steps` yet. Run the calibration first "
                             "(--calibrate), then set steps.")
        start_path = checkpoint_path(arm, 0 if arm == "B" else 1, seed, arch)
        final, log_folder = checkpoint_path(arm, 2, seed, arch), log_root
    if final is not None and final.exists():
        raise SystemExit(f"{final} already exists: stage 2 for arm {arm}, seed {seed} is done.")
    resume = (final.with_name(final.stem + ".resume.pt") if final is not None
              else log_root / "calibration" / f"seed{seed}" / f"{run_name}.resume.pt")

    device = get_device(args.device or cfg["device"], cfg["gpu_memory_fraction"])
    cfg["device"] = str(device)
    data.configure(cfg["num_workers"], pin_memory=device.type == "cuda", **data.augment_from(cfg))

    net, meta = load_checkpoint(start_path, device, arch=arch)
    set_seed(seed)                           # same new head (and data order) for every arm within a seed
    net.add_head("svhn")                     # A's / C's old head stays in the model, untouched
    train_loader, svhn_test = stage2_data(seed)

    # what to measure the old head on: test for real runs, validation during calibration
    old = None
    if arm in OLD:
        head, dataset = OLD[arm]
        if args.calibrate:
            old = (head, val_data()) if dataset == "cifar10" else None   # CIFAR-100 has no validation split
        else:
            old = (head, make_loader(load_dataset(dataset, train=False, augment=False), shuffle=False))
    probe = Probe(device, probe_cfg, score_on="val" if args.calibrate else "test")

    total = cfg["steps"]
    every, probe_every = cfg["measure_every_steps"], cfg["probe_every_steps"]
    measure_at = set(range(0, total + 1, every)) | set(range(0, total + 1, probe_every)) | {total}

    def measure(step):
        r = evaluate(net, svhn_test, device)
        out = {"svhn_acc": r["acc"], "svhn_loss": r["loss"]}
        if old is not None:
            r = evaluate_head(net, old[1], device, old[0])
            name = "old_val" if args.calibrate else "old"       # calibration measures on validation
            out.update({f"{name}_acc": r["acc"], f"{name}_loss": r["loss"]})
        if step % probe_every == 0 or step == total:
            out.update(probe(net))
        return out

    logger = Logger(run_name, seed, config={**cfg, "probe": probe_cfg}, folder=log_folder,
                    resume=resume.exists(), device=device)
    mode = f" (calibration, lr {cfg['lr']:g})" if args.calibrate else " (smoke test)" if args.smoke else ""
    print(f"arm {arm}, seed {seed}: {total} SVHN steps from {start_path.name} on {device.type}{mode}")

    summary = train(net, train_loader, device, cfg, total, measure_at, measure, logger,
                    warmup_steps=cfg["warmup_steps"], resume_path=resume,
                    resume_every=cfg["resume_every_steps"])

    if final is not None:
        info = {"arm": arm, "stage": 2, "seed": seed, "started_from": start_path.name, **summary}
        save_checkpoint(net, final, info=info, config={**cfg, "probe": probe_cfg})
        print(f"\nsaved → {final}")
    print("final:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})


if __name__ == "__main__":                   # needed: the data loaders start worker processes
    main()
