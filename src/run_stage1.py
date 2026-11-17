"""Stage 1 for one arm and one seed. For A on seed 1, this is also the baseline.

    python src/run_stage1.py --arm A --seed 1            A learns CIFAR-10 (40k pretrain pool)
    python src/run_stage1.py --arm C --seed 1            C learns CIFAR-100 (40k)
    python src/run_stage1.py --arm B --seed 1            B: only saves its starting weights
    python src/run_stage1.py --arm A --seed 1 --smoke    1-epoch test run, saved apart from real results
    python src/run_stage1.py --arm A --seed 1 --calibrate --lr 3e-4    calibration: one learning rate
    add --device cuda:1 (or mps) to use another device than the one in configs/base.yaml
    add --arch vit for the ViT experiment (configs/vit/, checkpoints/vit/, logs/vit/)

Every arm builds its model right after set_seed(seed), so within a seed A, B and C
start from the same backbone weights; only stage 1 makes them different.

Writes:
    checkpoints/seed{seed}/B_stage0.pt         starting weights (written by any arm, if missing)
    checkpoints/seed{seed}/{arm}_stage1.pt     model after stage 1 (A and C)
    logs/seed{seed}/{arm}_stage1.jsonl         test accuracy and loss after every epoch
Smoke runs write to checkpoints/smoke/ and logs/smoke/ instead.

Calibration (--calibrate, arm A only) trains the full stage 1 at one learning rate and scores it on the
CIFAR-10 VALIDATION split instead of test (logged as val_acc, val_loss). Nothing is saved as a
checkpoint and no starting weights are written; the log goes to logs/calibration/. Compare the runs,
then write the chosen lr in the stage-1 config. A stage file with `calibrated: false` refuses real runs.

If the run is interrupted, run the same command again: it continues from the last resume file.
"""

import argparse                              # command-line options
import shutil                                # to clear old smoke-test files

import data
from data import stage1_data, val_data
from evaluate import evaluate
from model import Net
from train import train
from model import ARCHS
from utils import (ROOT, Logger, checkpoint_path, config_file, get_device, load_config, roots,
                   save_checkpoint, set_seed)

HEAD = {"A": "cifar10", "C": "cifar100"}     # which head each arm learns in stage 1


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 for one arm and seed.")
    parser.add_argument("--arm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--seed", required=True, type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="train 1 epoch and save under checkpoints/smoke/ and logs/smoke/ (ViT: .../vit/smoke/)")
    mode.add_argument("--calibrate", action="store_true", help="score on validation, save nothing; use with --lr")
    parser.add_argument("--lr", type=float, default=None, help="learning rate to try (calibration only)")
    parser.add_argument("--device", default=None,
                        help="cuda:0, cuda:1 or mps (default: the device in configs/base.yaml)")
    parser.add_argument("--arch", default="resnet18", choices=ARCHS, help="backbone (default: resnet18)")
    return parser.parse_args()


def main():
    args = parse_args()
    arm, seed, arch = args.arm, args.seed, args.arch
    cfg = load_config("stage1", arch)
    stage_file = config_file("stage1", arch).relative_to(ROOT)   # e.g. configs/stage1.yaml, for messages
    if args.lr is not None and not args.calibrate:
        raise SystemExit(f"--lr is only for calibration; real runs use the lr in {stage_file}.")
    if args.calibrate:
        if arm != "A":
            raise SystemExit("stage-1 calibration uses arm A only: it is scored on the CIFAR-10 validation split.")
        if args.lr is None:
            raise SystemExit(f"--calibrate needs --lr (one of lr_choices in {stage_file}).")
        cfg["lr"] = args.lr
    elif not args.smoke and not cfg.get("calibrated", True):
        raise SystemExit(f"{stage_file} is not calibrated yet: choose lr with --calibrate, then set calibrated: true.")
    if args.smoke:
        cfg["epochs"] = 1
    elif cfg["epochs"] <= 1:
        raise SystemExit(f"{stage_file} still says epochs: 1. Set the real "
                         "number of epochs first, or use --smoke for a test run.")
    cfg["arch"] = arch                       # recorded with the log and the checkpoint
    ckpt_root, log_root = roots(arch)

    # where this run's files go
    run_name = f"{arm}_stage1"
    if args.smoke:
        final = ckpt_root / "smoke" / f"seed{seed}" / f"{arm}_stage1.pt"
        log_folder = log_root / "smoke"
        for old in (final.parent, log_folder / f"seed{seed}"):   # smoke outputs are throwaway: start clean
            shutil.rmtree(old, ignore_errors=True)
    elif args.calibrate:
        run_name += f"_lr{args.lr:g}"        # each learning rate tried gets its own log
        final, log_folder = None, log_root / "calibration"
    else:
        final = checkpoint_path(arm, 1, seed, arch)
        log_folder = log_root
    resume = (final.with_name(final.stem + ".resume.pt") if final is not None
              else log_folder / f"seed{seed}" / f"{run_name}.resume.pt")

    if final is not None and arm != "B" and final.exists():
        raise SystemExit(f"{final} already exists: stage 1 for arm {arm}, seed {seed} is done.")

    device = get_device(args.device or cfg["device"], cfg["gpu_memory_fraction"])
    cfg["device"] = str(device)              # record the device actually used
    data.configure(cfg["num_workers"], pin_memory=device.type == "cuda", **data.augment_from(cfg))

    # the starting weights: same for every arm within a seed
    set_seed(seed)
    net = Net(arch).to(device)
    if final is not None:                    # calibration never writes into the real checkpoint folders
        start = final.with_name("B_stage0.pt")
        if not start.exists():
            save_checkpoint(net, start, info={"arm": "B", "stage": 0, "seed": seed})
            print(f"saved starting weights → {start}")
    if arm == "B":
        print("arm B has no stage 1; its starting weights are saved. Done.")
        return

    net.add_head(HEAD[arm])
    train_loader, test_loader = stage1_data(arm, seed)
    score_loader, prefix = (val_data(), "val") if args.calibrate else (test_loader, "test")

    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    measure_every = cfg["eval_every_epochs"] * steps_per_epoch
    measure_at = set(range(0, total_steps + 1, measure_every))

    def measure(step):
        result = evaluate(net, score_loader, device)
        return {f"{prefix}_acc": result["acc"], f"{prefix}_loss": result["loss"]}

    logger = Logger(run_name, seed, config=cfg, folder=log_folder, resume=resume.exists(), device=device)
    note = " (smoke test)" if args.smoke else f" (calibration, lr {cfg['lr']:g})" if args.calibrate else ""
    print(f"arm {arm}, seed {seed}: {cfg['epochs']} epochs × {steps_per_epoch} steps = {total_steps} steps "
          f"on {device.type}{note}")

    summary = train(net, train_loader, device, cfg, total_steps, measure_at, measure, logger,
                    warmup_steps=cfg["warmup_epochs"] * steps_per_epoch,
                    resume_path=resume, resume_every=cfg["resume_every_epochs"] * steps_per_epoch)

    if final is None:                        # calibration: the log is the result
        print(f"\nval acc {summary['val_acc']:.4f} · val loss {summary['val_loss']:.4f} · "
              f"{summary['seconds'] / 60:.1f} min → {logger.path}")
        return

    info = {"arm": arm, "stage": 1, "seed": seed, "epochs": cfg["epochs"],
            "steps_per_epoch": steps_per_epoch, **summary}
    save_checkpoint(net, final, info=info, config=cfg)
    print(f"\nsaved → {final}")
    print(f"test acc {summary['test_acc']:.4f} · test loss {summary['test_loss']:.4f} · "
          f"{summary['seconds'] / 60:.1f} min ({summary['seconds'] / cfg['epochs']:.1f} s per epoch)")


if __name__ == "__main__":                   # needed: the data loaders start worker processes
    main()
