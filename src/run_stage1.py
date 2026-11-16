"""Stage 1 for one arm and one seed. For A on seed 1, this is also the baseline.

    python src/run_stage1.py --arm A --seed 1            A learns CIFAR-10 (40k pretrain pool)
    python src/run_stage1.py --arm C --seed 1            C learns CIFAR-100 (40k)
    python src/run_stage1.py --arm B --seed 1            B: only saves its starting weights
    python src/run_stage1.py --arm A --seed 1 --smoke    1-epoch test run, saved apart from real results
    add --device cuda:1 (or mps) to use another device than the one in configs/base.yaml

Every arm builds its model right after set_seed(seed), so within a seed A, B and C
start from the same backbone weights; only stage 1 makes them different.

Writes:
    checkpoints/seed{seed}/B_stage0.pt         starting weights (written by any arm, if missing)
    checkpoints/seed{seed}/{arm}_stage1.pt     model after stage 1 (A and C)
    logs/seed{seed}/{arm}_stage1.jsonl         test accuracy and loss after every epoch
Smoke runs write to checkpoints/smoke/ and logs/smoke/ instead.

If the run is interrupted, run the same command again: it continues from the last resume file.
"""

import argparse                              # command-line options
import shutil                                # to clear old smoke-test files

import data
from data import stage1_data
from evaluate import evaluate
from model import Net
from train import train
from utils import (CHECKPOINTS, LOGS, Logger, checkpoint_path, get_device, load_config,
                   save_checkpoint, set_seed)

HEAD = {"A": "cifar10", "C": "cifar100"}     # which head each arm learns in stage 1


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 for one arm and seed.")
    parser.add_argument("--arm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="train 1 epoch and save under checkpoints/smoke/ and logs/smoke/")
    parser.add_argument("--device", default=None,
                        help="cuda:0, cuda:1 or mps (default: the device in configs/base.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    arm, seed = args.arm, args.seed
    cfg = load_config("stage1")
    if args.smoke:
        cfg["epochs"] = 1
    elif cfg["epochs"] <= 1:
        raise SystemExit("configs/stage1.yaml still says epochs: 1. Set the real number of epochs "
                         "first, or use --smoke for a test run.")

    # where this run's files go
    if args.smoke:
        final = CHECKPOINTS / "smoke" / f"seed{seed}" / f"{arm}_stage1.pt"
        log_folder = LOGS / "smoke"
        for old in (final.parent, log_folder / f"seed{seed}"):   # smoke outputs are throwaway: start clean
            shutil.rmtree(old, ignore_errors=True)
    else:
        final = checkpoint_path(arm, 1, seed)
        log_folder = LOGS
    resume = final.with_name(final.stem + ".resume.pt")

    if arm != "B" and final.exists():
        raise SystemExit(f"{final} already exists: stage 1 for arm {arm}, seed {seed} is done.")

    device = get_device(args.device or cfg["device"], cfg["gpu_memory_fraction"])
    cfg["device"] = str(device)              # record the device actually used
    data.configure(cfg["num_workers"], pin_memory=device.type == "cuda")

    # the starting weights: same for every arm within a seed
    set_seed(seed)
    net = Net().to(device)
    start = final.with_name("B_stage0.pt")
    if not start.exists():
        save_checkpoint(net, start, info={"arm": "B", "stage": 0, "seed": seed})
        print(f"saved starting weights → {start}")
    if arm == "B":
        print("arm B has no stage 1; its starting weights are saved. Done.")
        return

    net.add_head(HEAD[arm])
    train_loader, test_loader = stage1_data(arm, seed)

    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    measure_every = cfg["eval_every_epochs"] * steps_per_epoch
    measure_at = set(range(0, total_steps + 1, measure_every))

    def measure(step):
        result = evaluate(net, test_loader, device)
        return {"test_acc": result["acc"], "test_loss": result["loss"]}

    logger = Logger(f"{arm}_stage1", seed, config=cfg, folder=log_folder, resume=resume.exists(), device=device)
    print(f"arm {arm}, seed {seed}: {cfg['epochs']} epochs × {steps_per_epoch} steps = {total_steps} steps "
          f"on {device.type}{' (smoke test)' if args.smoke else ''}")

    summary = train(net, train_loader, device, cfg, total_steps, measure_at, measure, logger,
                    warmup_steps=cfg["warmup_epochs"] * steps_per_epoch,
                    resume_path=resume, resume_every=cfg["resume_every_epochs"] * steps_per_epoch)

    info = {"arm": arm, "stage": 1, "seed": seed, "epochs": cfg["epochs"],
            "steps_per_epoch": steps_per_epoch, **summary}
    save_checkpoint(net, final, info=info, config=cfg)
    print(f"\nsaved → {final}")
    print(f"test acc {summary['test_acc']:.4f} · test loss {summary['test_loss']:.4f} · "
          f"{summary['seconds'] / 60:.1f} min ({summary['seconds'] / cfg['epochs']:.1f} s per epoch)")


if __name__ == "__main__":                   # needed: the data loaders start worker processes
    main()
