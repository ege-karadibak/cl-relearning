"""The training loop shared by every stage.

One loop, counted in steps (one step = one batch), so all three stages fit:
    stage 1   cosine LR from 0.1, e.g. 200 epochs × 313 steps
    stage 2   cosine LR from 0.1, a fixed number of SVHN steps (from calibration)
    stage 3   warmup, then a constant LR, a fixed step budget on 5k images

The run scripts turn epochs into steps (epochs × len(train_loader)) and decide what to
measure. The loop calls `measure(step)` at the chosen steps and logs whatever it returns.

What it trains: the backbone + the ACTIVE head only. Other heads (e.g. A's old
"cifar10" head during stage 2) are left out of the optimizer and stay unchanged.
Every stage starts with a fresh optimizer and schedule.
"""

import math                                  # cos, for the cosine schedule
import random                                # to save / restore Python's random state on resume
import time                                  # seconds per run
from pathlib import Path                     # file paths
import numpy as np                           # to save / restore numpy's random state on resume
import torch                                 # tensors, optimizer, schedule
import torch.nn.functional as F              # cross_entropy


# ---------------------------------------------------------------- optimizer and schedule
def make_optimizer(net, cfg):
    """SGD over the backbone + the active head, with the settings from the config.

    cfg keys: lr, momentum, weight_decay, nesterov (optional, default False)
    """
    params = list(net.backbone.parameters()) + list(net.heads[net.active].parameters())
    return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"], nesterov=cfg.get("nesterov", False))


def lr_factor(step, total_steps, warmup_steps, schedule):
    """How much of the base LR to use at `step` (a number between 0 and 1).

    warmup:   during the first warmup_steps, grow linearly from almost 0 to 1
    "cosine": after warmup, go smoothly from 1 down to 0 at total_steps
    "constant": after warmup, stay at 1
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {schedule!r}, use 'cosine' or 'constant'")


def make_scheduler(optimizer, total_steps, warmup_steps, schedule):
    """A LambdaLR that multiplies the base LR by lr_factor(step, ...) at every step."""

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup_steps, schedule)
    )


# ---------------------------------------------------------------- batches
def batches(loader):
    """Yield (epoch, images, labels) forever, starting a new shuffled epoch each time the loader ends."""
    epoch = 0
    while True:
        for images, labels in loader:        # a shuffled loader gives a new order every pass
            yield epoch, images, labels
        epoch += 1


# ---------------------------------------------------------------- resume files
def save_resume(path, net, optimizer, scheduler, step):
    """Save everything needed to continue an interrupted run from `step`.

    Not a result: it is deleted when the run finishes. Saves the model, optimizer
    (momentum), schedule, step and the random-number states.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "state": net.state_dict(),           # backbone + every head
        "heads": list(net.heads),
        "active": net.active,
        "optimizer": optimizer.state_dict(), # includes momentum
        "scheduler": scheduler.state_dict(), # where the LR schedule is
        "step": step,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "device": device_rng_state(next(net.parameters()).device),
        },
    }, tmp)
    tmp.replace(path)                        # rename: the old file is swapped for the new one in one go


def load_resume(path, net, optimizer, scheduler):
    """Continue from a resume file: restore everything and return the step to continue from.

    The data order after resuming differs a little from an uninterrupted run; fine for crash recovery.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)   # our own file; it holds random states
    if ckpt["heads"] != list(net.heads) or ckpt["active"] != net.active:
        raise RuntimeError(f"resume file {path} has heads {ckpt['heads']} (active {ckpt['active']!r}), "
                           f"but the model has {list(net.heads)} (active {net.active!r})")
    net.load_state_dict(ckpt["state"])       # copies the weights onto the model's device
    optimizer.load_state_dict(ckpt["optimizer"])   # moves momentum to the device of the weights
    scheduler.load_state_dict(ckpt["scheduler"])
    random.setstate(ckpt["rng"]["python"])
    np.random.set_state(ckpt["rng"]["numpy"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    set_device_rng_state(next(net.parameters()).device, ckpt["rng"]["device"])
    return ckpt["step"]


def device_rng_state(device):
    """The GPU's own random state (MPS or CUDA)."""
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng_state(device, state):
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)


# ---------------------------------------------------------------- the loop
def train(net, train_loader, device, cfg, total_steps, measure_at, measure, logger,
          warmup_steps=0, resume_path=None, resume_every=None):
    """Train the active head + backbone for total_steps. Returns a summary dict.

    net:          a Net with the head to train already added and active
    train_loader: shuffled, augmented loader from data.py
    cfg:          the stage config (lr, momentum, weight_decay, schedule)
    total_steps:  how many batches to train on
    measure_at:   steps at which to call measure, e.g. {0, 313, 626, ...}; the last step is always added
    measure:      function(step) → dict of numbers, e.g. {"test_acc": ..., "test_loss": ...}
                  (the run script decides what: test set, old head, probe, ...)
    logger:       a utils.Logger
    warmup_steps: 0 in stages 1–2; > 0 in stage 3
    resume_path:  where to keep the resume file (None = no resume file)
    resume_every: save the resume file every this many steps

    Each log line holds: step, epoch, lr, train_loss and train_acc (averaged since the
    last log line), seconds so far, and everything measure(step) returned.
    """
    optimizer = make_optimizer(net, cfg)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps, cfg["schedule"])
    measure_at = set(measure_at) | {total_steps}
    steps_per_epoch = len(train_loader)

    start = 0
    if resume_path is not None and Path(resume_path).exists():
        start = load_resume(resume_path, net, optimizer, scheduler)
        logger.drop_after(start)             # steps after the resume point are trained (and logged) again
        print(f"resumed from step {start}")

    last = {}
    if start == 0 and 0 in measure_at:       # the starting point, before any training
        last = measure(0)
        logger.log(step=0, epoch=0.0, lr=scheduler.get_last_lr()[0],
                   train_loss=None, train_acc=None, seconds=0.0, **last)

    net.train()
    t0 = time.time()
    # running sums stay on the device: reading a number back every step would make the GPU wait
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    count = 0
    it = batches(train_loader)

    for step in range(start + 1, total_steps + 1):
        _, images, labels = next(it)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        scores = net(images)
        loss = F.cross_entropy(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.detach() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum()
        count += len(labels)

        if step in measure_at:
            train_loss = loss_sum.item() / count
            if not math.isfinite(train_loss):
                raise RuntimeError(f"training loss is {train_loss} at step {step}: something blew up")
            last = measure(step)
            net.train()                      # measure may have switched to eval mode
            logger.log(step=step, epoch=round(step / steps_per_epoch, 3), lr=scheduler.get_last_lr()[0],
                       train_loss=train_loss, train_acc=correct.item() / count,
                       seconds=round(time.time() - t0, 1), **last)
            loss_sum.zero_()
            correct.zero_()
            count = 0

        if resume_path is not None and resume_every and step % resume_every == 0 and step < total_steps:
            save_resume(resume_path, net, optimizer, scheduler, step)

    if resume_path is not None:
        Path(resume_path).unlink(missing_ok=True)   # the run finished; the resume file is no longer needed
    return {"steps": total_steps, "seconds": round(time.time() - t0, 1), **last}


# ---------------------------------------------------------------- quick check
if __name__ == "__main__":                   # run: python src/train.py
    import json
    import tempfile
    from model import Net
    from data import stage1_data, val_data
    from evaluate import evaluate
    from utils import get_device, set_seed, Logger

    # 1. the LR schedule
    f = lambda s, n, w, sch: round(lr_factor(s, n, w, sch), 4)
    print("cosine:  start", f(0, 100, 0, "cosine"), "| half", f(50, 100, 0, "cosine"), "| end", f(100, 100, 0, "cosine"),
          "(expect 1.0, 0.5, 0.0)")
    print("warmup:  step 0", f(0, 100, 10, "constant"), "| step 9", f(9, 100, 10, "constant"),
          "| step 50", f(50, 100, 10, "constant"), "(expect 0.1, 1.0, 1.0)")

    device = get_device("cuda:0" if torch.cuda.is_available() else "mps")
    cfg = {"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4, "schedule": "cosine"}
    train_loader, _ = stage1_data("A", 1)
    val = val_data()
    measure = lambda step: {f"val_{k}": v for k, v in evaluate(net, val, device).items()}

    with tempfile.TemporaryDirectory() as tmp:
        # 2. a short real run, with an inactive head that must not change
        set_seed(1)
        net = Net().to(device)
        net.add_head("svhn")
        net.add_head("cifar10")
        svhn_before = net.heads["svhn"].weight.detach().clone()
        log = Logger("quick", seed=1, folder=tmp)
        summary = train(net, train_loader, device, cfg, total_steps=50, measure_at={0, 25}, measure=measure, logger=log)
        print("summary:", summary)
        print("inactive head unchanged:", torch.equal(svhn_before, net.heads["svhn"].weight))

        # 3. crash at step 15, then resume from the file saved at step 10
        set_seed(2)
        net = Net().to(device)
        net.add_head("cifar10")
        resume = Path(tmp) / "resume.pt"
        log = Logger("crash", seed=1, folder=tmp)
        def crash_at_15(step):
            if step == 15:
                raise KeyboardInterrupt("simulated crash")
            return {"val_acc": 0.0}
        try:
            train(net, train_loader, device, cfg, 20, set(range(0, 21, 5)), crash_at_15, log,
                  resume_path=resume, resume_every=10)
        except KeyboardInterrupt:
            print("crashed at step 15; resume file exists:", resume.exists())
        set_seed(2)
        net = Net().to(device)
        net.add_head("cifar10")
        log = Logger("crash", seed=1, folder=tmp, resume=True)
        train(net, train_loader, device, cfg, 20, set(range(0, 21, 5)), lambda s: {"val_acc": 0.0}, log,
              resume_path=resume, resume_every=10)
        steps = [json.loads(l)["step"] for l in open(log.path)]
        print("logged steps:", steps, "(expect 0, 5, 10, 15, 20 once each)")
        print("resume file removed at the end:", not resume.exists())
