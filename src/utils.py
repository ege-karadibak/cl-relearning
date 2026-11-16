"""Small shared tools: device, seeding, checkpoints, logging, configs.

Every run script uses these, so all runs are set up, saved and logged the same way.

    checkpoints/seed{seed}/{arm}_stage{stage}.pt      model at the end of each stage
                                                      (stage 0 = starting weights, used for B)
    logs/seed{seed}/{run_name}.jsonl                  one line of metrics per evaluation
    logs/seed{seed}/{run_name}.meta.json              what produced the run (config, split, versions)

Every checkpoint and every log records where it came from: the config, a fingerprint
of data/splits.npz, the device, library versions and the date. Loading a checkpoint stops
if the split on disk is not the one it was trained with, or if it was made on another
kind of device (MPS vs CUDA give slightly different numbers; one experiment uses one kind).

Device: set in configs/base.yaml ("cuda:0", "cuda:1" or "mps"), or per run with --device.
On CUDA, the process is limited to gpu_memory_fraction of the GPU (lab rule: 0.33 = 32 GB).
"""

from pathlib import Path                     # file paths
from datetime import datetime                # date stamp
import hashlib                               # fingerprint of splits.npz
import json                                  # log lines are JSON
import platform                              # Python version
import random                                # Python's own random numbers (seeded too)
import numpy as np                           # numpy random numbers (seeded too)
import torch                                 # tensors, saving, MPS
import torchvision                           # version, for provenance
import yaml                                  # reading configs/*.yaml

from model import Net                        # to rebuild a model when loading a checkpoint

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parent.parent   # project root (works from any folder)
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
CONFIGS = ROOT / "configs"
SPLITS = ROOT / "data" / "splits.npz"

REQUIRED_INFO = ("arm", "stage", "seed")     # every checkpoint must say what it is


# ---------------------------------------------------------------- device
def get_device(name=None, memory_fraction=None):
    """Return the device to train on. Never falls back to another device.

    name:            "cuda:0", "cuda:1" or "mps"; None = the "device" in configs/base.yaml
    memory_fraction: CUDA only, share of the GPU this process may use;
                     None = "gpu_memory_fraction" in configs/base.yaml (lab rule: 0.33 = 32 GB)
    """
    if name is None or memory_fraction is None:
        with open(CONFIGS / "base.yaml") as f:
            base = yaml.safe_load(f) or {}
        name = name or base.get("device", "mps")
        memory_fraction = memory_fraction if memory_fraction is not None else base.get("gpu_memory_fraction", 0.33)

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available. Stopping instead of switching device.")
        return torch.device("mps")

    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Stopping instead of switching device.")
        device = torch.device(name)
        if device.index is None:
            raise ValueError("say which GPU to use: 'cuda:0' or 'cuda:1'")
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f"{name} doesn't exist; this machine has {torch.cuda.device_count()} GPU(s)")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        return device

    raise ValueError(f"unknown device {name!r}, use 'cuda:0', 'cuda:1' or 'mps'")


def device_info(device):
    """Plain description of a device, for provenance."""
    device = torch.device(device)
    if device.type == "cuda":
        return {"device_type": "cuda", "device_name": torch.cuda.get_device_name(device), "cuda": torch.version.cuda}
    return {"device_type": device.type, "device_name": platform.machine()}


# ---------------------------------------------------------------- seeding
def set_seed(seed):
    """Seed every random number generator, so a run can be repeated.

    Covers: Python's random, numpy, and torch (which also seeds MPS).
    Call it once at the start of every run, before building the model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- provenance
def splits_fingerprint():
    """Short fingerprint of data/splits.npz. Same file → same fingerprint."""
    if not SPLITS.exists():
        raise FileNotFoundError(f"{SPLITS} not found. Run: python src/make_splits.py")
    return hashlib.sha256(SPLITS.read_bytes()).hexdigest()[:16]


def provenance(device=None):
    """Where a result came from: split fingerprint, device, library versions, date."""
    return {
        "splits_sha256": splits_fingerprint(),
        **(device_info(device) if device is not None else {}),
        "torch": str(torch.__version__),             # str(): plain text, so safe loading accepts it
        "torchvision": str(torchvision.__version__),
        "python": platform.python_version(),
        "date": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- checkpoints
def checkpoint_path(arm, stage, seed):
    """Where one model is saved, e.g. checkpoints/seed1/A_stage1.pt.

    arm: "A", "B" or "C"
    stage: 0 (starting weights, before any training), 1 or 2
    seed: 1, 2, 3, ...
    """
    if arm not in ("A", "B", "C"):
        raise ValueError(f"unknown arm {arm!r}, use 'A', 'B' or 'C'")
    if stage not in (0, 1, 2):
        raise ValueError(f"unknown stage {stage!r}, use 0, 1 or 2")
    return CHECKPOINTS / f"seed{seed}" / f"{arm}_stage{stage}.pt"


def save_checkpoint(net, path, info, config=None):
    """Save a model with all its heads, plus what produced it.

    net:    a Net
    path:   from checkpoint_path(...)
    info:   dict with at least "arm", "stage", "seed"; add results too, e.g. "test_acc", "steps"
    config: the settings used for this stage (from load_config)
    """
    missing = [k for k in REQUIRED_INFO if k not in info]
    if missing:
        raise ValueError(f"info is missing {missing}")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; a finished stage is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": net.state_dict(),           # all weights: backbone + every head
        "heads": list(net.heads),            # head names, in order, to rebuild them when loading
        "active": net.active,                # which head was in use
        "info": info,
        "config": config or {},
        "provenance": provenance(next(net.parameters()).device),
    }, path)


def load_checkpoint(path, device, allow_other_device=False):
    """Load a saved model. Returns (net, meta).

    meta = {"info": ..., "config": ..., "provenance": ...}
    Stops if data/splits.npz is not the file the model was trained with, or if the model
    was made on another kind of device (allow_other_device=True only for looking at results,
    never for continuing the experiment).
    The saved weights include heads, so the heads are added before loading.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)   # weights_only: never runs code from the file
    saved, current = ckpt["provenance"]["splits_sha256"], splits_fingerprint()
    if saved != current:
        raise RuntimeError(f"{path} was trained with splits {saved}, but data/splits.npz is now {current}")
    made_on = ckpt["provenance"].get("device_type")
    if made_on is not None and made_on != torch.device(device).type and not allow_other_device:
        raise RuntimeError(f"{path} was made on {made_on}, but this run uses {torch.device(device).type}; "
                           "one experiment must use one kind of device")
    net = Net().to(device)
    for name in ckpt["heads"]:
        net.add_head(name)
    net.load_state_dict(ckpt["state"])      # strict: fails loudly if anything doesn't match
    if ckpt["active"] is not None:
        net.set_head(ckpt["active"])
    meta = {k: ckpt[k] for k in ("info", "config", "provenance")}
    return net, meta


# ---------------------------------------------------------------- logging
class Logger:
    """Write metrics to a .jsonl file: one JSON object per line.

    Example line: {"step": 500, "test_acc": 0.71, "train_loss": 0.83}
    Read it back later with pandas: pd.read_json(path, lines=True)
    Next to it, {run_name}.meta.json records the run name, seed, config and provenance.
    """

    def __init__(self, run_name, seed, config=None, folder=LOGS, resume=False, device=None):
        """resume=True: continue the log of an interrupted run instead of refusing it."""
        folder = Path(folder) / f"seed{seed}"
        self.path = folder / f"{run_name}.jsonl"
        self.meta_path = folder / f"{run_name}.meta.json"
        if resume and self.meta_path.exists():
            return                           # keep the existing log and its metadata
        if self.path.exists() or self.meta_path.exists():
            raise FileExistsError(f"{self.path} already exists; don't mix two runs in one log")
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"run": run_name, "seed": seed, "config": config or {}, "provenance": provenance(device)}
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def drop_after(self, step):
        """Remove lines logged after `step`. Used when resuming: those steps are trained again."""
        if not self.path.exists():
            return
        lines = [l for l in self.path.read_text().splitlines() if json.loads(l).get("step", 0) <= step]
        self.path.write_text("".join(l + "\n" for l in lines))

    def log(self, **metrics):
        """Append one line, e.g. logger.log(step=500, test_acc=0.71)."""
        line = json.dumps(metrics, default=float)   # default=float turns tensors / numpy numbers into plain numbers
        with open(self.path, "a") as f:
            f.write(line + "\n")
        print(line)


# ---------------------------------------------------------------- configs
def load_config(stage_name):
    """Read configs/base.yaml, then configs/{stage_name}.yaml on top of it.

    stage_name: "stage1", "stage2", "stage3" or "probe"
    Settings in the stage file replace the same settings from base.yaml.
    Returns a dict.
    """
    with open(CONFIGS / "base.yaml") as f:
        base = yaml.safe_load(f) or {}      # empty file → {}
    with open(CONFIGS / f"{stage_name}.yaml") as f:
        stage = yaml.safe_load(f) or {}
    return {**base, **stage}


# ---------------------------------------------------------------- quick check
if __name__ == "__main__":                   # run: python src/utils.py
    import tempfile                          # a throwaway folder, so the check never touches checkpoints/ or logs/

    device = get_device("cuda:0" if torch.cuda.is_available() else "mps")
    print("device:", device)
    print("provenance:", provenance(device))

    set_seed(1)
    a = torch.rand(3)
    set_seed(1)
    b = torch.rand(3)
    print("same seed, same numbers:", torch.equal(a, b))

    with tempfile.TemporaryDirectory() as tmp:
        set_seed(1)
        start = Net().to(device)                                     # B's starting weights (no head yet)
        save_checkpoint(start, Path(tmp) / "B_stage0.pt", info={"arm": "B", "stage": 0, "seed": 1})
        b0, _ = load_checkpoint(Path(tmp) / "B_stage0.pt", device)
        print("stage-0 (no heads) round trip:", list(b0.heads) == [] and b0.active is None)

        net = Net().to(device)
        net.add_head("cifar10")
        net.add_head("svhn")
        path = Path(tmp) / "test.pt"
        save_checkpoint(net, path, info={"arm": "A", "stage": 2, "seed": 1}, config={"lr": 0.1})
        loaded, meta = load_checkpoint(path, device)
        same = all(torch.equal(v, loaded.state_dict()[k]) for k, v in net.state_dict().items())
        print("checkpoint round trip identical:", same, "| heads:", list(loaded.heads), "| active:", loaded.active)
        print("meta:", meta["info"], meta["config"], meta["provenance"]["splits_sha256"])

        logger = Logger("test_run", seed=1, config={"lr": 0.1}, folder=tmp)
        logger.log(step=0, test_acc=torch.tensor(0.1))
        logger.log(step=500, test_acc=0.71)
        print("log meta:", json.loads(logger.meta_path.read_text())["config"])

    print("config (empty files for now):", load_config("stage1"))
