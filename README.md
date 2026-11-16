# cl-relearning

Continual-learning relearning experiments. Python 3.11 (uv venv), PyTorch with MPS on Apple Silicon.

## Activate the environment

```bash
source .venv/bin/activate
```

To recreate it from scratch:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
python -m ipykernel install --user --name cl-relearning --display-name "cl-relearning"
```

## Check the environment

```bash
python check_env.py
```

Prints Python/torch/torchvision versions, checks MPS, runs a matmul on `mps`, and prints the train/test sizes of CIFAR-10, CIFAR-100 and SVHN in `data/`.

## Datasets

`data/` is gitignored, so datasets must be downloaded on each machine. CIFAR-10 and CIFAR-100 download normally with torchvision (`download=True`).

### SVHN

torchvision's SVHN source (`http://ufldl.stanford.edu/housenumbers/`) was refusing connections as of 2026-10-30, so `download=True` fails. The files in `data/svhn/` were fetched from the Internet Archive's copy of that server instead:

```bash
mkdir -p data/svhn
for f in train_32x32.mat test_32x32.mat; do
  curl -L --fail -o data/svhn/$f \
    "https://web.archive.org/web/2020id_/http://ufldl.stanford.edu/housenumbers/$f"
done
md5 data/svhn/*.mat   # on Linux: md5sum
```

The MD5s must match the ones torchvision checks (otherwise it raises "Dataset not found or corrupted"):

| File | MD5 | Size | Samples |
|---|---|---|---|
| `train_32x32.mat` | `e26dedcc434d2e4c54c9b2d4a06d8373` | 182,040,794 B | 73,257 |
| `test_32x32.mat` | `eb5a983be6a315427106f1b164d9cef3` | 64,275,384 B | 26,032 |

If the Stanford server is back up, `datasets.SVHN(..., download=True)` works as usual and gives the same files.

## JupyterLab

```bash
jupyter lab
```

Then select the **cl-relearning** kernel (Kernel → Change Kernel). Notebooks in `notebooks/` are for exploration and plotting; training runs as scripts in `src/`.

## Running on a CUDA GPU

The real runs were done on a lab CUDA GPU (PyTorch 2.15.0, CUDA build). **All runs of the experiment use the same kind of device**: MPS and CUDA give slightly different numbers, and loading a checkpoint made on the other kind of device stops with an error. The Mac is for writing code and smoke tests.

Set `device:` in `configs/base.yaml` (`cuda:0`, `cuda:1` or `mps`), or pass `--device` to a run. Before running, check that the splits match:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from utils import splits_fingerprint; print(splits_fingerprint())"
# must print f3308e9a4aafd2c0
```

If a run stops, run the same command again: it continues from the last resume file. The server setup for our lab is kept in a local `LAB.md`, which is not in the repo.
