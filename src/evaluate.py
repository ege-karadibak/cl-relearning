"""Measure a model: accuracy and loss on a loader.

Used everywhere a number is reported:
    end of every epoch / every N steps      test accuracy and loss
    stage 2, A only                         old-head test: A's "cifar10" head on CIFAR-10 test
    calibration                             accuracy on the validation split

Measuring must never change the model, so evaluation always runs in eval mode
(BatchNorm uses its saved statistics and doesn't update them) and without gradients.
"""

import torch                                 # tensors, no_grad
import torch.nn.functional as F              # cross_entropy


def evaluate(net, loader, device):
    """Accuracy and average loss of the active head on every image in `loader`.

    net:    a Net with an active head
    loader: a DataLoader (not shuffled, not augmented), e.g. the test loader from data.py
    device: from get_device()
    Returns {"acc": float between 0 and 1, "loss": float}
    """
    train_mode = net.training
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    try:
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                scores = net(images)
                # "sum", not "mean": the last batch is smaller, so averaging batch means would weigh it more
                loss_sum += F.cross_entropy(scores, labels, reduction="sum").item()
                correct += (scores.argmax(dim=1) == labels).sum().item()
                total += len(labels)
    finally:
        net.train(train_mode)                # back to train mode if it was training before
    return {"acc": correct / total, "loss": loss_sum / total}


def evaluate_head(net, loader, device, head):
    """Evaluate with another head, then switch back. Used for A's old-head test in stage 2.

    Example: evaluate_head(net, c10_test_loader, device, "cifar10") while "svhn" is active.
    """
    previous = net.active
    net.set_head(head)
    try:
        return evaluate(net, loader, device)
    finally:
        net.set_head(previous)               # always switch back, even if evaluation fails


# ---------------------------------------------------------------- quick check
if __name__ == "__main__":                   # run: python src/evaluate.py
    from model import Net
    from data import val_data
    from utils import get_device, set_seed

    device = get_device("cuda:0" if torch.cuda.is_available() else "mps")
    set_seed(1)
    net = Net().to(device)
    net.add_head("cifar10")
    net.train()
    before = {k: v.clone() for k, v in net.state_dict().items()}

    result = evaluate(net, val_data(), device)
    print("untrained model on validation:", result, "(expect acc ≈ 0.10, loss ≈ 2.3)")
    print("still in train mode:", net.training)
    print("model unchanged:", all(torch.equal(v, net.state_dict()[k]) for k, v in before.items()))

    net.add_head("svhn")
    old = evaluate_head(net, val_data(), device, "cifar10")
    print("old-head result equals the first result:", old == result, "| active head now:", net.active)
