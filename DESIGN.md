# Experiment design: does pretraining make re-learning faster?

Model: ResNet-18 (CIFAR stem, see §7). Status: design. No code written yet.

## 1. Question

After training on unrelated data makes a model appear to forget its original data D, does it relearn D faster and better than models that never saw D? If so, does the advantage come from **retaining D specifically** or just from **having been pretrained at all**, and is what's retained **the specific training images** or **general knowledge of the domain**??

## 2. Arms

| Arm | Stage 1 (pretrain) | Stage 2 (overwrite) | Stage 3 (relearn) |
|---|---|---|---|
| **A** | D = CIFAR-10 (40k) | D_or = SVHN | 5k subset of CIFAR-10 |
| **B** | none (random init) | D_or = SVHN | 5k subset of CIFAR-10 |
| **C** | D_other = CIFAR-100 (40k) | D_or = SVHN | 5k subset of CIFAR-10 |

- **A vs C** isolates retention of D specifically.
- **C vs B** isolates the generic benefit of pretraining.

CIFAR-100 is kept as D_other on purpose. It is close to CIFAR-10 (natural photos, related classes), so C is a strict control: if A still beats C, the advantage comes from D itself, not from having seen natural images. The expected cost is a small A–C gap, so it needs enough seeds. A distant D_other (e.g. textures) is a possible later arm.

### Reading the outcomes

| Result | Interpretation |
|---|---|
| A > C > B | retention plus a generic pretraining benefit |
| A > C = B | retention only. Unrelated pretraining doesn't help |
| A = C > B | generic pretraining benefit only, no retention |
| A = C = B | neither |
| B ≥ A or B ≥ C | pretraining hurts re-learning (e.g. lost plasticity) |

"=" means the difference is within the spread across seeds (§6).

## 3. Data

| Pool | Source | Size | Use |
|---|---|---|---|
| CIFAR-10 pretrain | CIFAR-10 train | 40,000 | A stage 1; **A-seen** subsets drawn from here |
| CIFAR-10 held out | CIFAR-10 train | 10,000 | **never-seen** subsets (+ validation, see §8.2) |
| CIFAR-10 test | CIFAR-10 test | 10,000 | all reported CIFAR-10 numbers |
| CIFAR-100 pretrain | CIFAR-100 train | 40,000 (see §8.1) | C stage 1 |
| SVHN | SVHN train / test | 73,257 / 26,032 | stage 2 for all arms |

**Relearning subset:** 5,000 images, 500 per class (10% of CIFAR-10). There are two versions, named by **where the images come from**, not by what each arm has seen:
- **A-seen subset:** drawn from the 40k pretrain pool, the images A trained on in stage 1.
- **never-seen subset:** drawn from the 10k held-out pool. No arm ever trains on these before stage 3.

Only A has seen the A-seen images. For B and C, **both subsets are new images**: B never trained on CIFAR-10, and C trained on CIFAR-100, which shares no images with CIFAR-10.

| Subset | A | B | C |
|---|---|---|---|
| A-seen 5k | **seen in stage 1** | new | new |
| never-seen 5k | new | new | new |

**Why B and C run on both:** they are the control for subset difficulty. If A relearns faster on A-seen than on never-seen, that could be memory of those images, or the A-seen subset could just be easier. B and C can't remember those images, so any gap they show between the two subsets comes from difficulty alone, and it is subtracted out (§5, A-seen vs never-seen analysis).

- Within a seed, all arms use exactly the same subset indices. Across seeds, a new subset is drawn, so the seeds also capture sampling variation.
- Every arm runs on both subsets: 3 arms × 2 subsets × seeds.

Extensions for later: 1% and 5% subsets.

## 4. Classifier heads

- One head per dataset: 10-way CIFAR-10, 10-way SVHN, 100-way CIFAR-100. No shared head.
- Stage 3: **every arm gets a freshly initialized 10-way CIFAR-10 head**, so only the feature extractor is compared and every arm starts the same at the head.
- A's stage-1 CIFAR-10 head is saved and used **only for diagnosis** (§5).
- Possible later variant: A reusing its saved head in stage 3.

Fresh heads narrow the claim to "the **features** retain knowledge that speeds up relearning," not "the whole model relearns faster." Any retention that lives in how the backbone and old head fit together is thrown away, so this setup can underestimate A's advantage but can't fake one. If the main result is weak, or the whole-model claim is needed, run the A-reuse variant.

## 5. Metrics

**After stage 2 (the starting point for stage 3)**
- **Linear probe accuracy** on the frozen backbone, for all arms. This is the step-0 measure. The probe is trained on the **never-seen** subset and evaluated on the CIFAR-10 test set (protocol in §8.6).
  Note: test accuracy and loss with the fresh head at step 0 are about chance (≈10%, loss ≈ ln 10 ≈ 2.30) for every arm, so they carry no information and are not used.
- Optional: k-NN accuracy on the backbone features (no training needed).
- **A only, diagnostic:** CIFAR-10 test accuracy using A's saved stage-1 head on the post-SVHN backbone ("how forgotten A looks"). Compare it with A's probe ("how much is actually retained").
- **SVHN test accuracy at the end of stage 2**, for all arms, to check they were overwritten by comparable amounts.

**During stage 3**
- Full learning curve: CIFAR-10 test loss and accuracy every N steps, evaluated more often early on.
- **Steps to target:** first step reaching each of several fixed CIFAR-10 test-accuracy thresholds, provisionally **50%, 70%, 80%** (to be confirmed after the baseline, §9). The thresholds are absolute, so they are the same for every arm and seed and don't depend on A's stage-1 accuracy. Runs that never reach a threshold are reported as "not reached by step X" (§8.4), not dropped.
- Area under the accuracy curve.
- Final test loss and accuracy.

**Aligned curves (started closer vs learns faster)**
Shift each arm's stage-3 curve so that step 0 is the first step at which it reaches a common accuracy (e.g. 50%), then compare the curves and their slopes from that point on. A raw-curve advantage shows that an arm *started closer*; an advantage that remains after alignment shows that it *learns faster*. This needs dense early evaluation, and a **constant stage-3 learning rate** (§7): with a decaying schedule, an arm that reaches the alignment point later would be at a lower LR there, which would bias its slope downward.

**Probe vs stage-3 curves (was retention lost early in stage 3?)**
The probe measures what each arm has before stage 3 touches the backbone, so comparing it with the stage-3 curves separates "nothing was retained" from "it was retained but damaged at the start of relearning" (see the warmup in §7):

| Probe after stage 2 | Stage-3 curves | Meaning |
|---|---|---|
| A ≫ B | A ≫ B | retention, and relearning uses it |
| A ≫ B | A ≈ B | retention existed but was lost early in stage 3. Check the warmup and rerun |
| A ≈ B | A ≈ B | nothing was retained |

Also watch the first evaluations of stage 3: if A's accuracy dips before it rises, early feature damage is happening.

**A-seen vs never-seen analysis (does A remember specific images?)**
For each arm, take the gap between its two stage-3 runs:

  gap(arm) = result on A-seen − result on never-seen

- gap(A) = memory of specific images + subset difficulty
- gap(B), gap(C) = subset difficulty only (they have seen neither subset)

The memory effect is what remains after subtracting the difficulty part:

  memory effect = gap(A) − gap(B)   (and the same with gap(C))

Example: gap(A) = +4 points, gap(B) = +1 point → 3 points come from A remembering those images, and 1 point is just the A-seen subset being easier. A positive memory effect means A retained *specific images*, not just general knowledge of CIFAR-10.

## 6. Seeds

- 3 seeds for the main run. Increase to 5 if the A–C gap is within the spread across seeds.
- A seed sets: network init, data order, augmentation and subset draw.
- Whether stage 1 is retrained for each seed is open (§8.5).

## 7. Shared settings

- **ResNet-18, CIFAR-style stem:** 3×3 stride-1 first conv, no max-pool. torchvision's default stem (7×7 stride 2 + max-pool) is built for 224px ImageNet and shrinks 32×32 inputs too much. The same network is used for every arm and stage.
- **Normalization:** one set of values for all stages and datasets (CIFAR-10 mean/std proposed), so the backbone doesn't see a shifted input distribution between stages.
- **Optimizer (all stages, all arms):** SGD, momentum 0.9, weight decay 5e-4, batch size 128.
- **LR schedule:**
  - Stages 1 and 2: cosine decay from base LR 0.1.
  - Stage 3: **linear warmup, then constant LR**, for every arm. The constant LR is needed for the aligned-curve analysis in §5; its value is §8.7. The warmup (a few hundred steps from near 0, length fixed during calibration) is standard, not optional: at the start of stage 3 the head is random, and its noisy gradients flow into the backbone. At full LR they can damage retained features before the head learns to use them. That hurts the arms with the most to lose (A, then C), so it can hide retention, though it can never fake it.
- **Reset at every stage boundary:** a new optimizer (fresh momentum buffers) and a new LR schedule, done the same way for every arm. Nothing optimizer-related carries over between stages.
- **Augmentation:**
  - CIFAR-10 and CIFAR-100: random crop 32 with padding 4, plus horizontal flip.
  - SVHN: random crop 32 with padding 4 only. No flip, since flipped digits aren't valid digits.
  - Evaluation: no augmentation.
- Device: MPS (Apple M4 Max).

## 8. Open decisions

Each has a recommendation. Tick it when decided.

- [ ] **8.1 Match C's pretraining to A's.** Recommended: 40k CIFAR-100 images, with the same number of steps, batch size, LR schedule and augmentation as A. Otherwise C differs from A in data and compute, not just content. Note: this means 400 images per class (vs 4,000 for A), so C's stage-1 accuracy will be lower. That is expected and isn't a confound for the A–C comparison.
- [ ] **8.2 Validation set.** The test set must not be used for tuning or early stopping. Recommended: split the 10k held-out pool into a fixed 5k validation set and a 5k never-seen pool. The catch is that never-seen subsets are then all 5k of that pool (no variation across seeds). The alternative is a separate validation split carved from the 40k pretrain pool.
- [ ] **8.3 Stage-2 length.** Use the same number of SVHN steps for every arm. Choose it so that A's old-head CIFAR-10 accuracy drops close to chance (the operational definition of "overwritten"). Check that SVHN accuracy at the end of stage 2 is comparable across arms.
- [ ] **8.4 Stage-3 budget.** A fixed maximum number of steps for relearning, used as the cutoff for "not reached" in steps to target.
- [ ] **8.5 Retrain stage 1 per seed?** If all seeds share one pretrained checkpoint, the seed spread leaves out pretraining variance and makes the A–C gap look more certain than it is. Retraining per seed is more honest but the most expensive part. Decide once the baseline (§9) gives time per epoch.
- [ ] **8.6 Linear probe protocol.** Decided: trained on the **never-seen** 5k subset, never the A-seen one. A's backbone already trained on the A-seen images, so a probe trained on them would flatter A. Evaluated on the CIFAR-10 test set. Still open: classifier type, number of epochs or iterations, LR or regularization strength, and feature standardization, all fixed and the same for every arm. Recommended implementation: run the frozen backbone once in **eval mode** under `torch.no_grad()` with no augmentation, save the features (5000×512 train, 10000×512 test), and train a linear classifier on them (`nn.Linear(512, 10)` or scikit-learn `LogisticRegression`). Eval mode matters: in train mode BatchNorm updates its running statistics even when gradients are off, which would change the backbone that stage 3 starts from. Sanity check: the backbone's `state_dict` is identical before and after the probe. Note: if §8.2 uses half of the held-out pool for validation, the never-seen pool is a fixed 5k, so the probe and the never-seen relearning subset use the same images in every seed.
- [ ] **8.7 Stage-3 learning rate.** This is a constant LR (§7). Either one value for all arms, or a small sweep per arm chosen on validation. Pretrained and randomly initialized backbones often prefer different LRs, so a single shared LR can favor one arm.

## 9. Next step: baseline

Train ResNet-18 (CIFAR stem) from scratch on the 40k CIFAR-10 pretrain pool on MPS. This gives:

- a check that the pipeline works end to end
- the stage-1 reference accuracy, used to confirm or adjust the steps-to-target thresholds (§5)
- **time per epoch**, used to estimate the total cost: 3 arms × 3 stages × seeds (+ 2 subsets in stage 3). That decides 8.5 and 8.7.
