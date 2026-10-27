# Pap smear cell classification: an honest evaluation

A methodological audit of the **SIPaKMeD** dataset (4049 cervical cells manually cropped from 966
Pap smear slides, five classes).

The literature routinely reports above 99% accuracy on this dataset. This project does not try to
beat that number. It asks whether the number means what it appears to mean, and what the model is
actually using to decide.

---

## Key findings

| Finding | Evidence |
|---|---|
| Random splitting leaves **91.8%** of test cells with a sibling from the same cluster in training | direct measurement over the folds |
| That split hides roughly **1 in 4 errors** and halves the apparent variance | 0.9796 vs 0.9731 macro F1, McNemar p = 0.012 |
| Adding 26 morphological descriptors from expert contours **hurts** the model | 0.9671 vs 0.9731, McNemar p = 0.0075 |
| The network encodes nuclear **color and texture**, but not nuclear **geometry** | linear probing: R² 0.64 to 0.83 vs R² 0.045 to 0.19 |
| Fine-tuning **degrades** the morphological representation relative to the generic network | the ImageNet backbone wins on all 12 probed properties |
| The network **does attend to the nucleus**: 2.4x more attention than its area warrants | area-normalized Grad-CAM, n = 4049 |
| Under stain shift, accuracy drops 20 points while nuclear attention **increases** | 2.39 → 2.71 |

One sentence summary: **the network looks at the nucleus, but reads its color rather than its
shape.** That explains why shape descriptors add nothing, and why a stain shift breaks it while it
keeps looking in the right place.

---

## Repository layout

```
notebooks/
  cervical_cancer_completo_local.ipynb    full pipeline, executed (outputs included)
  cervical_cancer_completo_colab.ipynb    same pipeline adapted for Google Colab
  por_fases/                              individual parts, for reading or rerunning in isolation
docs/
  explicacion_fases_3_y_4.md              three-level walkthrough with annotated code
  fases_3_y_4.html                        same walkthrough as a navigable web page
  instalacion_local.md                    step-by-step local setup on Linux
  plan-proyecto-cancer-cervical.pdf       original project plan
  exposicion/                             presentation scripts
resultados/                               notebook outputs (git-ignored)
```

Notebooks and documentation are written in Spanish. The full notebook is the entry point; the
per-part files hold the same content, split up.

---

## The five parts

**1. What is inside the dataset.** The 4049 cells come from 966 slides, averaging 4.2 cells per
slide with one extreme case of 26. The filename prefix identifies the source slide, and everything
else rests on that. Annotation is complete, no contours are missing, and only one physically
impossible annotation was found.

**2. How much random splitting inflates results.** The same model trained twice, changing only how
images are partitioned. The macro F1 gap looks small, but counted as errors nearly a quarter of them
disappear, and fold-to-fold variance drops by more than half: the published figure is not only high,
it also looks more precise than it is.

**3. Do expert measurements help?** 26 morphological descriptors computed from the annotated
contours, compared in a single-variable ablation (same network, same head, same folds). They do not
help, they hurt, and detectably so. A model using only those descriptors, never seeing an image,
reaches 0.85, so the problem is not their quality.

**4. So why don't they help?** Linear probing on frozen embeddings to measure what the network
encodes, with an unfine-tuned control. What the network lacks (nuclear geometry) turns out not to be
what it needs. This also killed our own hypothesis that resizing destroyed the original cell scale:
the network recovers it anyway, R² 0.79.

**5. Where it looks and how far it generalizes.** An area-normalized Grad-CAM attention metric
measured across all 4049 cells rather than illustrated with four cherry-picked examples, plus
test-time stain, blur and compression perturbations.

---

## Methodological commitments

Three things hold the results together and should not be broken when reusing the code:

- **Slide-level grouping in every split.** This is the core of the work. Any comparison that ignores
  it produces inflated numbers.
- **All scaling statistics are fit on training folds only**, including the median used to impute the
  handful of missing values. Computing them over the full dataset is a subtle leak, the same kind as
  the splitting one.
- **Ablations change exactly one variable.** The network with and without descriptors is the same
  class, the same head and the same training loop; only the input dimension differs.

In part 4, each cell's embedding comes from the model that did not see it during training. Using a
model that already knows the cell would measure memorization rather than representation.

---

## Running it

**On Colab:** open `notebooks/cervical_cancer_completo_colab.ipynb`, enable GPU and run top to
bottom. It needs a Kaggle token stored in the notebook's Secrets.

**Locally:** follow `docs/instalacion_local.md`. Short version:

```bash
python3 -m venv ~/venvs/sipakmed
source ~/venvs/sipakmed/bin/activate
pip install -r requirements.txt
jupyter lab
```

Careful with PyTorch: a plain PyPI install may pull the CPU-only build. The requirements file notes
this.

The dataset downloads itself through `kagglehub` (about 6.4 GB compressed, 9.2 GB extracted) and
needs a Kaggle token at `~/.kaggle/access_token`.

Heavy stages cache their results under `resultados/` and reuse them if present. Set
`USAR_CACHE = False` in the configuration cell to rerun everything. The full pipeline is around 35
training runs, roughly 2 to 3 hours on an RTX 4050.

---

## Limitations

- **Patient identity is unknown.** The 966 slides may come from considerably fewer patients, so
  residual contamination cannot be ruled out with the published data. The slide is the finest
  available grouping.
- **A single seed.** Effect directions were stable across machines, but the magnitude of the leak
  varied (from a quarter to 40% of errors) between runs. Averaging several seeds would give a firmer
  estimate.
- **A single architecture.** The splitting effect should not depend on the model, but that is not
  verified here.
- **Linear probing has a known ceiling.** A low R² shows the property is not linearly recoverable
  from the post-global-pooling vector, not that the network lacks it. Geometry is spatial and does
  not survive averaging; probing the pre-pooling feature map would settle this.
- **Perturbations are simulated.** They approximate another lab's conditions rather than replacing
  them.

---

## Stack

PyTorch and torchvision (ImageNet-pretrained EfficientNet-B0), scikit-learn, scikit-image, OpenCV,
statsmodels, matplotlib.

Metrics: macro F1, Matthews correlation coefficient, accuracy, confusion matrices, McNemar's test
for paired comparisons and Mann-Whitney for attention contrasts.

---

## Dataset

Plissiti, M. E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., Charchanti, A. (2018).
*SIPAKMED: A new dataset for feature and image based classification of normal and pathological
cervical cells in Pap smear images.* IEEE International Conference on Image Processing (ICIP),
Athens, Greece.

Copy used: [Cervical Cancer largest dataset (SipakMed)](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
on Kaggle. Original: https://www.cs.uoi.gr/~marina/sipakmed.html

The dataset is not included in this repository.

---

## License

Code and documentation under the MIT License (see `LICENSE`). The dataset keeps its own terms.

## Author

Manuel M. Muñoz-Cárdenas
