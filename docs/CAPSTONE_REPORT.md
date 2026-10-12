# Self-Supervised Learning for Low-Label Remote Sensing Analysis
**Capstone Project Report** — *[Your Name] · [Programme / Institution] · [Supervisor] · [Month Year]*

> Fill the bracketed items. Numbers marked ≈ were read from the label-efficiency plot; replace them with the exact
> table printed at the end of your Kaggle `evaluate.label_efficiency` run.

## Abstract
Land-cover mapping from satellite imagery depends on labeled data that is slow and costly to produce, while unlabeled
imagery is abundant. This project applies self-supervised learning (SimCLR) to pretrain a ResNet-18 on unlabeled
Sentinel-2 tiles from EuroSAT and fine-tunes it on small labeled subsets. With only 1% of the training labels
(≈190 tiles) the pretrained model reaches ≈83% test accuracy versus ≈58% for a randomly initialised network, a
gain of ≈25 percentage points; the gap closes as labels increase and vanishes near 100%. The final model is deployed as
a containerised FastAPI web application on Hugging Face Spaces.

## 1. Introduction and motivation
Earth-observation satellites produce petabytes of imagery, but supervised models need annotated examples that require
experts or field surveys. Self-supervised learning (SSL) learns visual representations from images alone, reducing the
number of labels needed downstream. This is valuable for agriculture, urban planning and disaster response, where
labels are scarce.

## 2. Objectives
1. Pretrain a deep network on unlabeled satellite imagery with a contrastive SSL method.
2. Quantify how much SSL reduces label requirements (label-efficiency experiment).
3. Deploy the trained model as a usable web service.

## 3. Related work
- **SimCLR** (Chen et al., 2020): contrastive learning with strong augmentations and the NT-Xent loss.
- **EuroSAT** (Helber et al., 2019): 27,000 Sentinel-2 patches, 10 land-use classes.
- **ResNet** (He et al., 2016): the backbone architecture.
- **SatMAE** (Cong et al., 2022): masked-autoencoder pretraining for multispectral/temporal satellite data (future work).

## 4. Dataset
EuroSAT RGB: 27,000 images of 64×64 pixels, 10 classes (AnnualCrop, Forest, HerbaceousVegetation, Highway,
Industrial, Pasture, PermanentCrop, Residential, River, SeaLake). Fixed random split with seed 42:
70% train (18,900), 10% validation (2,700), 20% test (5,400). Label fractions are sampled from the training split only;
validation and test are never reduced.

## 5. Methodology
**5.1 Backbone.** ResNet-18 with the stem changed to a 3×3 stride-1 convolution and no max-pool, because the standard
7×7/stride-2 stem discards too much resolution on 64×64 inputs.

**5.2 Self-supervised pretraining.** Two augmented views per tile (random resized crop, flips, 90° rotations, colour
jitter, grayscale, Gaussian blur). A 2-layer MLP projection head maps features to 128-d; training minimises the NT-Xent
loss (temperature 0.5). AdamW, lr 3e-4, cosine schedule, batch 256, [100] epochs on a Kaggle [T4/P100] GPU. Labels are
never used.

**5.3 Fine-tuning.** The projection head is discarded and a linear classifier is added. AdamW, lr 1e-3, cosine
schedule, [15–30] epochs; the checkpoint with the best validation accuracy is evaluated once on the test set.

**5.4 Baseline.** Identical architecture and training recipe, but random initialisation.

**5.5 Metrics.** Test accuracy and macro-F1.

## 6. Results
| Labeled data | SSL-pretrained | Random init | Gain |
|---|---|---|---|
| 1% | ≈83.2% | ≈57.6% | ≈+25.6 pts |
| 5% | ≈92.0% | ≈77.6% | ≈+14.4 pts |
| 10% | ≈94.6% | ≈85.1% | ≈+9.5 pts |
| 25% | ≈96.2% | ≈92.1% | ≈+4.1 pts |
| 50% | ≈97.1% | ≈94.4% | ≈+2.7 pts |
| 100% | ≈97.4% | ≈97.0% | ≈+0.4 pts |

![Label efficiency](../results/label_efficiency_curve.png)

**Discussion.** The advantage of pretraining is largest at 1% labels and shrinks monotonically as labels grow, the
typical SSL signature: the learned features substitute for labels when few exist. At full supervision the baseline
catches up, since EuroSAT is a comparatively easy, large benchmark. [Add: per-class errors / confusion matrix, e.g.
which classes (Highway vs River, Pasture vs HerbaceousVegetation) are confused.]

## 7. System design and deployment
- **Inference service:** FastAPI (`/predict`, `/health`, `/api/info`, `/api/samples`), PyTorch CPU, input validation
  (10 MB limit, image decoding errors handled).
- **Frontend:** single-page web UI with upload/sample tiles, class probabilities and latency, plus results and
  limitations sections.
- **Packaging:** Docker image (CPU-only PyTorch, non-root user), model weights tracked with Git LFS.
- **Hosting:** Hugging Face Spaces (Docker SDK); source on GitHub. Live demo: `[URL]`. Typical CPU latency: `[x] ms`.

## 8. Limitations
1. Only 64×64 RGB European tiles; no near-infrared/SWIR bands, no other sensors or regions.
2. Random (not geographic) split can leak spatial correlation and inflate accuracy.
3. Patch-level classification only.
4. Single run per setting; no confidence intervals (use ≥3 seeds for rigorous claims).
5. Probabilities are not calibrated.

## 9. Future work
Multispectral (13-band) inputs; masked-autoencoder pretraining (SatMAE) and a ViT backbone; geographic train/test
splits; segmentation and change detection heads; multiple seeds with error bars; model quantisation for edge use.

## 10. Conclusion
Self-supervised pretraining on unlabeled satellite tiles gives large accuracy gains when labels are scarce
(≈+25 points at 1%), demonstrating a practical way to cut annotation cost, and the resulting model is available as a
public web application.

## References
1. T. Chen, S. Kornblith, M. Norouzi, G. Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020.
2. P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
3. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
4. Y. Cong et al. *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* NeurIPS 2022.
