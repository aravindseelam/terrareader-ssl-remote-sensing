---
title: TerraReader SSL Land-Cover Classifier
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# TerraReader — Self-Supervised Learning for Low-Label Remote Sensing

A deep learning capstone project: a ResNet-18 is pretrained **without labels** on Sentinel-2 satellite
tiles using **SimCLR**, then fine-tuned with only a small fraction of labels to classify land cover.
Served as a web app (FastAPI + PyTorch) in a Docker container.

**Live demo:** [here](https://ssl-low-label-remote-sensing-analysis.onrender.com)  
 **Code:** [here](https://github.com/aravindseelam/terrareader-ssl-remote-sensing/blob/main/deployment/app.py)

## Key result

![Label efficiency curve](results/label_efficiency_curve.png)

| Labeled data | SSL-pretrained | Random init | Gain |
|---|---|---|---|
| 1% (~190 tiles) | ~83% | ~58% | +25 pts |
| 5% | ~92% | ~78% | +14 pts |
| 10% | ~95% | ~85% | +10 pts |
| 100% | ~97% | ~97% | ~0 |

*(approximate, read from the plot; exact values are in your Kaggle run log)*
Self-supervised pretraining matters most when labels are scarce and converges to the baseline when labels are plentiful.

## Pipeline

```
Unlabeled EuroSAT tiles ──► SimCLR pretraining (ResNet-18, NT-Xent) ──► ssl_backbone_final.pt
                                                                  │
Small labeled subset (1–100%) ──────────────► Fine-tune + linear head ──► classifier_final.pt
                                                                  │
                                           FastAPI + Docker ──► Hugging Face Space (live web app)
```

## Repository layout

```
ssl_pretrain/    SimCLR model, augmentations, dataset, training loop
finetune/        classifier head, fine-tune / linear-probe script
evaluate/        label-efficiency experiment (produces the curve above)
deployment/      FastAPI server + web UI (static/)
checkpoints/     trained weights (tracked with Git LFS)
results/         figures
docs/            DEPLOYMENT_GUIDE.md, CAPSTONE_REPORT.md
scripts/         export_samples.py (demo tiles)
data/            optional Sentinel-2 downloader (Google Earth Engine)
```

## Reproduce the training (Kaggle GPU)

```bash
python -m ssl_pretrain.train_ssl --epochs 100 --batch-size 256 --num-workers 2
python -m finetune.train_finetune --mode finetune --label-fraction 0.10 \
    --ssl-checkpoint ./checkpoints/ssl_backbone_final.pt --output ./checkpoints/classifier_final.pt
python -m evaluate.label_efficiency --ssl-checkpoint ./checkpoints/ssl_backbone_final.pt
```

## Run the app locally

```bash
pip install torch torchvision -r requirements-deploy.txt
python scripts/export_samples.py          # optional: one-click demo tiles
uvicorn deployment.app:app --port 8000    # http://localhost:8000
```

or with Docker: `docker build -t terrareader . && docker run -p 7860:7860 terrareader`

Full deployment steps: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).

## References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), ICML 2020.
- Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*, 2019.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
