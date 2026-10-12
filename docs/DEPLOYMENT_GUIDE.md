# Deployment Guide (step by step)

You have: `classifier_final.pt`, `ssl_backbone_final.pt`, `label_efficiency_curve.png` (already placed in `results/`
and `deployment/static/assets/`).

## Step 1 — Put the files in place
```
checkpoints/classifier_final.pt     <- from Kaggle (required)
checkpoints/ssl_backbone_final.pt   <- from Kaggle (for reproducibility)
```

## Step 2 — Create an environment and install the inference dependencies
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install torch torchvision -r requirements-deploy.txt
```

## Step 3 — (Optional but recommended) add one-click sample tiles
This downloads EuroSAT (~90 MB) once and copies one tile per class into `deployment/static/samples/`.
Install the extras it needs first if you skipped the training requirements: `pip install torchvision`.
```bash
python scripts/export_samples.py
```

## Step 4 — Test locally
```bash
uvicorn deployment.app:app --port 8000
```
- Open http://localhost:8000 → click a sample → **Classify tile**.
- Check http://localhost:8000/health → `"model_loaded": true`.
- API test: `curl -X POST -F "file=@deployment/static/samples/Forest.jpg" http://localhost:8000/predict`

## Step 5 — Test the Docker image (optional; needs Docker Desktop)
```bash
docker build -t terrareader .
docker run -p 7860:7860 terrareader        # http://localhost:7860
```

## Step 6 — Install Git LFS (model files are > 10 MB)
Hugging Face rejects files over 10 MB unless they are stored with Git LFS. GitHub also recommends it above ~50 MB.
```bash
git lfs install                  # once per computer (download from https://git-lfs.com)
```
`.gitattributes` in this project already tracks `*.pt` and `*.png` with LFS.

## Step 7 — Push the code to GitHub (your portfolio copy)
1. Create an empty repo on github.com (no README), e.g. `terrareader-ssl-remote-sensing`.
2. ```bash
   git init -b main
   git add .
   git commit -m "SSL remote sensing capstone: training, evaluation, deployment"
   git remote add origin https://github.com/<you>/terrareader-ssl-remote-sensing.git
   git push -u origin main
   ```

## Step 8 — Create the Hugging Face Space and push the same folder
1. Sign up at huggingface.co → **Settings → Access Tokens → New token** with **Write** permission. Copy it.
2. **New Space** → name `terrareader` → SDK **Docker** → Blank → **Public** → Create.
3. From your project folder:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-hf-username>/terrareader
   git push space main --force
   ```
   When asked for a password, paste the **access token** (username = your HF username).
   `--force` is needed once because a new Space starts with its own initial commit.
4. Open the Space → **Logs** tab. The build takes ~5–10 min. When it shows *Running*, your app is live at
   `https://<your-hf-username>-terrareader.hf.space`.

The README.md front-matter (`sdk: docker`, `app_port: 7860`) is what tells Hugging Face how to run the app.

## Step 9 — Verify the live deployment
```bash
curl https://<your-hf-username>-terrareader.hf.space/health
```
Then open the URL, run a sample tile, and test an upload from your phone.

## Step 10 — Finish the capstone package
- Paste the live URL and GitHub URL into the top of `README.md`, commit, and push to both remotes
  (`git push origin main && git push space main`).
- Replace the approximate numbers in `docs/CAPSTONE_REPORT.md` with the exact table printed at the end of your Kaggle
  `evaluate.label_efficiency` run.
- Take screenshots of the live demo for your slides; record a 2–3 minute screen demo.
- Optional: upload `ssl_backbone_final.pt` to a GitHub Release or a Hugging Face model repo so others can reuse it.

## Updating the app later
Change files → `git add . && git commit -m "..." && git push origin main && git push space main`. The Space rebuilds automatically.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Push rejected: file larger than 10 MiB | Run `git lfs install`, then `git lfs migrate import --include="*.pt,*.png" --everything`, push again |
| Space stuck on "Building" / build error | Open the Logs tab; most often `checkpoints/classifier_final.pt` is missing from the repo (the Dockerfile copies it) |
| App loads but `/predict` returns 503 | Checkpoint not found; check the filename and that the file is in `checkpoints/` |
| "Application startup failed" with `torch.load` error | Train and run with the same torch major version, or pass `weights_only=False` in `deployment/app.py` if you added custom objects |
| Space sleeps after inactivity (free tier) | Normal; first request wakes it in ~30–60 s |
| Authentication failed on push | Use the access token (Write) as the password, not your account password |
