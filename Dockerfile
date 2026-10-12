# Inference image for Hugging Face Spaces (Docker SDK) or any container host.
#   docker build -t terrareader .
#   docker run -p 7860:7860 terrareader      ->  http://localhost:7860
FROM python:3.11-slim

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch keeps the image small (~700 MB instead of several GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user ssl_pretrain/ ./ssl_pretrain/
COPY --chown=user finetune/ ./finetune/
COPY --chown=user deployment/ ./deployment/
COPY --chown=user checkpoints/classifier_final.pt ./checkpoints/classifier_final.pt

USER user
ENV MODEL_PATH=/app/checkpoints/classifier_final.pt
EXPOSE 7860
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "7860"]
