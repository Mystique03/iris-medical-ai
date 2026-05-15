FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    gradio \
    huggingface_hub \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY app.py .

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["python", "app.py"]
