FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app


# ---- Install CPU-only PyTorch BEFORE other deps ----
RUN pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# ---- Now install the rest ----
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]