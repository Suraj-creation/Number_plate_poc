FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock pyproject.toml ./
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.3.1 torchvision==0.18.1 \
    && python -m pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ onnxruntime-gpu==1.20.1 \
    && python -m pip install -r requirements.lock
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-deps . && python -m pip check \
    && groupadd --gid 10001 anpr && useradd --uid 10001 --gid anpr --create-home anpr \
    && mkdir models output && chown anpr:anpr models output
USER 10001:10001
EXPOSE 7860
CMD ["python", "-m", "src.cli", "ui"]
