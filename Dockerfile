# Reproducible CPU-only runtime for simulation, evaluation and CI.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS sim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libegl1 libosmesa6 libglib2.0-0 libx11-6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 画面（desktop の箱）に窓を出すための OpenGL（mesa）。--viewer のときだけ使う。rerun の利用状況の送信は切る
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-dri libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/* \
    && rerun analytics disable

# headless rendering without a GPU（--viewer のときは common.py が glx に切り替える）
ENV PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1

# ---- desktop：仮想の画面。ブラウザ（noVNC）で http://localhost:6080/vnc.html を開いて見る ----
# Xvfb（仮想の画面 :1）を、ほかの箱からも描けるように開く（-listen tcp -ac）。compose の中の網からだけ届く
FROM sim AS desktop
RUN apt-get update && apt-get install -y --no-install-recommends xvfb x11vnc novnc websockify fluxbox \
    && rm -rf /var/lib/apt/lists/*
RUN printf '%s\n' '#!/bin/sh' \
      'Xvfb :1 -screen 0 1600x900x24 -listen tcp -ac >/tmp/xvfb.log 2>&1 &' \
      'sleep 1' \
      'DISPLAY=:1 fluxbox >/tmp/fluxbox.log 2>&1 &' \
      'x11vnc -display :1 -nopw -forever -shared -quiet -rfbport 5900 >/tmp/x11vnc.log 2>&1 &' \
      'echo "desktop: ブラウザで http://localhost:6080/vnc.html を開き「接続」を押す"' \
      'exec websockify --web /usr/share/novnc 6080 localhost:5900' > /usr/local/bin/desktop && chmod +x /usr/local/bin/desktop
CMD ["desktop"]
