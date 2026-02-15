FROM python:3.11-slim

WORKDIR /app

# ヘルスチェック用のcurlをインストール
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 依存ライブラリのインストール（キャッシュ活用のため先にコピー）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルのコピー
COPY app.py .
COPY companies.csv .
COPY ideal_society.csv .
COPY .streamlit/config.toml .streamlit/config.toml

# 事前計算済み埋め込みベクトル
COPY data/embeddings/ data/embeddings/

# 利用回数ファイル用ディレクトリ
RUN mkdir -p data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
