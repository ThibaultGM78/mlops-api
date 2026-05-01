FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Indispensable pour LightGBM
RUN apt update
RUN apt install libgomp1 -y

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY app.py /app/app.py
COPY src/ /app/src/

RUN uv sync --no-cache-dir

# On ouvre et expose le port 80
EXPOSE $PORT

# Lancement de l'API
# Attention : ne pas lancer en daemon !
# Utilise la variable PORT fournie par Cloud Run, sinon 8080 par défaut
CMD ["sh", "-c", "uv run gunicorn app:app -b 0.0.0.0:${PORT:-8080} -w 4"]