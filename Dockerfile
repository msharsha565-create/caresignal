# --- build the TypeScript UI ---
FROM node:22-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- API + agent + MCP server ---
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY caresignal/ caresignal/
# Deterministic (seeded): generates the synthetic cohort and trains the model.
RUN python -m caresignal.model \
 && python -m caresignal.model --dataset uci
COPY --from=web /web/dist web/dist
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
# Cloud Run supplies PORT; default to 8000 elsewhere (ECS, local).
CMD ["sh", "-c", "exec uvicorn caresignal.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
