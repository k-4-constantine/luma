FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml
COPY app ./app
RUN mkdir -p /app_data /chroma_data
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]