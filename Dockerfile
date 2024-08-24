FROM python:3.12-slim

WORKDIR /app

COPY poetry.lock pyproject.toml README.md ./
COPY src/ src/

RUN pip install poetry

RUN poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root && \
    poetry build

CMD ["poetry", "run", "uvicorn", "--app-dir", "src", "ml_api.main:app", "--host", "0.0.0.0", "--port", "80"]