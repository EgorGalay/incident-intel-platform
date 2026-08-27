FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY pyproject.toml README.md /app/
COPY src /app/src

EXPOSE 8080

CMD ["python", "-m", "mii.server"]
