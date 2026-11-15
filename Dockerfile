FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY core core
COPY capture capture
COPY labeling labeling
COPY augment augment
COPY vision vision
COPY depth depth
COPY portion portion
COPY menu menu
COPY analytics analytics
COPY api api
COPY eval eval
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
