FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE.md ./
COPY kapcherka ./kapcherka
RUN pip install --no-cache-dir . && useradd -r -u 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["kapcherka", "serve", "--host", "0.0.0.0", "--port", "8000"]
