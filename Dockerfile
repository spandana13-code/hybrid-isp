FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY isp ./isp
COPY models ./models
COPY examples ./examples
COPY web ./web
COPY docs ./docs
COPY app.py .
EXPOSE 7860
CMD ["python", "app.py"]
