FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt --no-cache-dir

COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
 libsm6 \
 libxext6 \
 libxrender-dev \
 libgtk2.0-dev
