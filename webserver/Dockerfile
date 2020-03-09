FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt /app/

COPY . /app

RUN ["apt-get", "update"]

RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev", "libgtk2.0-dev"]

RUN pip install -r /app/requirements.txt
