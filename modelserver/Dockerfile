FROM python:3.7-slim-buster

COPY requirements.txt /app/

RUN pip install -r --no-cache-dir /app/requirements.txt
RUN nvidia-smi
RUN python -c "from tensorflow.test import is_gpu_available; print(is_gpu_available())"
# Download ResNet50 model and cache in image
#RUN python -c "from keras.applications import ResNet50; ResNet50(weights='imagenet')"

COPY . /app

RUN systemctl enable jupyter.service
RUN systemctl daemon-reload
RUN systemctl restart jupyter.service
RUN systemctl -a | grep jupyter

CMD ["python", "/app/main.py"]
