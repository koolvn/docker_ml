FROM python:3.7-slim-buster

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt --no-cache-dir

#RUN python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
# Download ResNet50 model and cache in image
#RUN python -c "from keras.applications import ResNet50; ResNet50(weights='imagenet')"

COPY . /app

#SHELL ["systemctl", "enable", "jupyter.service"]
#SHELL ["systemctl", "daemon-reload"]
#SHELL ["systemctl", "restart", "jupyter.service"]
#SHELL ["systemctl", "-a", "|", "grep jupyter"]

CMD ["python", "/app/main.py"]
