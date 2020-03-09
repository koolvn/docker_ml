"""
Web server script that exposes endpoints and pushes images to Redis for classification by model server. Polls
Redis for response from model server.

Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import io
import json
import os
import time
import uuid
import tensorflow as tf
import cv2

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image
import redis

from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request

print('\n!!!!!!!!!!\n', '\n', os.listdir('.'), '\n')

app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))


def get_faces(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img = cv2.imdecode(np.frombuffer(image, np.float), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return 0
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 0)
        # Save the output image
        # cv2.imwrite('detected.jpg', img[y:y + h, x:x + w])
        img = img[y:y + h, x:x + w]

    return cv2.imencode('.jpg', img)[1]


def prepare_image(image, target):
    # If the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # Return the processed image
    return image


@app.get("/")
def index():
    return f"I'm using Tensorflow version: {tf.__version__}"


@app.get("/predict")
def predict_index():
    return f'Please send me your picture via POST request'


@app.post("/predict")
def predict(request: Request, img_file: bytes = File(...)):
    data = {"success": False}

    if request.method == "POST":
        faces = get_faces(img_file)
        if faces == 0:
            return 'No faces detected'

        image = Image.open(io.BytesIO(faces))
        image = prepare_image(image,
                              (int(os.environ.get("IMAGE_WIDTH")),
                               int(os.environ.get("IMAGE_HEIGHT")))
                              )

        # Ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it
        image = image.copy(order="C")

        # Generate an ID for the classification then add the classification ID + image to the queue
        k = str(uuid.uuid4())
        image = base64.b64encode(image).decode("utf-8")
        d = {"id": k, "image": image}
        db.rpush(os.environ.get("IMAGE_QUEUE"), json.dumps(d))

        # Keep looping for CLIENT_MAX_TRIES times
        num_tries = 0
        while num_tries < CLIENT_MAX_TRIES:
            num_tries += 1

            # Attempt to grab the output predictions
            output = db.get(k)

            # Check to see if our model has classified the input image
            if output is not None:
                # Add the output predictions to our data dictionary so we can return it to the client
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)

                # Delete the result from the database and break from the polling loop
                db.delete(k)
                break

            # Sleep for a small amount to give the model a chance to classify the input image
            time.sleep(float(os.environ.get("CLIENT_SLEEP")))

            # Indicate that the request was a success
            data["success"] = True
        else:
            raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
    return data
