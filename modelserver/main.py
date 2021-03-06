"""
Model server script that polls Redis for images to classify
Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import json
import os
import sys
import time

import gdown
# from keras.applications import ResNet50
# from keras.applications import imagenet_utils
import numpy as np
import redis
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

D_TYPE = np.float32
# Connect to Redis server
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))


# Load the pre-trained Keras model (here we are using a model
# pre-trained on ImageNet and provided by Keras, but you can
# substitute in your own networks just as easily)


def base64_decode_image(a, dtype, shape):
    # If this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    a = a.reshape(shape)

    # Return the decoded image
    return a


def classify_process():
    url = 'https://drive.google.com/uc?id=1-0G7gqL_0R346aXmIXCpaE0-CKUQ6W5S'
    gdown.download(url, 'best_model.h5', quiet=False)
    model = load_model("./best_model.h5")
    optimizer = Adam(learning_rate=1e-4)  # LR = 0.001
    model.compile(optimizer=optimizer,
                  loss={'gender': 'binary_crossentropy', 'race': 'sparse_categorical_crossentropy', 'age': 'mse'},
                  metrics={'gender': 'accuracy', 'race': 'accuracy', 'age': 'mse'})
    # Continually poll for new images to classify
    while True:
        # Pop off multiple images from Redis queue atomically
        with db.pipeline() as pipe:
            pipe.lrange(os.environ.get("IMAGE_QUEUE"), 0, int(os.environ.get("BATCH_SIZE")) - 1)
            pipe.ltrim(os.environ.get("IMAGE_QUEUE"), int(os.environ.get("BATCH_SIZE")), -1)
            queue, _ = pipe.execute()

        imageIDs = []
        batch = None
        for q in queue:
            # Deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))

            image = base64_decode_image(q["image"],
                                        D_TYPE,
                                        (1, int(os.environ.get("IMAGE_HEIGHT")),
                                         int(os.environ.get("IMAGE_WIDTH")),
                                         int(os.environ.get("IMAGE_CHANS")))
                                        )

            # Check to see if the batch list is None
            if batch is None:
                batch = image

            # Otherwise, stack the data
            else:
                #                 batch = np.vstack([batch, image])
                batch = image
            # Update the list of image IDs
            imageIDs.append(q["id"])

        # Check to see if we need to process the batch
        if len(imageIDs) > 0:
            gender_mapping = {0: 'Male', 1: 'Female'}
            race_mapping = dict(list(enumerate(['White', 'Black', 'Asian', 'Indian', 'Others'])))
            max_age = 116
            # Classify the batch
            print("* Batch size: {}".format(batch.shape))
            predicted_labels = model.predict(batch)
            gender, race, age = int(predicted_labels[0][0] > 0.5), np.argmax(predicted_labels[1][0]), predicted_labels[2][0]
            output = {"gender": f"{gender_mapping[gender]}",
                      "race": f"{race_mapping[race]}",
                      "age": f"{int(age[0] * max_age)}"
                      }
            # Store the output predictions in the database, using image ID as the key so we can fetch the results
            db.set(imageIDs[0], json.dumps(output))

        # Sleep for a small amount
        time.sleep(float(os.environ.get("SERVER_SLEEP")))


if __name__ == "__main__":
    classify_process()
