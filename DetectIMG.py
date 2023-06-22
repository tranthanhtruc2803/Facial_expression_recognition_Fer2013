# Detect module for images
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import efficientnet.tfkeras

# dictionary which assigns each label an emotion
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


#model = tf.keras.models.load_model('my_model_vgg16_1.h5')
model = tf.keras.models.load_model('RNmodel.h5')

IMAGE_W = 197
IMAGE_H = 197

img = tf.keras.utils.load_img("fear.jpg", target_size=(IMAGE_W, IMAGE_H, 3))

img = tf.keras.utils.img_to_array(img)
img = img / 255

tensor = np.expand_dims(img, axis=0)

prediction = model.predict(tensor)

maxindex = int(np.argmax(prediction))

print(emotion_dict[maxindex])

#plt.imshow(img)
#plt.show()

