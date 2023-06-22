import os
import itertools
import seaborn as sn
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import efficientnet.keras as efn
import keras.backend as K
import tensorflow_addons as tfa

from keras.models import load_model
from keras import metrics
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, GaussianNoise
from keras.models import Model, Sequential
from keras_vggface.vggface import VGGFace
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

IMAGE_W = 197
IMAGE_H = 197

folder = 'logs/ResNet-50-notebook'

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

epochs_top_layers = 50

# Read csv and split train test
csv_file = "trainFER2013.csv"
testcsv_file = "testFER2013.csv"

train_df = pd.read_csv(csv_file)
test_df = pd.read_csv(testcsv_file)


def scalar(img):
    return img / 127.5 - 1


x_col = 'Filenames' # set this to point to the df column that holds the full path to the image file
y_col = 'Label'  # set this to the df column that holds the class labels
batch_size = 128  # set this to the batch size
classmode = 'categorical'
image_shape = (IMAGE_W, IMAGE_H)  # set this to desired image size

tgen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=scalar, rotation_range = 10,
                                                       shear_range = 10, zoom_range = 0.1,
                                                       fill_mode = 'reflect', horizontal_flip=True)
gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=scalar)

train_gen=tgen.flow_from_dataframe(train_df, x_col=x_col,
                                   y_col=y_col,target_size=image_shape, seed=42,
                                   class_mode=classmode, batch_size=batch_size)
test_gen=gen.flow_from_dataframe(test_df, x_col= x_col, y_col=y_col,
                                   class_mode=classmode, seed=42,
                                   batch_size=batch_size, shuffle=False)

print(len(train_df))
# Create model
base_model = VGGFace(
    model       = 'resnet50',
    include_top = False,
    pooling='avg')

# Freeze some layers
batch_norm_indices = [2, 6, 9, 13, 14, 18, 21, 24, 28, 31, 34, 38, 41, 45, 46, 53, 56, 60, 63, 66, 70, 73, 76, 80,
                      83, 87, 88, 92, 95, 98, 102, 105, 108, 112, 115, 118, 122, 125, 128, 132, 135, 138, 142, 145,
                      149, 150, 154, 157, 160, 164, 167, 170]
for i in range(170):
    if i not in batch_norm_indices:
        base_model.layers[i].trainable = False

model = Sequential()
model.add(base_model)

# Flattens the input. Does not affect the batch size
model.add(Flatten())

# Add 2 fully-connected layers
model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                    bias_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                    bias_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Add a logistic layer
model.add(Dense(7, activation="softmax"))


# Compile model--tf.keras.optimizers.Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0),
lrd = ReduceLROnPlateau(monitor = 'val_accuracy',patience = 2,verbose = 1,factor = 0.20, min_lr = 0.00001)
optm = tf.keras.optimizers.Adam(learning_rate = 0.0001)
# optm = tf.keras.optimizers.SGD(lr = 0.01, momentum=0.9, decay = 0.0001, nesterov=True)
check_point = ModelCheckpoint('NewRN-cp1.h5',
                              save_best_only= True, mode = 'auto')

# Focal loss
# loss_func = tfa.losses.S

model.compile( optimizer = optm, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the model on the new data for a few epochs (Fits the model on data yielded batch-by-batch by a Python generator)
history = model.fit(x = train_gen, epochs = epochs_top_layers, validation_data = test_gen,
                    shuffle=True,
                    callbacks = [check_point])
    # samples_per_epoch / batch_size    steps_per_epoch = len(train_df) // batch_size,

model.save("RNmodel.h5")
hs = history.history

# Evaluate
train_evaluation = model.evaluate(train_gen)
test_evaluation = model.evaluate(test_gen)

# summarize history for accuracy
plot1 = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
# summarize history for loss
plot2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
