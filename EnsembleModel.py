import tensorflow as tf
import pandas as pd
import efficientnet.keras as efn

from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, GaussianNoise
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


from numpy import argmax

from sklearn.model_selection import train_test_split


IMAGE_W = 197
IMAGE_H = 197

i = 0

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Read csv and split train test
csv_file = "trainFER2013.csv"
testcsv_file = "testFER2013.csv"

train_df = pd.read_csv(csv_file)
test_df = pd.read_csv(testcsv_file)


def scalar(img):
    return img / 127.5 - 1


x_col = 'Filenames' # set this to point to the df column that holds the full path to the image file
y_col = 'Label'  # set this to the df column that holds the class labels
batch_size = 64  # set this to the batch size
classmode = 'categorical'
image_shape = (IMAGE_W, IMAGE_H)  # set this to desired image size

gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=scalar)

test_gen=gen.flow_from_dataframe(test_df, x_col= x_col, y_col=y_col,
                                   class_mode=classmode, seed=42,
                                   batch_size=batch_size, shuffle=False)

X_train, y_train = next(train_gen)
X_test, y_test = next(test_gen)

m1 = load_model('RNmodel.h5')
m1._name = 'test1'

m2 = load_model('EfNetB1.h5')
m2._name = 'test2'

commonInput = tf.keras.Input(shape=(197, 197, 3))
out1 = m1(commonInput)
out2 = m2(commonInput)
merged = tf.keras.layers.Add()([out1, out2])

ensemble_model = Model(commonInput, merged)
ensemble_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ensemble_model.save('Ensemble.h5')
ensemble_model.evaluate(test_gen)