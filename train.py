import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import splitext, join, isfile
from os import environ
from math import ceil
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import layers_builder as layers
import utils
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.client import device_lib
from keras.optimizers import SGD


def get_compiled_model(model_name, lrn_rate):
    dir_name = 'weights/keras'

    json_path = join("weights", "keras", model_name + ".json")
    h5_path = join("weights", "keras", model_name + ".h5")
    if isfile(json_path) and isfile(h5_path):
        print("Keras model & weights found, loading...")
        with open(json_path, 'r') as file_handle:
            model = model_from_json(file_handle.read())
        model.load_weights(h5_path)
    sgd = SGD(lr=lrn_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_train_test_data():
    pass

def overfit_test():
    model = get_compiled_model('pspnet50_custom', 1e-2)

    x = np.zeros((64,473,473,3))
    y = np.zeros((64, 473, 473, 2))
    y = np.concatenate((y,np.ones((64, 473, 473, 1))), axis=3)
    print(y.shape)

    x_train = x[:50]
    y_train = y[:50]
    x_val = x[50:]
    y_val = y[50:]

    model.fit(x_train, y_train,
                    batch_size=8,
                    epochs=1000,
                    validation_data=(x_val,y_val))


if __name__ == '__main__':
    overfit_test()