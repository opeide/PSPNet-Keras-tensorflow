from keras import backend as K
from keras.models import model_from_json, Sequential, Model
import tensorflow as tf
from os.path import splitext, join, isfile
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers import Concatenate, Add
from keras.optimizers import SGD
import layers_builder as layers
import h5py

if __name__ == '__main__':
    source_model = 'pspnet50_ade20k'
    input_shape = (473,473)
    num_classes_new = 3
    model_name_new = 'pspnet50_custom'
    dir_name = 'weights/keras'


    model = layers.build_pspnet(nb_classes=num_classes_new,
                                    resnet_layers=50,
                                    input_shape=input_shape)


    #LOAD OLD WEIGHTS
    #Ignore final layer with weights
    model.layers[-3].name += '_custom'
    h5_path = '{}/{}.h5'.format(dir_name, source_model)
    model.load_weights(h5_path, by_name=True)

    #only set final layer trainable
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    model.layers[-3].trainable = True
    #compile again since changed trainable
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    #SAVE MODEL
    new_model_json = model.to_json()
    with open('{}/{}.json'.format(dir_name, model_name_new), 'w+') as f:
        f.write(new_model_json)
    model.save_weights('{}/{}.h5'.format(dir_name, model_name_new))
