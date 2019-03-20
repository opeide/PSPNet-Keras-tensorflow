import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import join, isfile
import numpy as np
#from keras.models import model_from_json
#from keras.optimizers import SGD
import glob
from PIL import Image



def get_x_y_data(dir_path, img_size=(473,473), n_classes=3, n_samples=-1):
    #red-0-sky, green-1-land, blue-2-sea
    mask_encodings = {(255,0,0):0, (0,255,0):1, (0,0,255):2}


    img_paths = sorted(glob.glob('{}/images/*.png'.format(dir_path)))
    if n_samples > 0:
        n_samples = min(len(img_paths), n_samples)
    else:
        n_samples = len(img_paths)

    x = np.empty((n_samples,img_size[0], img_size[1],3))
    for img_path in img_paths[:n_samples]:
        img = Image.open(img_path)
        img.show()


    mask_paths = sorted(glob.glob('{}/masks/*.png'.format(dir_path)))

    y = np.empty((n_samples, img_size[0], img_size[1], n_classes))
    for mask_path in mask_paths[:n_samples]:
        mask_img = Image.open(mask_path)
        mask_img.show()
        mask_np = np.array(mask_img)
        for color in mask_encodings:
            class_id = mask_encodings[color]
            print('class_id: {}'.format(class_id))
            class_mask = np.all(mask_np == color, axis=2)
            print(class_mask.shape)
            Image.fromarray(class_mask.astype(dtype=np.uint8)*255, 'L').show()


if __name__ == '__main__':
    #model = get_compiled_model('pspnet50_custom', 1e-2)
    train_path = '/home/fredrik/sensorfusion/world_tracker/segmentation/training_data'
    get_x_y_data(train_path, n_samples=1)
    exit()

    n_img = len(x)
    x_train = x[:int(0.8*n_img)]
    y_train = y[:int(0.8 * n_img)]
    x_val = x[int(0.8*n_img):]
    y_val = y[int(0.8 * n_img):]


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