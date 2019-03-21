import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import join, isfile
import numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
import glob
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard

# red-0-sky, green-1-land, blue-2-sea
CLASS_ENCODING = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order

def preprocess_img(img_pil):
    centered_img = np.array(img_pil, dtype='float16')-DATA_MEAN
    centered_img_bgr = centered_img[:, :, ::-1]
    return centered_img_bgr

def get_x_y_data(dir_path, img_size=(473,473), n_classes=3, n_samples=-1):
    print('Loading x,y data')
    if n_samples < -1 or n_samples == 0:
        raise ValueError('Cannot create {} training samples'.format(n_samples))

    img_paths = sorted(glob.glob('{}/images/*.png'.format(dir_path)))
    mask_paths = sorted(glob.glob('{}/masks/*.png'.format(dir_path)))
    if not len(img_paths) == len(mask_paths):
        raise ValueError('Size mismatch: img paths and mask paths!')

    if n_samples == -1:
        n_samples = len(img_paths)
    else:
        n_samples = min(len(img_paths), n_samples)

    x = np.zeros((n_samples,img_size[0], img_size[1],3), dtype='float16')
    y = np.zeros((n_samples, img_size[0], img_size[1], n_classes), dtype='float16')

    print('loading imgs: ',end='')
    for n,img_path in enumerate(img_paths[:n_samples]):
        if n%(n_samples/10)==0:
            print('{}%->'.format(int(100*n/n_samples)), end='', flush=True)
        img = Image.open(img_path)
        x[n,:,:,:] = preprocess_img(img)
    print('100%')

    print('loading masks: ', end='')
    for n, mask_path in enumerate(mask_paths[:n_samples]):
        if n%(n_samples/10)==0:
            print('{}%->'.format(int(100*n/n_samples)), end='', flush=True)
        mask_img = Image.open(mask_path)
        mask_np = np.array(mask_img)
        for class_id, color in CLASS_ENCODING.items():
            class_mask = np.all(mask_np == color, axis=2).astype('float16')
            y[n,:,:,class_id] = class_mask
            #Image.fromarray(class_mask.astype(dtype=np.uint8)*255, 'L')
    print('100%')
    return x,y


def get_compiled_model(model_name, lrn_rate, checkpoint=None):
    dir_name = 'weights/keras'

    json_path = join("weights", "keras", model_name + ".json")
    h5_path = join("weights", "keras", model_name + ".h5")
    if isfile(json_path) and isfile(h5_path):
        print("Keras model & weights found, loading...")
        with open(json_path, 'r') as file_handle:
            model = model_from_json(file_handle.read())
        if checkpoint is not None and isfile(checkpoint):
            print('LOADED CHECKPOINT')
            model.load_weights(checkpoint)
        else:
            print('LOADED START WEIGHTS')
            model.load_weights(h5_path)
    sgd = SGD(lr=lrn_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def overfit_test():
    model = get_compiled_model('pspnet50_custom', 1e-2)

    x = np.zeros((64, 473, 473, 3))
    y = np.zeros((64, 473, 473, 2))
    y = np.concatenate((y, np.ones((64, 473, 473, 1))), axis=3)
    print(y.shape)

    x_train = x[:50]
    y_train = y[:50]
    x_val = x[50:]
    y_val = y[50:]

    model.fit(x_train, y_train,
              batch_size=8,
              epochs=1000,
              validation_data=(x_val, y_val))

def train(resume_checkpoint=None):
    # model = get_compiled_model('pspnet50_custom', 1e-2)
    train_path = '/home/fredrik/sensorfusion/world_tracker/segmentation/training_data'
    n_samples = 4400
    x, y = get_x_y_data(train_path, n_samples=n_samples)
    n_samples = len(x)

    n_train = int(0.8 * n_samples)

    x_train = x[:int(0.8 * n_train)]
    y_train = y[:int(0.8 * n_train)]
    x_val = x[int(0.8 * n_train):n_train]
    y_val = y[int(0.8 * n_train):n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    model = get_compiled_model('pspnet50_custom', .5e-3, checkpoint=resume_checkpoint)

    checkpoint_path = 'checkpoints/checkpoint-best.h5'
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='max')

    #tensorboard_callback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train,
              batch_size=16,
              epochs=200,
              validation_data=(x_val, y_val),
              callbacks=[checkpoint_callback])

def mask2img(mask_np):
    return Image.fromarray((mask_np * 255).astype('uint8'), 'L')

def predict(checkpoint_path):
    x, y = get_x_y_data('/home/fredrik/sensorfusion/world_tracker/segmentation/training_data', n_samples=1)

    img_np = x[0,:,:,::-1] + DATA_MEAN
    Image.fromarray(img_np.astype('uint8')).show()

    model = get_compiled_model('pspnet50_custom', .5e-3, checkpoint=checkpoint_path)

    output = model.predict(x, batch_size=16, verbose=1)

    for i in [0,1,2]:
        out_img = mask2img(output[0,:,:,i])
        out_img.show()

if __name__ == '__main__':
    train()
    #predict('checkpoints/checkpoint-epoch28-acc0.9607.h5')