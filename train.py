import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import join, isfile
import numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
import glob
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix
import datetime

# red-0-sky, green-1-land, blue-2-sea
CLASS_ENCODING = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order

def preprocess_img(img_pil):
    centered_img = np.array(img_pil, dtype='float16')-DATA_MEAN
    centered_img_bgr = centered_img[:, :, ::-1]
    return centered_img_bgr

def mask2img(mask_np):
    return Image.fromarray((mask_np * 255).astype('uint8'), 'L')

def get_x_y_data(dir_path, img_size=(473,473), n_classes=3, n_samples=-1):
    print('Loading x,y data')
    if n_samples < -1 or n_samples == 0:
        raise ValueError('Cannot create {} training samples'.format(n_samples))

    img_paths = sorted(glob.glob('{}/images/*.png'.format(dir_path)))
    mask_paths = sorted(glob.glob('{}/masks/*.png'.format(dir_path)))
    if not len(img_paths) == len(mask_paths):
        raise ValueError('Size mismatch: img paths and mask paths!')

    shuffled_indices = np.arange(len(img_paths))
    np.random.shuffle(shuffled_indices)
    img_paths = np.array([img_paths[i] for i in shuffled_indices])
    mask_paths = np.array([mask_paths[i] for i in shuffled_indices])

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
    return x, y, img_paths[:n_samples], mask_paths[:n_samples]

def update_training_data():
    train_path = '/home/fredrik/sensorfusion/world_tracker/segmentation/training_data'
    x, y, x_paths, y_paths = get_x_y_data(train_path)
    n_samples = len(x)

    n_train = int(0.8 * n_samples)

    x_train = x[:int(0.8 * n_train)]
    y_train = y[:int(0.8 * n_train)]
    x_train_paths = x_paths[:int(0.8 * n_train)]
    y_train_paths = y_paths[:int(0.8 * n_train)]
    x_val = x[int(0.8 * n_train):n_train]
    y_val = y[int(0.8 * n_train):n_train]
    x_val_paths = x_paths[int(0.8 * n_train):n_train]
    y_val_paths = y_paths[int(0.8 * n_train):n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]
    x_test_paths = x_paths[n_train:]
    y_test_paths = y_paths[n_train:]
    
    np.save('train_data/xy_train.npy', (x_train, y_train))
    np.save('train_data/xy_val.npy', (x_val, y_val))
    np.save('train_data/xy_test.npy', (x_test, y_test))

    for fileName, paths in [('x_train_paths',x_train_paths),('y_train_paths',y_train_paths),
                           ('x_val_paths',x_val_paths),('y_val_paths',y_val_paths),
                           ('x_test_paths',x_test_paths),('y_test_paths',y_test_paths)]:
        with open('train_data/{}.txt'.format(fileName), 'w+') as f:
            f.write('\n'.join(paths))

def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if isinstance(weights,list) or isinstance(np.ndarray):
        weights=K.variable(weights)

    def loss(target,output,from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')
    return loss

def get_compiled_model(model_name, lrn_rate, checkpoint=None):
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
                  loss=weighted_categorical_crossentropy([3.84, 6.25, 1.7]),
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

def train(model, lrn_rate):
    x_train, y_train = np.load('train_data/xy_train.npy')
    x_val, y_val = np.load('train_data/xy_val.npy')
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    t_now = str(datetime.datetime.now()).replace(' ','_').replace(':','-')
    os.mkdir('./checkpoints/{}'.format(t_now))
    checkpoint_path = 'checkpoints/{}/checkpoint-lrn{}'.format(t_now, lrn_rate)+'-epoch{epoch:03d}-val_acc{val_acc:.4f}.h5'
    print(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_acc',
                                          verbose=1,
                                          mode='max',
                                          save_weights_only=True)

    #os.mkdir('./Graph/{}'.format(t_now))
    tensorboard_callback = TensorBoard(log_dir='./Graph/{}'.format(t_now), histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train,
              batch_size=4,
              epochs=100,
              validation_data=(x_val, y_val),
              callbacks=[checkpoint_callback, tensorboard_callback])

def evaluate_test(model):
    print('EVALUATING ON TEST SET')
    x_test, y_test = np.load('train_data/xy_test.npy')

    score, acc = model.evaluate(x=x_test, y=y_test, batch_size=16, verbose=1)
    print('loss: {}'.format(score))
    print('acc: {}'.format(acc))
    y_pred = model.predict(x_test, batch_size=16, verbose=1)
    print('ANALYZING TEST DATA. CAN TAKE SEVERAL MINUTES.')
    cm = confusion_matrix(np.argmax(y_test, axis=3).flatten(), np.argmax(y_pred, axis=3).flatten())
    cm = cm.astype('float64')
    print('Class representation:')
    for class_id in [0,1,2]:
        print(class_id, sum(cm[class_id,:])/float(len(y_pred)*473*473))
    for row in range(cm.shape[0]):
        cm[row,:] = cm[row,:] / np.sum(cm[row,:])
    with np.printoptions(precision=3, suppress=True):
        print('CONFUSION MATRIX:')
        print(cm)

def predict_single_test_image(model, n):
    with open('train_data/x_test_paths.txt', 'r') as f:
        x_paths = [path.strip() for path in f.readlines()]
        print(x_paths)
        predict_img(model, x_paths[n])

def predict_img(model, img_path):
    img_pil = Image.open(img_path)
    x = np.array([preprocess_img(img_pil)])

    prediction = model.predict(x, batch_size=1, verbose=1)[0]
    prediction_img = prediction_to_img(prediction)

    Image.blend(img_pil, prediction_img, 0.25).show()

def prediction_to_img(prediction):
    max_class = np.argmax(prediction,axis=2)
    img_np = np.empty((473,473,3),dtype='uint8')
    for class_id, color in CLASS_ENCODING.items():
        img_np[max_class==class_id] = color

    return Image.fromarray(img_np)



if __name__ == '__main__':
    #update_training_data()
    lrn_rate = 1e-5
    resume_checkpoint_path = 'checkpoints/2019-03-25_20-43-05.539417/checkpoint-lrn5e-05-epoch003-val_acc0.9823.h5'
    model = get_compiled_model('pspnet50_all-train', lrn_rate, checkpoint=resume_checkpoint_path)
    train(model, lrn_rate)

    exit()
    test_checkpoint = 'checkpoints/2019-03-25_20-43-05.539417/checkpoint-lrn5e-05-epoch003-val_acc0.9823.h5'
    model = get_compiled_model('pspnet50_all-train', 0, checkpoint=test_checkpoint)
    for i in range(400,450):
        predict_single_test_image(model, i)
    #evaluate_test(model)


