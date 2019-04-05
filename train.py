import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import join, isfile
import numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
import glob
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix
import datetime
from plot_history import plot_training_log

# red-0-sky, green-1-land, blue-2-sea
CLASS_ENCODING = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
#DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order

def preprocess_img(img_pil):
    return np.array(img_pil, dtype='float16')
    #centered_img = np.array(img_pil, dtype='float16')-DATA_MEAN
    #centered_img_bgr = centered_img[:, :, ::-1]
    #return centered_img_bgr

def mask2img(mask_np):
    return Image.fromarray((mask_np * 255).astype('uint8'), 'L')

def parse_img_mask_data(dir_path, img_size=(473,473), n_classes=3, n_samples=-1):
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

    print('Parsing {} imgs: '.format(n_samples),end='')
    for n,img_path in enumerate(img_paths[:n_samples]):
        if n%int(n_samples/10)==0:
            print('{}%->'.format(int(100*n/n_samples)), end='', flush=True)
        img = Image.open(img_path)
        x[n,:,:,:] =  (img)
    print('100%')

    print('Parsing {} masks: '.format(n_samples), end='')
    for n, mask_path in enumerate(mask_paths[:n_samples]):
        if n%int(n_samples/10)==0:
            print('{}%->'.format(int(100*n/n_samples)), end='', flush=True)
        mask_img = Image.open(mask_path)
        mask_np = np.array(mask_img)
        for class_id, color in CLASS_ENCODING.items():
            class_mask = np.all(mask_np == color, axis=2).astype('float16')
            y[n,:,:,class_id] = class_mask
            #Image.fromarray(class_mask.astype(dtype=np.uint8)*255, 'L').save('mask_{}.png'.format(class_id))
    print('100%')
    return x, y, img_paths[:n_samples], mask_paths[:n_samples]

#Parse and store (cache) training data
def update_training_data(train_data_dir, n_samples=-1):
    for partition in ['train', 'val', 'test']:
        partition_data_dir = '{}/{}'.format(train_data_dir, partition)
        print('Parsing {} data from {}'.format(partition.upper(), partition_data_dir))
        x, y, x_paths, y_paths = parse_img_mask_data(partition_data_dir, n_samples=n_samples)
        print('saving numpy XY data')
        np.save('{}/xy_data_{}.npy'.format(train_data_dir, partition), (x, y))
        print('saving used img paths')
        with open('{}/paths_x_data_{}.txt'.format(train_data_dir, partition), 'w+') as f:
            f.write('\n'.join(x_paths))
        print('saving used mask paths')
        with open('{}/paths_y_data_{}.txt'.format(train_data_dir, partition), 'w+') as f:
            f.write('\n'.join(y_paths))


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
            print('LOADING CHECKPOINT')
            model.load_weights(checkpoint)
        else:
            print('LOADING START WEIGHTS')
            model.load_weights(h5_path)
    sgd = SGD(lr=lrn_rate, momentum=0.9, nesterov=True)
    print('COMPILING MODEL')
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

class TrainingLogger(Callback):
    def __init__(self, saveDir):
        self.logPath = '{}/training.log'.format(saveDir)
        with open(self.logPath, 'w+') as f:
            f.write('epoch,acc,loss,val_acc,val_loss,lrn_rate\n')
    def on_epoch_end(self, epoch, logs=None):
        print('end')
        print(logs)
        with open(self.logPath, 'a+') as f:
            f.write('{epoch},{acc},{loss},{val_acc},{val_loss},{lrn_rate}\n'.format(epoch=epoch,
                                                                                    acc=logs['acc'],
                                                                                    loss=logs['loss'],
                                                                                    val_acc=logs['val_acc'],
                                                                                    val_loss=logs['val_loss'],
                                                                                    lrn_rate=logs['lr']))
        plot_training_log(self.logPath)

def get_step_decay_schedule(lrn_rate, decay_factor, step_duration):
    schedule = lambda epoch: lrn_rate*decay_factor**(epoch//step_duration)
    return schedule

def train(model, lrn_rate):
    print('LOADING TRAINING DATA')
    x_train, y_train = np.load('/media/fredrik/WDusbdrive/segmentation_training_data/xy_data_train.npy')
    x_val, y_val = np.load('/media/fredrik/WDusbdrive/segmentation_training_data/xy_data_val.npy')
    print('trainX: {}, trainY: {}'.format(x_train.shape, y_train.shape))
    print('valX: {}, valY: {}'.format(x_val.shape, y_val.shape))
    print('LOADED TRAINING DATA')

    t_now = str(datetime.datetime.now()).replace(' ','_').replace(':','-')
    keras_logdir = '/media/fredrik/WDusbdrive/keras_logdir/{}'.format(t_now)
    os.mkdir(keras_logdir)

    checkpoint_path = keras_logdir+'/checkpoint-epoch{epoch:03d}.h5'
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_acc',
                                          verbose=1,
                                          mode='max',
                                          save_weights_only=True)

    tensorboard_logdir = '/media/fredrik/WDusbdrive/tensorboard_logdir/{}'.format(t_now)
    tensorboard_callback = TensorBoard(log_dir=tensorboard_logdir, histogram_freq=0, write_graph=True, write_images=True)

    lrn_rate_decay = 0.5
    lrn_rate_duration = 30
    lr_schedule = get_step_decay_schedule(lrn_rate, lrn_rate_decay, lrn_rate_duration)
    reduce_lr_callback = LearningRateScheduler(lr_schedule)

    logger_callback = TrainingLogger(keras_logdir)

    datagen_args = dict(
        rotation_range=20,
        zoom_range=[0.6, 1.2],
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='reflect'
    )

    img_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)
    seed = 1 #arbitrary seed
    img_flow = img_datagen.flow(x_train, batch_size=4, seed=seed, save_prefix='img')
    mask_flow = mask_datagen.flow(y_train, batch_size=4, seed=seed, save_prefix='mask')
    train_flow =  zip(img_flow, mask_flow)

    with open('{}/train_args.txt'.format(keras_logdir), 'w+'.format(keras_logdir)) as f:
        f.write('t_start: {}\n'.format(t_now))
        f.write('lrn_rate start: {}\n'.format(lrn_rate))
        f.write('lrn_rate step decay factor: {}\n'.format(lrn_rate_decay))
        f.write('lrn_rate step duration: {}\n'.format(lrn_rate_duration))
        f.write('data_augmentation: {}\n'.format(datagen_args))
        f.write('x_train: {} y_train {}\n'.format(x_train.shape, y_train.shape))
        f.write('x_val: {} y_val {}\n'.format(x_val.shape, y_val.shape))

    model.fit_generator(train_flow,
              steps_per_epoch=x_train.shape[0]//4,
              epochs=500,
              validation_data=(x_val, y_val),
              callbacks=[checkpoint_callback, tensorboard_callback, reduce_lr_callback, logger_callback],
                        verbose=1)

def evaluate_test(model):
    print('EVALUATING ON TEST SET')
    x_test, y_test = np.load('/media/fredrik/WDusbdrive/segmentation_training_data/xy_data_test.npy')

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
    with open('/media/fredrik/WDusbdrive/segmentation_training_data/paths_x_data_test.txt', 'r') as f:
        x_paths = [path.strip() for path in f.readlines()]
        print(x_paths)
        predict_img_path(model, x_paths[n])

def predict_img_path(model, img_path):
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
    #update_training_data('/media/fredrik/WDusbdrive/segmentation_training_data')
    #exit()
    lrn_rate = 1e-3
    model = get_compiled_model('pspnet50_all-train', lrn_rate)
    train(model, lrn_rate)
    exit()

    test_checkpoint = '/media/fredrik/WDusbdrive/keras_logdir/2019-04-05_13-25-31.056466//checkpoint-epoch074.h5'
    model = get_compiled_model('pspnet50_all-train', 0, checkpoint=test_checkpoint)
    for i in range(200,300):
        predict_single_test_image(model, i)
    #evaluate_test(model)


