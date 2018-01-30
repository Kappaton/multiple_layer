import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2

from keras.layers.merge import concatenate

#import matplotlib.image as mpimg
from scipy.misc import imresize
import numpy as np
from keras.backend import tensorflow_backend as backend
import keras.backend as K
import math
import keras
import matplotlib

#for hyperdash
from hyperdash import Experiment
from hyperdash_callback import Hyperdash

#add
#from keras.layers.advanced_activations import LeakyReLU
#leaky_relu = LeakyReLU()

#matplotlib.use('Agg')
#from keras.utils.visualize_util import plot
'''
### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###
'''
nb_classes=4
batch_size=32
#optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
'''
def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    #plt.show()
    plt.savefig('acc.png')

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    #plt.show()
    plt.savefig('loss.png')
'''
'''
model = Sequential()
model.add(Conv2D(32,(3, 3), padding='same', input_shape=(250, 250, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(32,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'Nadam',
              metrics=['accuracy'])
'''
#start
exp = Experiment("Experiment 1")

input_img = Input(shape=(250, 250, 3))
#input_img2 = Input(shape=(250, 250, 3))

tower_1 = Conv2D(42, (3, 3), activation='relu', padding='same')(input_img)
#tower_1 = BatchNormalization()(tower_1)
tower_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_1)
#tower_1 = BatchNormalization()(tower_1)
#tower_1 = Dropout(.2)(tower_1)

tower_1 = Conv2D(74, (3, 3), activation='relu', padding='same')(tower_1)
#tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_1)
#tower_1 = BatchNormalization()(tower_1)
#tower_1 = Dropout(.25)(tower_1)

tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
#tower_1 = MaxPooling2D((2, 2))(tower_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_1)
#tower_1 = BatchNormalization(axis=1)(tower_1)
#tower_1 = Dropout(.5)(tower_1)

#tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_1)
#add

tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_1)
merged_vector = Dropout(.5)(tower_1)
#tower_1 = BatchNormalization()(tower_1)
'''''''''''
#####################
tower_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#tower_2 = BatchNormalization()(tower_2)
tower_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_2)
#tower_2 = Dropout()(tower_2)

tower_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_2)

tower_2 = Conv2D(24, (3, 3), activation='relu', padding='same')(tower_2)
tower_2 = Conv2D(24, (3, 3), activation='relu', padding='same')(tower_2)
tower_2 = Conv2D(24, (3, 3), activation='relu', padding='same')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_2)

tower_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_2)

tower_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_2)
tower_2 = Dropout(.5)(tower_2)

###########################

tower_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
#tower_3 = BatchNormalization()(tower_3)
tower_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_3)

tower_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_3)

tower_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(tower_3)
tower_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(tower_3)
tower_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_3)

tower_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_3)

tower_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_3)
tower_3 = Dropout(.5)(tower_3)

############################

tower_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)   #tower_3 = BatchNormalization()(tower_3)
tower_4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_4)

tower_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(tower_4)
tower_4 = BatchNormalization()(tower_4)
tower_4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_4)

tower_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(tower_4)
tower_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(tower_4)
tower_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(tower_4)
tower_4 = BatchNormalization()(tower_4)
tower_4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_4)

tower_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(tower_4)
tower_4 = BatchNormalization()(tower_4)
tower_4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_4)

tower_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(tower_4)
tower_4 = BatchNormalization()(tower_4)
tower_4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_4)
tower_4 = Dropout(.5)(tower_4)
'''
#merged_vector = concatenate([tower_1, tower_2, tower_3, tower_4])
#merged_vector = Dropout(.5)(merged_vector)
merged_vector = BatchNormalization()(merged_vector)

merged_vector = Flatten()(merged_vector)

#merged_vector = Dense(254, activation='relu')(merged_vector)
#merged_vector = Dropout(.5)(merged_vector)
merged_vector = Dense(254, activation='relu')(merged_vector)
merged_vector = Dropout(.5)(merged_vector)

#tower_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
#tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_2)
#tower_2 = Flatten()(tower_2)
#ower_2 = Dropout(.2)(tower_2)

# We can then concatenate the two vectors:
#merged_vector = keras.layers.concatenate([tower_1, tower_2])

output = Dense(nb_classes, activation='softmax')(merged_vector)

#assert tower_1.get_input_shape_at(0) == (None, 250, 250, 3)
#assert tower_2.get_input_shape_at(1) == (None, 64, 64, 3)

model = Model(inputs=input_img, outputs=output)

model.compile(loss='categorical_crossentropy',
              optimizer = 'Nadam',
              metrics=['accuracy'])

'''
input_img1 = Input(shape=(250, 250, 3))
input_img2 = Input(shape=(250, 250, 3))

conv = Conv2D(32, (3, 3), activation='relu')
conved_1 = conv(input_img1)
conved_2 = conv(input_img2)

merged_vector = keras.layers.concatenate([conved_1, conved_2])
output = Dense(nb_classes, activation='softmax')(merged_vector)

assert conv.get_input_shape_at(0) == (None, 250, 250, 3)
assert conv.get_input_shape_at(1) == (None, 250, 250, 3)

model = Model(inputs=[input_img1,input_img2], outputs=output)

model.compile(loss='categorical_crossentropy',
               optimizer = 'Nadam',
               metrics=['accuracy'])
'''
#model.load_weights('/home/student/data_2017/checkpoint2/model.11-0.70.hdf5', by_name=True)
'''
tb_cb = keras.callbacks.TensorBoard(log_dir="/home/student/data_2017/log/", histogram_freq=1)
cbks = [tb_cb]
'''
#plot(model, to_file="model.png", show_shapes=True)
print(model.summary())

# ディレクトリの画像を使ったジェネレータ
train_datagen = ImageDataGenerator(
     rescale=1./255,
     fill_mode='nearest',
     shear_range=0.2,
     zoom_range=0.2
     )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='/home/student/data_2017/train',
    target_size=(250, 250),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    directory='/home/student/data_2017/validation',
    target_size=(250, 250),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

"""
#画像の読み込み
def load_images(root,nb_img):
    all_imgs = []
    all_classes = []
    for i in range(nb_img):
        img_name = "%s/%d.jpg" % (root, i + 1)
        img_arr = mpimg.imread(img_name)
        resize_img_ar = imresize(img_arr, (img_rows, img_cols))
        all_imgs.append(resize_img_ar)
        all_classes.append(0)
    return np.array(all_imgs), np.array(all_classes)

X_train, y_train = load_images('/home/student/data_2017/train/scarf', 2385)
X_test, y_test = load_images('/home/student/data_2017/validation/scarf', 361)

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=100,
    shuffle=True,
    )

test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=100,
    shuffle=True,
    )
"""
checkpointer = ModelCheckpoint(filepath='/home/student/data_2017/checkpoint62/model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)
csv_logger = CSVLogger('model.csv')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
patience=5, min_lr=0.001)
#5 3
#Early-Stopping
#early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

hd_callback = Hyperdash(exp=exp)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=92830//batch_size,
    epochs=100,
    validation_data=test_generator,
    validation_steps=23200//batch_size,
    verbose=1,
    callbacks=[reduce_lr, csv_logger, checkpointer, hd_callback])
#save_history(history, os.path.join(result_dir, 'history_smallcnn.txt'))

# 学習履歴をプロット
#plot_history(history)
#KTF.set_session(old_session)

#hyperdash end
exp.end()
backend.clear_session()

