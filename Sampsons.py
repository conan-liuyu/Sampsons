import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from mpl_toolkits.axes_grid1 import AxesGrid
from keras.optimizers import SGD
import os

map_characters = {0 : 'abraham_grampa_simpson',
                  1 : 'agnes_skinner',
                  2 : 'apu_nahasapeemapetilon',
                  3 : 'barney_gumble',
                  4 : 'bart_simpson',
                  5 : 'carl_carlson',
                  6 : 'charles_montgomery_burns',
                  7 : 'chief_wiggum',
                  8 : 'cletus_spuckler',
                  9 : 'comic_book_guy',
                  10 : 'disco_stu',
                  11 : 'edna_krabappel',
                  12 : 'fat_tony',
                  13 : 'gil',
                  14 : 'groundskeeper_willie',
                  15 : 'homer_simpson',
                  16 : 'kent_brockman',
                  17 : 'krusty_the_clown',
                  18 : 'lenny_leonard',
                  19 : 'lionel_hutz',
                  20 : 'lisa_simpson',
                  21 : 'maggie_simpson',
                  22 : 'marge_simpson',
                  23 : 'martin_prince',
                  24 : 'mayor_quimby',
                  25 : 'milhouse_van_houten',
                  26 : 'miss_hoover',
                  27 : 'moe_szyslak',
                  28 : 'ned_flanders',
                  29 : 'nelson_muntz',
                  30 : 'otto_mann',
                  31 : 'patty_bouvier',
                  32 : 'principal_skinner',
                  33 : 'professor_john_frink',
                  34 : 'rainier_wolfcastle',
                  35 : 'ralph_wiggum',
                  36 : 'selma_bouvier',
                  37 : 'sideshow_bob',
                  38 : 'sideshow_mel',
                  39 : 'snake_jailbird',
                  40 : 'troy_mcclure',
                  41 : 'waylon_smithers'}
check_characters = {v:k for k,v in map_characters.items()}

pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(map_characters)

def create_model_six_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    return model, opt

def load_model_from_checkpoint(weights_path, input_shape=(pic_size,pic_size,3)):
    model, opt = create_model_six_conv(input_shape)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def load_test_set(path):
    pics, labels = [], []
    reverse_dict = {v:k for k,v in map_characters.items()}
    paths = os.listdir(path)
    for pic in paths:
        pic = path + pic
        char_name = "_".join(pic.split('/')[3].split('_')[:-1])
        if char_name in reverse_dict.keys():
            temp = cv2.imread(pic)
            temp = cv2.resize(temp, (pic_size, pic_size)).astype('float32')/255
            pics.append(temp)
            labels.append(reverse_dict[char_name])
    X_test = np.array(pics)
    y_test = np.array(labels)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('Test set', X_test.shape, y_test.shape)
    return X_test, y_test

model = load_model_from_checkpoint(r'E:\the-simpsons-characters-dataset\weights.best.hdf5')
#X_test, y_test = load_test_set('E:/the-simpsons-characters-dataset/kaggle_simpson_testset/')

def img_pro(path):
    F = plt.figure(1, (15,20))
    grid = AxesGrid(F, 111, nrows_ncols=(4,4), axes_pad=0, label_mode='1')


    for i in range(16):
        char = map_characters[i]
        list = [path+k for k in os.listdir(path) if char in k]
        image = cv2.imread(np.random.choice(list))


        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(image, (pic_size,pic_size)).astype('float32')/255
        a = model.predict(pic.reshape(1, pic_size, pic_size, 3))[0]
        actual = char.split('_')[0].title()
        text = sorted(['{:s} : {:.1f}%'.format(map_characters[i].split('_')[0].title(), 100*v) for k,v in enumerate(a)],
                      key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
        img = cv2.resize(img, (352, 352))
        cv2.rectangle(img, (0,260), (215,352), (255,255,255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Actual : %s' % actual, (10,280), font, 0.7, (0,0,0), 2, cv2.LINE_AA)
        for k,t in enumerate(text):
            cv2.putText(img, t, (10, 300+k*18), font, 0.65, (0,0,0), 2, cv2.LINE_AA)
        grid[i].imshow(img)
    plt.show()
path = 'E:/the-simpsons-characters-dataset/kaggle_simpson_testset/'     #修改为自己的路径
img_pro(path)

def video_pro(path):
    video_capture = cv2.VideoCapture(path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_write = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('m','p','4','v'), fps, size)
    success, frame = video_capture.read()
    while success:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(frame, (pic_size,pic_size)).astype('float32')/255
        a = model.predict(pic.reshape(1, pic_size, pic_size, 3))[0]
        text = sorted(['{:s} : {:.1f}%'.format(map_characters[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)],
                      key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
        img = cv2.resize(img, (size[0], size[1]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        for k,t in enumerate(text):
            cv2.putText(img, t, (10, 680+k*18), font, 0.65, (0,0,0), 2, cv2.LINE_AA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_write.write(img)
        success, frame = video_capture.read()
    video_write.release()
    video_capture.release()
path = r'D:\app\pycharm\codes\deeplearning\HongyiLi\yyy.mp4'    #修改为自己的路径
video_pro((path))
