import numpy as np
import pandas as pd
import cv2              #pip install opencv-python
import h5py

# transform jpg pictures and corresponding labels into h5 file
def fPreprocessData(cfg):
    # load config param
    img_size = (cfg['img_size'][0], cfg['img_size'][1])

    # load csv data
    df_train = pd.read_csv('./input/labels.csv')
    df_test = pd.read_csv('./input/sample_submission.csv')

    targets_series = pd.Series(df_train['breed'])
    one_hot = pd.get_dummies(targets_series, sparse = True)
    one_hot_labels = np.asarray(one_hot)

    # initialize the training and testing variable
    # initialized as list
    x_train = []
    y_train = []
    x_test = []

    for i, row in df_train.iterrows():
        img = cv2.imread('./input/train/{}.jpg'.format(row.id))
        label = one_hot_labels[i]
        x_train.append(cv2.resize(img, img_size))
        y_train.append(label)

    for id in df_test['id'].values:
        img = cv2.imread('./input/test/{}.jpg'.format(id))
        x_test.append(cv2.resize(img, img_size))

    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float32) / 255.  #divide 255 is feature normalization?
    x_test  = np.array(x_test, np.float32) / 255.

    with h5py.File('./dataset/dataset.h5', 'w') as hf:
        hf.create_dataset("x_train", data=x_train, dtype='float32')
        hf.create_dataset("y_train", data=y_train, dtype='uint8')
        hf.create_dataset("x_test", data=x_test, dtype='float32')

    return x_train, y_train, x_test
