import glob
import h5py
import yaml
from sklearn.model_selection import train_test_split
import DataPreprocessing as dp
import featuresExtract as fe

#!!!!create data dict and param dict from config yml file

#if we want to do cross validation, need to use for loop

# load param configuration


with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

if glob.glob('./dataset/dataset.h5'):
    f = h5py.File('./dataset/dataset.h5', 'r')
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']

else:
    x_train, y_train, x_test = dp.dataPreprocessing(cfg)

#doing cross validation
for _ in range(cfg['nFolds']):
    train_data, valid_data, train_labels, valid_labels = train_test_split(x_train, y_train,
                                                                            test_size=cfg['validSplit'],
                                                                            random_state=1)
    data_dict = {'x_train': train_data, 'x_valid': valid_data, 'y_train': train_labels,
                     'y_valid': valid_labels, 'x_test': x_test}

    fe.featureExtract(data_dict, models= cfg["models"])