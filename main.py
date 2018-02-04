import glob
import h5py
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import DataPreprocessing as dp
import featuresExtract as fe
import classifier as cl
from submit import submit



#!!!!create data dict and param dict from config yml file

#if we want to do cross validation, need to use for loop

# load param configuration


with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

# if glob.glob('./dataset/dataset.h5'):
#     f = h5py.File('./dataset/dataset.h5', 'r')
#     x_train = f['x_train']
#     y_train = f['y_train']
#     x_test = f['x_test']

if glob.glob('./dataset/*.npy'):
    x_train = np.load("./dataset/training_data.npy")
    y_train = np.load("./dataset/training_label.npy")
    x_test = np.load("./dataset/testing_data.npy")

else:
    x_train, y_train, x_test = dp.dataPreprocessing(cfg)

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)



#doing cross validation
if cfg["lTrain"]:
    for _ in range(cfg['nFolds']):
            #start with feature extraction
        train_data, valid_data, train_labels, valid_labels = train_test_split(x_train, y_train, test_size= 0.2, random_state = 1)
        input_data_dict = {'x_train' : train_data, 'y_train' : train_labels, 'x_valid' : valid_data, 'y_valid' : valid_labels}
        #input_data_dict = {'x_train' : x_train, 'y_train' : y_train}

        training_feature_list, valid_feature_list = fe.trainingfeatureExtract(input_data_dict)
        #training_feature_list = fe.trainingfeatureExtract(input_data_dict)
        #after extracting bt neck features, start classifying the features
        #start with reading in features

        #two models can be ensembled together, the first way is to concatenate the features


        train_feature_ensemble = training_feature_list[0]
        for i in range(1,len(training_feature_list)):
            train_feature_ensemble = np.concatenate((train_feature_ensemble, training_feature_list[i]), axis = 1)

        valid_feature_ensemble = valid_feature_list[0]
        for i in range(1, len(valid_feature_list)):
            valid_feature_ensemble = np.concatenate((valid_feature_ensemble, valid_feature_list[i]), axis=1)

        feature_data_dict = {'x_train': train_feature_ensemble, 'y_train' : train_labels, 'x_valid' : valid_feature_ensemble, 'y_valid' : valid_labels}
        #feature_data_dict = {'x_train' : train_feature_ensemble, 'y_train' : y_train}
        print('start training classifier:')
        cl.trainclassifier(feature_data_dict, 120, cfg, "ensemble")
        print('Training finishes!!')

else:
    test_feature_list = fe.testingfeatureExtract(x_test, "ensemble")
    test_feature_ensemble = test_feature_list[0]
    for i in range(1,len(test_feature_list)):
        test_feature_ensemble = np.concatenate((test_feature_ensemble, test_feature_list[i]), axis = 1)
    y_predict = cl.predict_from_classifier("ensemble", test_feature_ensemble, cfg)
    submit(y_predict, "ensemble")
