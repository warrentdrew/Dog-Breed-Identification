import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from os.path import exists

batch_size = 32
input_size = 299

def generate_features(model_info, data, labels, datagen):
    print("generating features...")
    datagen.preprocessing_function = model_info["preprocessor"]
    generator = datagen.flow(data, labels, shuffle=False, batch_size=batch_size)
    bottleneck_model = model_info["model"](weights='imagenet', include_top=False, input_shape=model_info["input_shape"],
                                           pooling=model_info["pooling"])
    return bottleneck_model.predict_generator(generator)

def generate_test_features(model_info, x_test): #test features does not need to be generated from generator
    print("generating test features")            #can be generated from model.predict()
    #datagen.preprocessing_function = model_info["preprocessor"]
    # generator = datagen.flow_from_directory(
    #                             directory='./input/test',
    #                             target_size=(input_size, input_size),
    #                             batch_size=batch_size,
    #                             class_mode=None,
    #                             shuffle=False)
    bottleneck_model = model_info["model"](weights='imagenet', include_top=False, input_shape=model_info["input_shape"],
                                           pooling=model_info["pooling"])
    return bottleneck_model.predict(x_test)     #this is not the final predict, it can be in any dimension

def featureExtract(data_dict, models):
    for model_name, model in models.items():
        trainfilename = './features/' + model_name + '_trainfeatures.npy'
        validfilename = './features/' + model_name + '_validfeatures.npy'
        testfilename = './features/' + model_name + '_testfeatures.npy'
        if exists(trainfilename):
            training_features = np.load(trainfilename)
            validation_features = np.load(validfilename)
        else:
            train_datagen = ImageDataGenerator(             #generator for training
                zoom_range=0.3,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rescale = 1./255)
            validation_datagen = ImageDataGenerator(rescale= 1./255)       #generator for validation
            training_features = generate_features(model, data_dict["x_train"], data_dict["y_train"], train_datagen)
            validation_features = generate_features(model, data_dict["x_valid"], data_dict["y_valid"], validation_datagen)

            np.save(trainfilename, training_features)
            np.save(validfilename, validation_features)
        if exists(testfilename):
            test_features = np.load(trainfilename)
        else:
            #test_datagen = ImageDataGenerator(rescale= 1./255)
            test_features = generate_test_features(model, data_dict["x_test"])
            np.save(testfilename, test_features)

        print("test feature shape for model " + model + ": ", test_features.shape)

    return training_features, validation_features, test_features

