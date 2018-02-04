import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from os.path import exists

#batch_size = 32
models ={ "InceptionV3": {
                    "model": InceptionV3,
                    "pooling": "avg"
                },
            "InceptionResNetV2": {
                    "model": InceptionResNetV2,
                    "pooling": "avg"
                }
        }

def generate_features(model_info, data, labels, datagen):
    print("generating features...")
    #datagen.preprocessing_function = model_info["preprocessor"]
    generator = datagen.flow(data, labels, shuffle=False, batch_size=32)
    #print("training data num:", len(generator.filename))
    #print("training label num:", len(generator.class_indices))
    bottleneck_model = model_info["model"](weights='imagenet', include_top=False, pooling=model_info["pooling"])
    #except for features, we need also to return the corresponding labels
    # get the class labels for the training data, in the original order
    # the generator will augment data in the same sequence as the original data if shuffle = False
    # so fitting one epoch will be get all the same labels
    #feature_labels = generator.classes
    #print("feature labels:", feature_labels)
    # convert the training labels to categorical vectors
    #feature_labels = to_categorical(feature_labels, num_classes = len(generator.class_indices)) #changing to one-hot
    return bottleneck_model.predict_generator(generator, verbose= 1)

def generate_test_features(model_info, x_test): #test features does not need to be generated from generator
    print("generating test features")            #can be generated from model.predict()
    #datagen.preprocessing_function = model_info["preprocessor"]
    # generator = datagen.flow_from_directory(
    #                             directory='./input/test',
    #                             target_size=(input_size, input_size),
    #                             batch_size=batch_size,
    #                             class_mode=None,
    #                             shuffle=False)
    bottleneck_model = model_info["model"](weights='imagenet', include_top=False, pooling=model_info["pooling"])
    return bottleneck_model.predict(x_test)     #this is not the final predict, it can be in any dimension

def trainingfeatureExtract(input_data_dict):
    training_feature_list = []
    valid_feature_list = []
    for model_name, model in models.items():
        trainfilename = './features/' + model_name + '_trainfeatures.npy'
        validfilename = './features/' + model_name + '_validfeatures.npy'
        #trainlabelfilename = './features/' + model_name + '_trainfeatureslabels.npy'
        #validlabelfilename = './features/' + model_name + '_validfeatureslabels.npy'
        if exists(trainfilename):
            training_features = np.load(trainfilename)
            #validation_features = np.load(validfilename)
            training_feature_list.append(training_features)
            #valid_feature_list.append(validation_features)
        else:
            train_datagen = ImageDataGenerator(             #generator for training, don't need to rescale since already rescaled
                zoom_range=0.3,
                width_shift_range=0.1,
                height_shift_range=0.1,
                )
            #validation_datagen = ImageDataGenerator(rescale= 1./255)       #generator for validation
            training_features = generate_features(model, input_data_dict["x_train"], input_data_dict["y_train"], train_datagen)
            #validation_features = generate_features(model, input_data_dict["x_valid"], input_data_dict["y_valid"], validation_datagen)

            training_feature_list.append(training_features)
            #valid_feature_list.append(validation_features)
            np.save(trainfilename, training_features)
            #np.save(validfilename, validation_features)

            #also to save the
            #np.save(trainlabelfilename, training_labels)
            #np.save(validlabelfilename, validation_labels)
        print("training feature shape for model: " + str(model) + ": ", training_features.shape)
        if exists(validfilename):
            valid_features = np.load(validfilename)
            valid_feature_list.append(valid_features)
        else:
            #validation is different from training as it does not need gata augmentation
            #training process use the top layers while in validation it is not necessary
            #so only need to do prediction
            valid_features = generate_test_features(model, input_data_dict["x_valid"])  # generator for validation
            valid_feature_list.append(valid_features)
            np.save(validfilename,valid_features)
        print("training feature shape for model: " + str(model) + ": ", valid_features.shape)

    return training_feature_list, valid_feature_list

def testingfeatureExtract(x_test):
    test_feature_list = []
    for model_name, model in models.items():
        testfilename = './features/' + model_name + '_testfeatures.npy'
        if exists(testfilename):
            test_features = np.load(testfilename)
            test_feature_list.append(test_features)
        else:
            #test_datagen = ImageDataGenerator(rescale= 1./255)

            test_features = generate_test_features(model, x_test)
            np.save(testfilename, test_features)
            test_feature_list.append(test_features)


        print("test feature shape for model: " + str(model) + ": ", test_features.shape)

    return test_feature_list
