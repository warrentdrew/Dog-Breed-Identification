import glob

from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import os

def createClassifier(data_dict, nClass):
    model = Sequential()
    model.add(BatchNormalization(input_shape=data_dict['train_data'].shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='sigmoid'))

    return model

def trainClassifier(data_dict, nClass, param_dict, model_name):
    model_file = './classifier/' + model_name + '_' + str(param_dict['img_size'][0]) + '_bs_' \
                 + str(param_dict['batchSize']) + '_model.h5'

    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = createClassifier(data_dict, nClass)


    model.compile(optimizer=param_dict['sOpti'], loss='categorical_crossentropy', metrics=['accuracy'])

    callback_list = [EarlyStopping(monitor='val_loss', patience=15, verbose=1)]
    callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1))
    callback_list.append(ModelCheckpoint(model_file))

    model.fit(data_dict['train_data'], data_dict['train_labels'],
                        epochs=param_dict['epochs'],
                        batch_size=param_dict['batchSize'],
                        validation_split=param_dict['validSplit'],
                        callbacks=callback_list)

    metrics = model.evaluate(data_dict['valid_data'], data_dict['valid_labels'], batch_size=param_dict['batchSize'], verbose=1)


    print('training data results:')
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

    if not os.path.exists(model_file):
        model.save(model_file)                #be sure to save the model if not existing

    return model

def predict_from_classifier(model_name, test_feature, param_dict):
    model_file = './model/' + model_name + '_' + str(param_dict['img_size'][0]) + '_bs_' \
                 + str(param_dict['batchSize']) + '_model.h5'

    if glob.glob(model_file):
        model = load_model(model_file)
        y_predict = model.predict(test_feature, batch_size=param_dict['batchSize'], verbose=1)

    else:
        print("The model is not exist!")

    return y_predict