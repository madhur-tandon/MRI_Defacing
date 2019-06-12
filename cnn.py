from dataloader import get_data
from keras import layers, models
from keras import regularizers
import keras.backend as K
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

np.random.seed(42)

def pickle_model(model, name):
    pickling_on = open(name,"wb")
    pickle.dump(model, pickling_on)
    pickling_on.close()

def load_model(name):
    pickle_off = open(name,"rb")
    model = pickle.load(pickle_off)
    return model

train_set_1, train_labels = get_data('IXI-T1-Preprocessed', 'mean', 0, 32)
train_set_1 = train_set_1.reshape((train_set_1.shape[0],32,32,1))

train_set_2, train_labels = get_data('IXI-T1-Preprocessed', 'mean', 1, 32)
train_set_2 = train_set_2.reshape((train_set_2.shape[0],32,32,1))

train_set_3, train_labels = get_data('IXI-T1-Preprocessed', 'mean', 2, 32)
train_set_3 = train_set_3.reshape((train_set_3.shape[0],32,32,1))

train_set_4, train_labels = get_data('IXI-T1-Preprocessed', 'slice', 0, 32)
train_set_4 = train_set_4.reshape((train_set_4.shape[0],32,32,1))

train_set_5, train_labels = get_data('IXI-T1-Preprocessed', 'slice', 1, 32)
train_set_5 = train_set_5.reshape((train_set_5.shape[0],32,32,1))

train_set_6, train_labels = get_data('IXI-T1-Preprocessed', 'slice', 2, 32)
train_set_6 = train_set_6.reshape((train_set_6.shape[0],32,32,1))

test_set_1, test_labels = get_data('IXI-T1-Preprocessed', 'mean', 0, 32, False)
test_set_1 = test_set_1.reshape((test_set_1.shape[0],32,32,1))

test_set_2, test_labels = get_data('IXI-T1-Preprocessed', 'mean', 1, 32, False)
test_set_2 = test_set_2.reshape((test_set_2.shape[0],32,32,1))

test_set_3, test_labels = get_data('IXI-T1-Preprocessed', 'mean', 2, 32, False)
test_set_3 = test_set_3.reshape((test_set_3.shape[0],32,32,1))

test_set_4, test_labels = get_data('IXI-T1-Preprocessed', 'slice', 0, 32, False)
test_set_4 = test_set_4.reshape((test_set_4.shape[0],32,32,1))

test_set_5, test_labels = get_data('IXI-T1-Preprocessed', 'slice', 1, 32, False)
test_set_5 = test_set_5.reshape((test_set_5.shape[0],32,32,1))

test_set_6, test_labels = get_data('IXI-T1-Preprocessed', 'slice', 2, 32, False)
test_set_6 = test_set_6.reshape((test_set_6.shape[0],32,32,1))

input_size = 32
batch_size = 16
n_epochs = 100
n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print("Data Loaded")


class CNN_Classifier(object):

    def __init__(self, input_shape=(32, 32)):
        self.input_shape = input_shape

    def relu6(self, x):
        return K.relu(x, max_value=6)

    def Conv_BN_RELU(self, x, filters, kernel, strides, padding):
        x = layers.Conv2D(filters, kernel, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.relu6)(x)
        return x

    def create_submodel(self):
        inp = layers.Input(shape=(32, 32, 1))

        conv1 = self.Conv_BN_RELU(
            inp, filters=8, kernel=3, strides=1, padding='same')
        conv1 = self.Conv_BN_RELU(
            conv1, filters=8, kernel=3, strides=1, padding='same')
        conv1 = layers.MaxPooling2D()(conv1)

        conv2 = self.Conv_BN_RELU(
            conv1, filters=16, kernel=3, strides=1, padding='same')
        conv2 = self.Conv_BN_RELU(
            conv2, filters=16, kernel=3, strides=1, padding='same')
        conv2 = layers.MaxPooling2D()(conv2)

        conv3 = self.Conv_BN_RELU(
            conv2, filters=32, kernel=3, strides=1, padding='same')
        conv3 = self.Conv_BN_RELU(
            conv3, filters=32, kernel=3, strides=1, padding='same')
        conv3 = layers.MaxPooling2D()(conv3)

        out = layers.Flatten()(conv3)

        submodel = models.Model(inp, out)

        print(submodel.summary())

        return submodel

    def create_model(self):

        inp1 = layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 1), name='input_1')
        inp2 = layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 1), name='input_2')
        inp3 = layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 1), name='input_3')
        inp4 = layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 1), name='input_4')
        inp5 = layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 1), name='input_5')
        inp6 = layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], 1), name='input_6')

        submodel = self.create_submodel()

        one = submodel(inp1)
        two = submodel(inp2)
        three = submodel(inp3)
        four = submodel(inp4)
        five = submodel(inp5)
        six = submodel(inp6)

        concat_1 = layers.Add()([one, two, three])
        concat_2 = layers.Add()([four, five, six])
        
        out_1 = layers.Dense(32, activation='sigmoid')(concat_1)
        out_2 = layers.Dense(32, activation='sigmoid')(concat_2)
        
        dropout_1 = layers.Dropout(0.5)(out_1)
        dropout_2 = layers.Dropout(0.5)(out_2)
        
        concat = layers.Add()([dropout_1, dropout_2])
        out = layers.Dense(1, activation='sigmoid',
                           name='output_node')(concat)

        return models.Model(inputs=[inp1, inp2, inp3, inp4, inp5, inp6], outputs=out)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (all_positives + K.epsilon())

def recall(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    denominator = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (denominator + K.epsilon())


input_shape = (input_size, input_size)
classifier = CNN_Classifier(input_shape=input_shape)
model = classifier.create_model()
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', precision, recall])
print(model.summary())

hist = model.fit([
           train_set_1,
           train_set_2,
           train_set_3,
           train_set_4,
           train_set_5,
           train_set_6
          ],
          train_labels,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_split = 0.2,
          shuffle = True
          )

metrics = model.evaluate([
                          test_set_1,
                          test_set_2,
                          test_set_3,
                          test_set_4,
                          test_set_5,
                          test_set_6
                          ],test_labels,batch_size=batch_size)

print(metrics)
