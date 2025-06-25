import tensorflow as tf
from keras import layers, models
import cv2 as cv
from Evaluation import evaluate_error


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, routings=3, kernel_size=(9, 9), strides=(2, 2), padding='valid',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routings = routings
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        _, _, _, filters = input_shape
        self.capsules = self.add_weight(name='capsules', shape=(1, 1, filters, self.num_capsules * self.capsule_dim),
                                        initializer='glorot_uniform', trainable=True)

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, -1)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, 1, self.num_capsules, 1])  # Fix the tile dimensions

        caps_reshaped = tf.reshape(self.capsules, [1, 1, -1, self.capsule_dim])
        caps_tiled = tf.tile(caps_reshaped, [tf.shape(inputs)[0], tf.shape(inputs)[1], 1, 1])

        pred = tf.matmul(inputs_tiled, caps_tiled, transpose_a=True)
        pred = tf.reshape(pred, [-1, tf.shape(inputs)[1], tf.shape(inputs)[2], self.num_capsules, self.capsule_dim])

        return tf.reduce_sum(pred, axis=-2)


# Capsule Network model
def CapsNet(input_shape, classes):
    x = tf.keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(256, (9, 9), activation='relu', padding='valid', strides=(1, 1))(x)
    primary_capsules = CapsuleLayer(num_capsules=8, capsule_dim=32)(conv1)
    output_capsules = CapsuleLayer(num_capsules=1, capsule_dim=32)(primary_capsules)
    flat = layers.Flatten()(output_capsules)
    output = layers.Dense(classes, activation='sigmoid')(flat)

    model = models.Model(x, output)
    return model


def Model_Capsnet(x_train, y_train, Test_X, testy, Act=None, sol=None):

    if Act is None:
        Act = 1
    Activation = ['linear', 'relu', 'leaky relu', 'tanH', 'sigmoid', 'softmax']
    input = (28, 28, 3)
    Classes = testy.shape[-1]

    Train_Temp = np.zeros((x_train.shape[0], input[0], input[1], input[2]))
    for i in range(x_train.shape[0]):
        Train_Temp[i, :] = np.resize(x_train[i], (input[0], input[1], input[2]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input[0], input[1], input[2])

    Test_Temp = np.zeros((Test_X.shape[0], input[0], input[1], input[2]))
    for i in range(Test_X.shape[0]):
        Test_Temp[i, :] = np.resize(Test_X[i], (input[0], input[1], input[2]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], input[0], input[1], input[2])

    model = CapsNet(input_shape=input, classes=Classes)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    model.fit(Train_X, y_train, epochs=2, validation_data=(Test_X, testy))
    pred = model.predict(Test_X)
    Eval = evaluate_error(pred, testy)
    return Eval, pred


import numpy as np

if __name__ == '__main__':
    # Generate input train data
    x_train = np.random.randn(100, 28, 28, 3)

    # Generate input test data
    x_test = np.random.randn(100, 28, 28, 3)

    # Generate output train data
    y_train = np.random.randint(0, 10, (100,))

    # Generate output test data
    y_test = np.random.randint(0, 10, (100,))
    eval, pred = Model_Capsnet(x_train, y_train, x_test, y_test)
    fvbfjnb = 43
