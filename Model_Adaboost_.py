from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import keras
from Evaluation import evaluate_error


def Model_Adaboost(trainX, y_train, X_test, y_test, Act=None, sol=None):
    if Act is None:
        Act = 1
    Activation = ['linear', 'relu', 'leaky relu', 'tanH', 'sigmoid', 'softmax']

    IMG_SIZE = [50]
    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0])

    Test_Temp = np.zeros((X_test.shape[0], IMG_SIZE[0]))
    for i in range(X_test.shape[0]):
        Test_Temp[i, :] = np.resize(X_test[i], (IMG_SIZE[0]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0])

    if y_train.shape[1] > 1:
        y_Train = np.argmax(y_train, axis=1)
        y_Test = np.argmax(y_test, axis=1)
    else:
        y_Train = y_train
        y_Test = y_test
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=1.0)
    clf.fit(Train_X, y_Train)
    predict = clf.predict(Test_X)
    pred = keras.utils.to_categorical(predict)
    Eval = evaluate_error(pred, y_test)
    return Eval, pred


if __name__ == '__main__':
    a = np.random.randint(0, 255, (10, 256, 256, 3))
    b = np.random.randint(0, 2, (10, 2))
    c = np.random.randint(0, 255, (10, 256, 256, 3))
    d = np.random.randint(0, 2, (10, 2))
    eval, pred = Model_Adaboost(a, b, c, d)
