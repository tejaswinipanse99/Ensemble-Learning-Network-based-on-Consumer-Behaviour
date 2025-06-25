from Evaluation import evaluate_error
from Model_Adaboost_ import Model_Adaboost
from Model_Capsnet import Model_Capsnet
from Model_ELM import Model_ELM

def HighRanking(Pred):
    Pred = np.asarray(Pred)
    pred = np.zeros((Pred.shape[1], 1))
    for i in range(Pred.shape[1]):
        p = Pred[:, i]
        uniq, count = np.unique(p, return_counts=True)
        index = np.argmax(count)
        pred[i] = uniq[index]

    return pred

def Model_Ensemble_Learning(Train_Data, Train_Target, Test_Data, Test_Target, Act):

    Eval1, pred1 = Model_Adaboost(Train_Data, Train_Target, Test_Data, Test_Target, Act)
    Eval2, pred2 = Model_Capsnet(Train_Data, Train_Target, Test_Data, Test_Target, Act)
    Eval3, pred3 = Model_ELM(Train_Data, Train_Target, Test_Data, Test_Target, Act)

    pred = [np.reshape(pred1, (-1, 1)), np.reshape(pred2, (-1, 1)), np.reshape(pred3, (-1, 1))]
    Predict = HighRanking(pred)
    Eval = evaluate_error(Predict, Test_Target)
    return Eval