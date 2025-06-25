import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt



def Plot_Batch_size():
    eval = np.load('Eval_All_err_BS.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE', 'Accuracy']
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 10, 11]

    Classifier = ['TERMS', 'DNN', 'Adaboost', 'CapsNet', 'Extreme learning', 'Ensemble Learning']
    Batchsize = [4, 8, 16, 32, 48]
    for i in range(eval.shape[0]):
        value = eval[i, :, :]
        Table = PrettyTable()
        Table.add_column(Classifier[0], np.asarray(Terms)[np.asarray(Table_Terms)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, Table_Terms])
        print('-------------------------------------------------- ', Batchsize[i], 'Batch size',
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)



def Plot_Activation():
    eval = np.load('Eval_All_err_act.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE', 'Accuracy']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j]]

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(Graph.shape[0])

        ax.bar(X + 0.00, Graph[:, 0], color='#0165fc', edgecolor='w', width=0.15, label="DNN")
        ax.bar(X + 0.15, Graph[:, 1], color='#ff474c', edgecolor='w', width=0.15, label="Adaboost")
        ax.bar(X + 0.30, Graph[:, 2], color='#be03fd', edgecolor='w', width=0.15, label="CapsNet")
        ax.bar(X + 0.45, Graph[:, 3], color='#21fc0d', edgecolor='w', width=0.15, label="Extreme learning")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="Ensemble Learning")
        plt.xticks(X + 0.15, ('linear', 'relu', 'leaky relu', 'tanH', 'sigmoid', 'softmax'))
        plt.xlabel('Activation Function', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Activation_%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    Plot_Batch_size()
    Plot_Activation()



def Plot_Results__():
    eval = np.load('Eval_All_err_BS.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE', 'Accuracy']
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    Classifier = ['TERMS', 'DNN', 'Adaboost', 'CapsNet', 'Extreme learning', 'Ensemble Learning']
    Batchsize = [4, 8, 16, 32, 48]
    for i in range(eval.shape[0]):
        value = eval[i, :, :]
        Table = PrettyTable()
        Table.add_column(Classifier[0], np.asarray(Terms)[np.asarray(Table_Terms)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, Table_Terms])
        print('-------------------------------------------------- ', Batchsize[i], 'Batch size',
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j]]

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(Graph.shape[0])

        ax.bar(X + 0.00, Graph[:, 0], color='#0165fc', edgecolor='w', width=0.15, label="DNN")
        ax.bar(X + 0.15, Graph[:, 1], color='#ff474c', edgecolor='w', width=0.15, label="Adaboost")
        ax.bar(X + 0.30, Graph[:, 2], color='#be03fd', edgecolor='w', width=0.15, label="CapsNet")
        ax.bar(X + 0.45, Graph[:, 3], color='#21fc0d', edgecolor='w', width=0.15, label="Extreme learning")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="Ensemble Learning")
        plt.xticks(X + 0.15, ('linear', 'relu', 'leaky relu', 'tanH', 'sigmoid', 'softmax'))
        plt.xlabel('Activation Function', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Activation_%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()
