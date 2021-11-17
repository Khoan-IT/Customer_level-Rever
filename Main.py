from Model import RandomForestModel, MLPModel
from ProcessData import *

from sklearn.model_selection import train_test_split
import os
dirPath = os.path.dirname(os.path.abspath(__file__))

list_of_features = ['Phân mức', 'Chỗ ở', 'Lương', 'Phương tiện', 'Thẻ tín dụng', 'Tuổi', 'Bằng cấp', 'Công việc']

def dataTrain(test_size):
    dataTrain_Path = os.path.join(dirPath,'Data/Training/Train.xlsx')
    data = process(dataTrain_Path)

    data = data.drop_duplicates(list_of_features)

    X = data[list_of_features[1:]]
    y = data[[list_of_features[0]]]

    return train_test_split(X, y, test_size=test_size)

def dataTest():
    dataTest_Path = os.path.join(dirPath,'Data/Testing/Test.xlsx')
    data = process(dataTest_Path)

    data_test = data.drop_duplicates(list_of_features)

    X_test = data_test[list_of_features[1:]]
    y_test = data_test[[list_of_features[0]]]

    return (X_test, y_test)


if __name__ == '__main__':

    #Model Multilayer Perceptron with K-Fold
    #Parameter: batch_size  = 16 , num_epochs = 50, num_fold_steps = 10, num_class=3

    # MLP = MLPModel()
    # MLP.fitModel(dataTrain(0.2))
    # MLP.evaluateModel(dataTest())

    #Model Random Forest 
    #Parameter: n_estimatos = 100

    RF = RandomForestModel(10)
    RF.fitModel(dataTrain(0.15))
    RF.evaluateModel(dataTest())
    