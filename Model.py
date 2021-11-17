
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

import os
dirPath = os.path.dirname(os.path.abspath(__file__))





def drawConfusionMatrix(y_pred,y_test):
        cm = confusion_matrix(y_test, y_pred)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Bình dân','Trung lưu', 'Thượng lưu']
        plt.title('Confusion_matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        for i in range(3):
            for j in range(3):
                plt.text(j,i, str(cm[i][j]))
        plt.show()
        precision_0 = cm[0][0] / (cm[0][0] + cm[1][0] + cm[2][0])
        precision_1 = cm[1][1] / (cm[0][1] + cm[1][1] + cm[2][1])
        precision_2 = cm[2][2] / (cm[0][2] + cm[1][2] + cm[2][2])
        recall_0 = cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])
        recall_1 = cm[1][1] / (cm[1][0] + cm[1][1] + cm[1][2])
        recall_2 = cm[2][2] / (cm[2][0] + cm[2][1] + cm[2][2])
        precision = (precision_0 + precision_1 + precision_2) / 3
        print("Precision:", precision)
        recall = (recall_0 + recall_1 + recall_2) / 3
        print("Recall:", recall)
        print("F1 Score:", 2 * precision * recall /(precision + recall))



class RandomForestModel:

    

    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators


    def fitModel(self,data):

        X_train, X_val, y_train, y_val = data

        clf = RandomForestClassifier(n_estimators=self.n_estimators)

        clf.fit(X_train, y_train)

        print("Accuracy on validation data: {}".format(clf.score(X_val, y_val)))
        
        self.saveModel(clf)


    def saveModel(self,model):
        save_Path = os.path.join(dirPath,'Model/RandomForestModel.sav')
        pickle.dump(model, open(save_Path, 'wb'))

    def loadModel(self):
        save_Path = os.path.join(dirPath,'Model/RandomForestModel.sav')
        return pickle.load(open(save_Path, 'rb'))

    def evaluateModel(self,data):

        X_test, y_test = data

        clf = self.loadModel()

        y_pred = clf.predict(X_test)

        drawConfusionMatrix(y_pred,y_test)


    



class MLPModel:

    def __init__(self, batch_size=16,num_epochs=50,num_fold_steps=10,num_class=3):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_fold_steps = num_fold_steps
        self.num_class = num_class

    def get_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(self.num_class,activation='softmax'))

        model.compile(loss = 'sparse_categorical_crossentropy',
                        optimizer = 'Adam',
                        metrics = ['accuracy']
                        )
        return model

    def fitModel(self,data):

        X_train, X_val, y_train, y_val = data

        kfold = KFold(n_splits=self.num_fold_steps, shuffle=True)

        fold_idx = 1

        for train_ids, val_ids in kfold.split(X_train,y_train):
            train = list(train_ids)
            val = list(val_ids)
            my_file = Path(os.path.join(dirPath,'Model/ModelMLP.h5'))
            if my_file.exists():
                model = tf.keras.models.load_model(os.path.join(dirPath,'Model/ModelMLP.h5'))
            else:
                model = self.get_model()
            print("Start Fold ",fold_idx)
            model.fit(X_train.iloc[train,:], y_train.iloc[train,:],
                        batch_size = self.batch_size,
                        epochs = self.num_epochs,
                        verbose = 1
                        )
            
            scores = model.evaluate(X_train.iloc[val,:], y_train.iloc[val,:], verbose = 0)
            model.save(os.path.join(dirPath,'Model/ModelMLP.h5'))

            print('Fold: {}'.format(fold_idx))
            print("Accuracy: {}\n Loss: {}".format(scores[1]*100, scores[0]))

            fold_idx = fold_idx + 1

    def loadModel(self):
        return tf.keras.models.load_model(os.path.join(dirPath,'Model/ModelMLP.h5'))

    def evaluateModel(self, data):

        X_test, y_test = data

        model = self.loadModel()

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred,axis=1)

        drawConfusionMatrix(y_pred,y_test)

    






    