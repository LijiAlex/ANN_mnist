import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class ANN:
    """[Create an ANN Model]
    """
    def __init__(self):
        """
        [Creates an ANN model with 
        * input layer of shape 28X28
        * hidden layer1 of 300 and activation function relu
        * hidden layer1 of 100 and activation function relu
        * output layer of 10 and activation function softmax
        * LOSS_FUNCTION = "sparse_categorical_crossentropy"
        * OPTIMIZER = "SGD"
        * METRICS = ["accuracy"]
        ]
        """
        def createModel(self):        
            LAYERS = [
                        tf.keras.layers.Flatten(input_shape=[28,28], name ="inputLayer"),
                        tf.keras.layers.Dense(300, activation="relu", name ="hiddenLayer1"),
                        tf.keras.layers.Dense(100, activation="relu", name ="hiddenLayer2"),
                        tf.keras.layers.Dense(10, activation="softmax", name ="outputLayer")
            ]
            model_clf = tf.keras.models.Sequential(LAYERS) #data flows sequentially ..no skipping or jumping
            model_clf.summary() 
            LOSS_FUNCTION = "sparse_categorical_crossentropy"
            OPTIMIZER = "SGD"
            METRICS = ["accuracy"]
            model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
            return model_clf
        self.model = createModel(self)

    def trainModel(self,EPOCHS,trainingData,validationData):
        """[Trains the model]

        Args:
            EPOCHS ([integer]): [No. of rounds of training]
            trainingData ([numpy array]): [Data for training]
            validationData ([numpy array]): [Data for validation]
        """
        (X_train,y_train) = trainingData
        history = self.model.fit(X_train,y_train, epochs=EPOCHS, validation_data=validationData)
        def plot_history():
            pd.DataFrame(history.history)
            pd.DataFrame(history.history).plot(figsize=(10,7))
            plt.grid(True)
            plt.show()
        plot_history()

    def evaluateModel(self,testData):
        (x_test, y_test) = testData
        results = self.model.evaluate(x_test, y_test, batch_size=128)
        print("test loss, test acc:", results)

    def saveModel(self, filename):
        self.model.save(filename)
        
    def plotModel(self,testData):
        """[Plot the model output using test data]

        Args:
            testData ([numpy array]): [Data for testing]
        """
        X_test,y_test = testData
        y_prob = self.model.predict(X_test)
        y_prob.round(3)
        Y_pred = np.argmax(y_prob,axis = -1)  #axis = -1 , give you output for each array/row

        for img_array, pred, actual in zip(X_test, Y_pred, y_test):
            plt.imshow(img_array, cmap="binary")
            plt.title(f"predicted: {pred}, Actual: {actual}")
            plt.axis("off")
            plt.show()  