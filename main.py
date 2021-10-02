from src.ANNModel import ann_model
from src.mnist import mnist_utils

if __name__ == '__main__': ##entry point
    #get data
    train_data, valid_data, test_data = mnist_utils.prepareData()
    
    #create, train and save ANN Model for mnist data set
    model = ann_model.ANN()    
    model.trainModel(trainingData=train_data, EPOCHS=3, validationData=valid_data )
    model.evaluateModel(test_data)
    model.plotModel((test_data[0][:3],test_data[1][:3]))
    model.saveModel("mnist_model.h5")