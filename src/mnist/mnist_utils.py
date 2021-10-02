import tensorflow as tf

def prepareData():
  """
    * Prepare the mnist data to be used in ANN model
    * Data normalized to lie between 0-1

    Return:
            X_train ([numpy array]): [Training input data]
            y_train ([numpy array]): [Training output data]
            X_valid ([numpy array]): [Validation input data]
            y_valid ([numpy array]): [Validation output data]
            X_test ([numpy array]): [Test input data]
            y_test ([numpy array]): [Test output data]
  """
  mnist = tf.keras.datasets.mnist
  (X_train_full,y_train_full),(X_test,y_test) = mnist.load_data()
  X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.   #normalizing and redusing the pixel value to be between 0-1
  y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
  X_test = X_test / 255.
  return (X_train, y_train),(X_valid, y_valid),(X_test,y_test)











