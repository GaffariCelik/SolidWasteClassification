import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *
parser = parse_flags()


#model = MyModel(input_shape = (128, 128, 1), nb_class = 4, depth = 5)

def data_nparray_load():
    #data = np.load(f'Data/Data_224x224_garbage_12Class.npz')    
    data = np.load(parser.data_dir) #
    x_train_original = data['x_train']
    x_test_original = data['x_test']
    Y_train = data['y_train']
    Y_test = data['y_test']
    x_train = x_train_original #.astype(dtype='float') / 255.0
    x_test = x_test_original #.astype(dtype='float') / 255.0
    
    X_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
    X_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
    
    testX_array=X_test
    testY_array=Y_test
    
    X_train_hepsi=np.concatenate((X_train, testX_array ))
    Y_train_hepsi=np.concatenate((Y_train, testY_array ))
    X_train_hepsi.shape,Y_train_hepsi.shape

    return X_train_hepsi,Y_train_hepsi
    

def select_data():
  X_train_hepsi, Y_train_hepsi=data_nparray_load()
  X_train, X_test, y_train, y_test = train_test_split(X_train_hepsi, Y_train_hepsi, test_size = 0.20, random_state = 32) #28 #42 28 #32
  # Feature Scaling
  #sc = StandardScaler()
  #X_train = sc.fit_transform(X_train)
  #X_test = sc.fit_transform(X_test)
  return X_train, X_test, y_train, y_test

# 0=4861 1=249

