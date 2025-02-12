import numpy as np
from metrics_visualization import *
from data_load import *
from extractedFeature import *
#from config import *
import pickle


#X_train.shape,X_test.shape
                           
        
def model_testing():
    parser = parse_flags()
    X_train, X_test, Y_train, Y_test=select_data()
    x_train_ftr, x_test_ftr=create_Vektor(X_train,X_test,parser.save_model,nb_class=parser.number_class)  
    
    x_train_ftr=x_train_ftr.reshape((x_train_ftr.shape[0],x_train_ftr.shape[1]))
    x_test_ftr=x_test_ftr.reshape((x_test_ftr.shape[0],x_test_ftr.shape[1]))
    
    model = pickle.load(open(parser.save_model_XGB, 'rb'))
    pred = model.predict(x_test_ftr)
    MetricsPlot(Y_test, pred, parser.number_class)


if __name__ == '__main__':
    model_testing()
