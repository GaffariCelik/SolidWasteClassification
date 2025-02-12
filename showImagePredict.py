import numpy as np
from data_load import *
from extractedFeature import *
#from config import *
import pickle
import matplotlib.pyplot as plt


#X_train.shape,X_test.shape
                           
        
def show_imgPredict():
    parser = parse_flags()
    X_train, X_test, Y_train, Y_test=select_data()
    x_train_ftr, x_test_ftr=create_Vektor(X_train,X_test,parser.save_model,nb_class=parser.number_class)  
    
    x_train_ftr=x_train_ftr.reshape((x_train_ftr.shape[0],x_train_ftr.shape[1]))
    x_test_ftr=x_test_ftr.reshape((x_test_ftr.shape[0],x_test_ftr.shape[1]))
    
    model = pickle.load(open(parser.save_model_XGB, 'rb'))
    
    plt.figure(figsize=(20,80))
    for i in range(128):
        ax = plt.subplot(32, 4, i + 1)
        plt.imshow(X_test[i].astype("uint8"))
        preds = model.predict(tf.expand_dims(x_test_ftr[i], 0))
        plt.title("Actual: "+str(Y_test[i]))
        plt.ylabel("Predicted: "+str(preds),fontdict={'color':'red'})
    
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.gca().axes.xaxis.set_ticklabels([])


if __name__ == '__main__':
    show_imgPredict()
    