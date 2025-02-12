from proposedModel_HyperColumn import *
from metrics_visualization import *
from train_XGBmodel import *
import numpy as np
from data_load import *
from extractedFeature import *
from tensorflow.keras.callbacks import ModelCheckpoint
from config import *



#X_train.shape,X_test.shape
                           
        
def training(): 
    parser = parse_flags()
    X_train, X_test, Y_train, Y_test=select_data()
    model=Model_HyperColumn(parser.number_class)  
    # save only weight
    checkpoint = ModelCheckpoint(parser.save_model, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1, save_weights_only=True)
    # save model
    # checkpoint = ModelCheckpoint('checkpoint_6Class/deneme', monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
    callbacks=[checkpoint]
    
    model.fit([X_train,X_train],Y_train, 
                    epochs=parser.epochs,
                    validation_data=([X_test,X_test],Y_test),
                    batch_size=parser.batch_size,verbose=1,callbacks=callbacks)

    #model.save(parser.save_model)#)
    x_train_ftr, x_test_ftr=create_Vektor(X_train,X_test,parser.save_model,nb_class=parser.number_class)  
    
    x_train_ftr=x_train_ftr.reshape((x_train_ftr.shape[0],x_train_ftr.shape[1]))
    x_test_ftr=x_test_ftr.reshape((x_test_ftr.shape[0],x_test_ftr.shape[1]))
    
    train_XGBmodel(x_train_ftr, Y_train,x_test_ftr,Y_test,parser.save_model_XGB,number_class=parser.number_class)

if __name__ == '__main__':
    training()