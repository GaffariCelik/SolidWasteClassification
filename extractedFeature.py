
from sklearn.metrics import accuracy_score
from proposedModel_HyperColumn import *

from data_load import *

def load_CNNmodel(fname,nb_class=6):
  model = model=Model_HyperColumn(nb_class=nb_class)
  model.load_weights(fname)
  return model

def create_Vektor(X_train,X_test,fname,nb_class=6): 
  model=load_CNNmodel(fname,nb_class=nb_class)   
  vector_layers = ['dens3', 'dens4']
  model_for_vector = Model(
      inputs=model.input,
      outputs=model.get_layer('dens4').output
      #outputs=[model.get_layer(layer).output for layer in vector_layers]
  )
    
  X_train_ftr = model_for_vector.predict([X_train,X_train])
  X_train_ftr = np.array(X_train_ftr)
  X_test_ftr = model_for_vector.predict([X_test,X_test])
  X_test_ftr = np.array(X_test_ftr)
    
  X_train_ftr=X_train_ftr.reshape((X_train_ftr.shape[0],X_train_ftr.shape[1],1))
  X_test_ftr=X_test_ftr.reshape((X_test_ftr.shape[0],X_test_ftr.shape[1],1))
  return X_train_ftr, X_test_ftr

def load_future_Vektor(fname,nb_class=6):
  model=load_CNNmodel(fname,nb_class=nb_class)
  v_X_train, v_X_test=create_Vektor(model)
  v_X_train=v_X_train.reshape((v_X_train.shape[0],v_X_train.shape[1],1))
  v_X_test=v_X_test.reshape((v_X_test.shape[0],v_X_test.shape[1],1))
  return v_X_train, v_X_test