from xgboost import XGBClassifier
from metrics_visualization import *
import pickle

def train_XGBmodel(x, y,xtest,ytest,path_XGB,number_class=6):   
    #path_XGB=checkpoint_fname+"best_model_XGB.sav"
    XGB=XGBClassifier(random_state=101)      
    XGB.fit(x, y)
    pred =XGB.predict(xtest)    
    # save the model to disk        
    pickle.dump(XGB, open(path_XGB, 'wb'))
    MetricsPlot(ytest,pred,number_class)  