import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from imblearn.metrics import geometric_mean_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.metrics import specificity_score

def plot_actual_vs_predicted(y_true,y_pred,title, number_class=6):
    cm = confusion_matrix(y_true,y_pred)
    cm=cm.astype(np.double)
    #cm=(np.round(cm / cm.sum(axis=1),2))#*100
    #cm=cm/np.sum(cm)

    plt.figure(figsize=(5,5))
    
    if (number_class==6):
        index=['0','1', '2', '3','4','5']
    else:
        index=['0','1', '2', '3','4','5','6','7','8','9','10','11']
    
    sns.heatmap(cm,annot=True,fmt='g',xticklabels=index,yticklabels=index)

    plt.title(title)
    plt.show()
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=4))



def Metric_auc(y_true,y_pred):

    ## Metric Accuracy
    #One-hot encoder
    y_valid=y_true.reshape(-1,1)
    ypred=y_pred.reshape(-1,1)
    y_valid = pd.DataFrame(y_valid)
    ypred=pd.DataFrame(ypred)

    onehotencoder = OneHotEncoder()
    y_valid= onehotencoder.fit_transform(y_valid).toarray()
    ypred = onehotencoder.fit_transform(ypred).toarray()
    metrics_auc=roc_auc_score(y_valid,ypred,multi_class='ovr')#model.predict(x_test)
    return metrics_auc


def Rocc_Curve(y_true,y_pred,metrics_auc,number_class=6):
   #One-hot encoder
    y_valid=y_true.reshape(-1,1)
    ypred=y_pred.reshape(-1,1)
    y_valid = pd.DataFrame(y_valid)
    ypred=pd.DataFrame(ypred)

    onehotencoder = OneHotEncoder()
    y_valid= onehotencoder.fit_transform(y_valid).toarray()
    ypred = onehotencoder.fit_transform(ypred).toarray()

    n_classes = ypred.shape[1]

    # Plotting and estimation of FPR, TPR
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if (number_class==6):
        index=['0','1', '2', '3','4','5']
    else:
        index=['0','1', '2', '3','4','5','6','7','8','9','10','11']
        

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], ypred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'green', 'red','darkorange','purple','navy','purple'])
    for i, color in zip(range(n_classes), colors):
        pyplot.plot(fpr[i], tpr[i], color=color, lw=1.5, label='{0}(RocCurveArea= {1:0.4f})' ''.format(index[i], roc_auc[i]))

    pyplot.plot([0, 0.95],[0, 0.05], color='navy', lw=1.5, label=" AUC ={:0.2f}%".format(100. *metrics_auc) )
    pyplot.plot([0, 1], [0, 1], 'k--', lw=1.5)
    pyplot.xlim([-0.05, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate',fontsize=10, fontweight='bold')
    pyplot.ylabel('True Positive Rate',fontsize=10, fontweight='bold')
    pyplot.tick_params(labelsize=12)
    pyplot.legend(loc="lower right")
    #ax = pyplot.axes()
    pyplot.show()

def Metric_Sensivity(y_true,y_pred):
    res = []
    for l in [0,1]:
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l,
                                                          np.array(y_pred)==l,
                                                          pos_label=True,average=None)
        res.append([l,recall[0],recall[1]])

    df1=pd.DataFrame(res,columns = ['class','sensitivity','specificity'])
    df1.describe()

    sensitivity,specificity=df1['sensitivity'],df1['specificity']
    Gmean=geometric_mean_score(y_true, y_pred)
    print('G-Mean={:0.4f}'.format(Gmean))

    #print('Sensivity={:0.4f}'.format(sensitivity.mean()))
    #print('specificity={:0.4f}'.format(specificity.mean()))

def metric_precision_score(y_true,y_pred):
  metric_pre=precision_score(y_true, y_pred, average='macro')
  #precision_score(y_true, y_pred, average='micro')
  #precision_score(y_true, y_pred, average='weighted')
  print("Metric_Precision={:0.4f}".format(metric_pre))
def metric_recall_score(y_true,y_pred):
  metric_recall=recall_score(y_true, y_pred, average='macro')
  print("Metric_Recall={:0.4f}".format(metric_recall))

def metric_f1_score(y_true,y_pred):
  metric_f1_score=f1_score(y_true, y_pred, average='macro')
  #f1_score(y_true, y_pred, average='micro')
  #f1_score(y_true, y_pred, average='weighted')
  print("Metric_f1_score={:0.4f}".format(metric_f1_score))

def metric_specificity(y_true,y_pred):

  specificity = specificity_score(y_true, y_pred, average='macro')
  print("Metric_Specificity_Macro={:0.4f}".format(specificity))
  #specificity = specificity_score(y_true, y_pred, average='micro')
  #specificity = specificity_score(y_true, y_pred, average='weighted')
def metric_each_Class(y_true,y_pred):

  mcm = multilabel_confusion_matrix(y_true, y_pred)
  tn = mcm[:, 0, 0]
  tp = mcm[:, 1, 1]
  fn = mcm[:, 1, 0]
  fp = mcm[:, 0, 1]

  #recall=np.round(tp / (tp + fn),4)
  specificity=np.round(tn / (tn + fp),4)
  '''
  acc=[np.mean([
        (y_true[pred_idx] == np.round(y_preds)) for pred_idx, y_preds in enumerate(y_pred)
      if y_true[pred_idx] == int(class_label)
                    ]) for class_label in np.unique(y_true)]
  '''
  class_accuracies = []
  for class_ in np.unique(y_true):
      class_acc = np.mean(y_pred[y_true == class_] == class_)
      class_accuracies.append(class_acc)
  print("Specificity for each class=  ", specificity, "Avg Specificity=", np.sum(specificity/len(np.unique(y_true))))
  print("Accuracy for each class=  ", np.round(class_accuracies,4),"Avg Accuracy=", np.sum(np.round(class_accuracies,4)/len(np.unique(y_true))))

def MetricsPlot(v_y_test,pred,number_class):
    acc=accuracy_score(v_y_test, pred)
    print("Accuracy:",acc)
    metric_precision_score(v_y_test,pred)
    metric_recall_score(v_y_test,pred)
    metric_f1_score(v_y_test,pred)
    metric_specificity(v_y_test,pred)
    Metric_Sensivity(v_y_test,pred)
    metrics_auc=Metric_auc(v_y_test,pred)
    print('Metric AUC={:0.4f}'.format(metrics_auc))
    metric_each_Class(v_y_test,pred)
    plot_actual_vs_predicted(v_y_test,pred,"Test Data Predictions",number_class=number_class)
    Rocc_Curve(v_y_test,pred,metrics_auc,number_class=number_class)
