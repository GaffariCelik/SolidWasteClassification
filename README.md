
# Multi-Layer Feature Fusion for High-Accuracy Solid Waste Classification using a Hybrid Deep Learning Model

This study utilizes **deep learning and computer vision techniques** to classify solid waste images. The goal is to automatically identify different waste types (plastic, paper, metal, glass, etc.) to facilitate recycling processes. The basic structure of the proposed model is given below.



![Garbage_Classification-ProposedModule drawio](https://github.com/user-attachments/assets/c2597307-2216-4817-86f6-1cb4f4c17367)


# Results

** Results obtained based on test data on the TrashNet dataset:**

Method	Acc. (%)	Pre. (%)	Re. (%) 	F1. (%)	Spe. (%)	AUC (%)
MobileNetV2	70.36	61.07	62.01	60.32	93.96	83.58
DenseNet121	88.53	88.51	88.14	88.29	97.64	92.89
Xception	88.93	88.13	88.30	97.84	97.74	93.05
EfficientNetB0	89.72	88.12	90.40	89.04	97.93	94.10
InceptionV3	87.55	85.55	87.24	86.27	97.48	92.36
Proposed Model	99.40	99.49	99.54	99.51	99.88	99.71


![TrashNet](https://github.com/user-attachments/assets/e4ed8dd7-4aad-4926-b88c-8e030a9ab971)
Confusion matrix results of (a) the proposed model, (b) EfficientNetB0, and (c) InceptionV3 models

** Results obtained based on test data on the Household_Garbage dataset:**
Method	Acc. (%)	Pre. (%)	Re. (%) 	F1. (%)	Spe. (%)	AUC (%)
MobileNetV2	78.47	77.94	73.93	73.60	98.05	85.99
DenseNet121	91.27	88.29	88.09	87.77	99.22	93.66
Xception	92.23	89.59	90.02	89.29	99.30	94.66
EfficientNetB0	94.61	93.14	92.34	92.67	99.50	95.92
InceptionV3	88.14	87.46	82.84	84.44	98.88	90.86
Proposed Model	99.87	99.77	99.81	99.79	99.99	99.90

![HouseHold_Garbage](https://github.com/user-attachments/assets/b0e29985-2150-4b62-b002-ff3fa86fa004)
Confusion matrix results of (a) the proposed model, (b) EfficientNetB0, and (c) InceptionV3 models 


# Technologies Used  
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas, Matplotlib  

# Dataset  
This study uses the following datasets containing various types of solid waste:  

- **Household_Garbage Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)  
- **TrashNet Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data)  

The images are categorized and labeled accordingly, and the model is trained on this data.  

Additionally, the organized data and the weights of the proposed model can be accessed [here](https://drive.google.com/drive/folders/1QRU2_Sj_AbgGYsIxt8XwHueI0VBaujsR?usp=sharing).


## Usage Steps  
1. **img2nparray.py** → Resizes images and converts them into `nparray` format.  
2. **config.py** → Sets parameters.  
3. **train.py** → Trains the model.  
4. **test_model.py** → Tests the trained model.  
5. **1_Our_Model_Training_Testing.ipynb** → Training, testing, and visualization processes can only be performed through this file.

## Citation  

If you use this code in your research, please cite:  
@article{
  author = {Gaffari Celik},
  title = {"Multi-Layer Feature Fusion for High-Accuracy Solid Waste Classification using a Hybrid Deep Learning Model"},
  journal = {The Visual Computer},
  year = {2025},
  note = {Submitted}
}


