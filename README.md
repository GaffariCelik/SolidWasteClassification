
# Multi-Layer Feature Fusion for High-Accuracy Solid Waste Classification using a Hybrid Deep Learning Model

This study utilizes **deep learning and computer vision techniques** to classify solid waste images. The goal is to automatically identify different waste types (plastic, paper, metal, glass, etc.) to facilitate recycling processes. The basic structure of the proposed model is given below.



![Garbage_Classification-ProposedModule drawio](https://github.com/user-attachments/assets/c2597307-2216-4817-86f6-1cb4f4c17367)


# Experimental Results

- **Results obtained based on test data on the TrashNet dataset:**

![TrashNet_tablo](https://github.com/user-attachments/assets/d4d9048e-a2f2-4d65-a06b-6b0b528b9dd3)


 

![TrashNet](https://github.com/user-attachments/assets/e4ed8dd7-4aad-4926-b88c-8e030a9ab971)
     Confusion matrix results of (a) the proposed model, (b) EfficientNetB0, and (c) InceptionV3 models. 
      0: cardboard, 1: glass, 2: metal, 3: paper, 4: plastic, and 5: trash

- **Results obtained based on test data on the Household_Garbage dataset:**
  
![HouseHold_Garbage_Tablo](https://github.com/user-attachments/assets/e20b0a89-27b8-4994-a553-cdf223263332)

![HouseHold_Garbage](https://github.com/user-attachments/assets/b0e29985-2150-4b62-b002-ff3fa86fa004)
   
     Confusion matrix results of (a) the proposed model, (b) EfficientNetB0, and (c) InceptionV3 models. 
     0: battery, 1: biological, 2: brown-glass, 3: cardboard, 4: clothes, 5Ç green-glass, 6: metal, 
     7: paper, 8: plastic, 9: shoes, 10: trash, and 11: white-glass


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


