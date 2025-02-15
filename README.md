
# Multi-Layer Feature Fusion for High-Accuracy Solid Waste Classification using a Hybrid Deep Learning Model

This study utilizes **deep learning and computer vision techniques** to classify solid waste images. The goal is to automatically identify different waste types (plastic, paper, metal, glass, etc.) to facilitate recycling processes. The basic structure of the proposed model is given below.

![Garbage_Classification-ProposedModule drawio](https://github.com/user-attachments/assets/c2597307-2216-4817-86f6-1cb4f4c17367)


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

Additionally, the organized data and the weights of the proposed model can be accessed here.

-**Datasets and weights**: [Link](https://drive.google.com/drive/folders/1QRU2_Sj_AbgGYsIxt8XwHueI0VBaujsR?usp=sharing)
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


