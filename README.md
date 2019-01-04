# Student Performance Prediction using ANFIS and ANN

This project runs a UI that gives a prediction of the student grade either by using ANN(Artificial Neural Network) or ANFIS(Adaptive Neuro Fuzzy Inference System) by taking in the user input.

## Getting Started

The code was written using Python3 for UI and ANN and Matlab for ANFIS.

### Prerequisites

Following are the packages you will need:
1. Numpy
2. Pandas
3. TensorFlow
4. Matlab engine
5. Tkinter
6. Neuro Fuzzy tool box in Matlab

### Installation
For the matlab engine - follow the steps in this link</br>
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

Numpy, Pandas and Tkinter
```
pip install numpy 
pip install pandas
pip install tkinter
```
Tensorflow - follow the steps in this link</br>
https://www.tensorflow.org/install/pip

## Training new models
### Training a new ANN
  1.	Please change the code line 146 in UICallNN.py , from do_train = 0 to do_train = 1
  2.	Run the GUI application again.  When you click “Predict by NN” button, a new model will be trained, saved, and used to    predict the result.

### Training a new ANFIS
  1.	Run the code subc1.m in MATLAB.
  2.	Save “fismat2” model from the workspace as ‘ANFIS1.fis’ in the current directory.
  3.	Run the GUI application again. When you click on “Predict by ANFIS”, a new model will be used to predict the values
  
## Code Directory
***acc.m, acc1.m, acc2.m*** – calculates accuracy of ANFIS+ Classification model for different data splits</br>
***subc1.m*** – training an ANFIS model  </br>
***mse.m*** – calculates the MSE for the ANFIS model</br>
***norm_data.m*** – does Min –Max Normalization</br>
***labels.m*** – assigns labels to the original data after normalizing</br>
***classmodel.m*** – contains the code for the classification model </br>
***ANFIS_predictmodel*** – takes the user input from the GUI and returns the predicted label</br>
***UICallNN.py*** – GUI and code for ANN</br>
***Saved_nn*** – folder where the trained NN is saved and is used later to predict labels 




