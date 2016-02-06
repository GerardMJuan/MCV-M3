# Master in Computer Vision - M3 Machine learning for computer vision
Team :

* Sergio Sancho
* Gerard Martí
* Eric López
* Adriana Fernández

# Description of the project

The goal of this project is to learn the basic concepts and techniques to build a trained classifier to recognize specific objects. In this project we focus on Traffic Signs Detection and Recognition (TSDR) in images recorded by an on-board vehicle camera. This project is framed in the field of the computer-aided driver assistance, along with obstacle detection, pedestrian detection, parking assistance or lane departure warning, as well as a range of non-visual components like GPS-based vehicle positioning or intelligent route planning. For these reasons, TSDR represents a typical problem where machine learning can be successfully applied to obtain accurate automatic results in a real-world problem. 

The learning objectives for the students are the use of local image descriptors, such as Histogram of Oriented Gradients (HOG), Haar-like features, and basic binary machine learning methods such as Support Vector Machine (SVM), Adaptive Boosting (AdaBoost), ensemble methods and techniques to design multiple-class classifiers. In this way, the students can experience with the problems of evaluating the performance and cross-validation techniques.

This project was done in the Master in Computer Vision - UAB, for Module 3 - M3 Machine learning for computer vision


# Installing
Para utilizar este segundo sistema, se deben instalar las librerías xgboost y PyWavelets.

Xgboost: Navegar en la carpeta libs, xgboost y ejecutar el siguiente comando: ‘python setup.py install’.

PyWavelets: se debe ejecutar el siguiente comando: ‘pip install PyWavelets’


For the matlab code to work properly:
=========================================================
Need to have a working MATLAB installation in the computer.

To install the MATLAB Engine for Python, execute the following commands where "matlabroot" is the path to the MATLAB folder.

Windows® system:

cd "matlabroot\extern\engines\python"
python setup.py install
Mac or Linux® system:

cd "matlabroot/extern/engines/python"
python setup.py install
