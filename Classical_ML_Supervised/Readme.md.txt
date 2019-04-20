## Classical Machine Learning Algorithms

# Overview
This package contains file to peroform PCA, LDA, Bayes', K-NN, Kernel SVM and boosted SVM.
The files contains the following functions:
1) PCA_fun.m 
2) LDA_fun.m 
3) KNN_fun.m
4) Bayes_Classifier_fun.m

Note: Kernel SVM and boosted SVMs are in the form of scipts.

# Dependencies
- MATLAB
- CVX (convex optimization soler) for boosted SVM.
 You can download CVX for free using this link: http://cvxr.com/cvx/doc/install.html

# Runnig the code:
In the files folder there are 14 file names of the form:

DimReductionTechnique_ClassifierName_ClassificationTask.m

Where:
- DimReductionTechnique is either PCA or LDA
- ClassifierName is Bayes, K-NN, RBF_Kernel_SVM, Poly_Kernel_SVM or Boosted SVM
- Classification task is either person classification or Expression Classification 

You can simply open each of these MATLAB scripts and run them.
It will call all the required functions and output the results.

Note: Make sure all the dependencies are satisfied before running the code (especially Bossted SVM).

