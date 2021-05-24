# Naive-Bayes-Classifier
This project is for COMPSC 361 Assignment 3

## Introduction
The task is to implement an improved version of the Naive Bayes algorithm that is able to predict the domain - one of Archaea, Bacteria, Eukaryota or Virus - from the abstract of research papers about proteins taken from the MEDLINE database.

## code
The runing process, result, report are in the jupyter notebook and html file.
The python files contains two version of Naive Bayes Classifier:
1. Standard Naive Bayes Classifier
2. Complement Naive Bayes Classifier
There are fit and predict function in both classes.
The fit function takes the text and classes as input to train the classifier.
The predict function takes text and ids as input to predict the class for each instance and return the result -- \[(id, class), ...\].
