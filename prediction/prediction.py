import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


import os
import pickle

class dataTraining(object):
    def __init__(self):
        pass

    def linearReg():
        pass

    def logReg():
        pass

    def decisionTree():
        pass

    def randForest():
        pass

    def xgBoost():
        pass

    def timeSeries():
        pass