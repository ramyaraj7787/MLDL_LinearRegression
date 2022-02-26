import pandas as pd
import numpy as np
from application_logging import logger
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self):
        self.file_object = open("logs/preprocessing.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def remove_correlated(self, dataset, threshold):
        self.logger_object.log(self.file_object, 'Entered the remove_correlated method of the Preprocessor class')
        self.dataset=dataset
        self.threshold=threshold
        try: 
            self.col_corr = set()
            self.corr_matrix = self.dataset.corr()
            for i in range(len(self.corr_matrix.columns)):
                for j in range(i):
                    if (abs(self.corr_matrix.iloc[i,j])) > self.threshold:
                        self.colname = self.corr_matrix.columns[i]
                        self.col_corr.add(self.colname)
            self.dataset_updated = self.dataset.drop(self.col_corr, axis=1)   
            return self.dataset_updated
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in feature selection using correlation:  '+str(e))
            raise Exception()
    
    def separate_label_feature(self, data, label_column_name):
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    