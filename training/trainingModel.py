from sklearn.model_selection import train_test_split

from data_preprocessing import preprocessing
from application_logging import logger
from data_ingestion import data_loader
from best_model_finder import tuner
from file_operations import file_methods


class trainModel:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("logs/Training.txt", 'a+')

    def linearRegModel(self):
        self.log_writer.log(self.file_object, 'Start of Training Linear Regression')
        try:
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_boston_data()

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data= preprocessor.remove_correlated(data, threshold=0.8)

            X,Y=preprocessor.separate_label_feature(data,label_column_name='MEDV')

            X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.3, random_state=4)

            model_finder = tuner.Model_Finder(self.file_object,self.log_writer)

            lr_model, lr_score = model_finder.get_best_params_for_linear_regression(X_train, Y_train, X_test, Y_test)

            file_op = file_methods.File_Operation(self.file_object,self.log_writer)
            save_model=file_op.save_model(lr_model,'LinearRegression')

            if save_model=='success':
                # logging the successful Training
                self.log_writer.log(self.file_object, 'Successful End of Training Linear Regression Model')
                self.file_object.close()
            else: 
                raise Exception
        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training Linear Regression Model')
            self.file_object.close()
            raise Exception

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

