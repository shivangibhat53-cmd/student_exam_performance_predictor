import os 
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)

    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        model_keys = list(models.keys())
        
        for name in model_keys:
            model = models[name]
            para = param.get(name, {})

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Extract the actual, fully-trained model object
            models[name] = gs.best_estimator_

            # Generate predictions using the verified trained model
            y_train_pred = models[name].predict(X_train)
            y_test_pred = models[name].predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            #print(test_model_score)
            # Save the score and the working model object
            report[name] = test_model_score
            
        # Return both dictionaries back to the component file
        return report


    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

