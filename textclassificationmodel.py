# -*- coding: utf-8 -*-

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class TextClassificationModel():
    
    def __init__(self, optimization_params={}, params={}, scoring="accuracy",
                 cross_validation=10):
        """
        Create the Classification Model Pipeline
        The model has thre main steps
            CountVectorizer, which creates the dictionary and convert the input text
            into ocurrences of the dictionary
            TfidTransformer, which convert the occurrences into frequencies of the
            word
            SGDClassifier, which makes the proper classification
        Initial params can be passed as a dictionary
        """
        
        self.best_params = params
        self.optimization_params = optimization_params
        self.scoring = scoring
        self.best_score = 0.0
        
        self.text_clf = Pipeline([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', SGDClassifier())
                                  ])
        
        self.update_model()
        
        
    def get_best_score(self):
        return self.best_score
    
    def get_best_parameters(self):
        return self.best_parameters


    def update_model(self):
        """
        Update the model given a dict of params
        """
        
        self.text_clf.set_params(**self.best_params)
        
    
    def train_model(self, data, target, grid_search=True):
        """
        Obtain best params by parametrization and train the model
        """
        
        if grid_search:
            
            print("Choosing best parameters")
            grid_search = GridSearchCV(estimator = self.text_clf,
                                       param_grid = self.optimization_params,
                                       scoring = self.scoring,
                                       cv = self.cross_validation)
    
            grid_search = grid_search.fit(data, target)
            self.best_parameters = grid_search.best_params_
            self.best_score = grid_search.best_score_
            
            print("Updating the model")
            self.update_model()
            
        print("Training the model")
        self.text_clf.fit(data, target) 
        
        
    def make_prediction(self, data, probability=True):
        """
        Given a list of data, make a classification into a class or a set of
        probabilities of each class
        """
        
        if probability:
            return self.text_clf.predict_proba(data)
        else:
            return self.text_clf.predict(data)