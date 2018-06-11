# -*- coding: utf-8 -*-

import numpy as np

from inputreader import prepare_input
from outputwriter import create_output
from textclassificationmodel import TextClassificationModel
        

if __name__ == '__main__':
    
    # Obtain the input
    train_dataset, test_dataset, text_classes = prepare_input()
    optimization_params = {'vect__stop_words': ['english'],
                           'vect__strip_accents': ['unicode'],
                           'tfidf__norm': ['l1', 'l2'],
                           'clf__loss': ['modified_huber','log'],
                           'clf__alpha': [1e-3, 1e-4]
                          }
    text_clf = TextClassificationModel(optimization_params=optimization_params)
    
    # Train the model
    text_clf.train_model(train_dataset.data, train_dataset.target)
    
    # Check the accuracy of the model with test data
    prediction = text_clf.make_prediction(test_dataset.data, probability=False)
    score = np.mean(prediction == test_dataset.target) 
    
    print("Train accuracy: {}".format(text_clf.get_best_score()))
    print("Test accuracy: {}".format(score))
    
    # Now create the expected ouput
    prob_prediction = text_clf.make_prediction(test_dataset.data)
    create_output(prob_prediction, test_dataset.id, text_classes)

