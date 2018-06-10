# -*- coding: utf-8 -*-

import numpy as np

def create_output(probabilities, text_ids, text_classes, output_file):
    """
    Generate a CSV with the expected format
    """
    
    # Convert the data into numpy arrays
    text_ids.insert(0, "document_id")
    ids_array = np.array(text_ids)
    classes_array = np.array(text_classes)
    prob_array = np.array(probabilities)
    
    # Let's join all data into a matrix
    joined_array = np.vstack([classes_array, prob_array])
    final_array = np.insert(joined_array, 0, ids_array, axis=1)
    
    # Write the data into a file
    final_array.tofile('sergio-vidieilla_submission.csv',sep=', ',format='%10.5f')