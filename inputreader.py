# -*- coding: utf-8 -*-

import os


class TextClassifierData():
    
    def __init__(self):
        self.data = []
        self.target = []
        self.id = []
        
    def add_data(self, data, target, idx):
        """Add a new entry to each list of the object"""
        
        self.data.append(data)
        self.target.append(target)
        self.id.append(idx)


def file_structure_to_dataset(input_path, classes_list):

    try:
        dataset = TextClassifierData()
    
        # Iterate over the folder and find all the folders inside it
        for name in os.listdir(input_path):
            folder_path = "{}{}".format(input_path, name)
            if os.path.isdir(folder_path):
                # Folders within class folders contain the data of interest
                if name in classes_list:
                    target = classes_list.index(name)
                    for filename in os.listdir(folder_path):
                        file_path = "{}{}{}".format(folder_path, os.sep, 
                                     filename)
                        with open(file_path, 'r') as myfile:
                            # Save the data, and identifier of the class 
                            # (target) and the name of the file (id)
                            dataset.add_data(myfile.read(), target, filename)
        
        return dataset
    
    except Exception as e:
        print(e)


def get_text_classes(input_folder):
    """
    Get the folder names as classes. Their position in the list will be their
    identifier for classification
    """
    
    text_classes = [name for name in os.listdir(input_folder) if \
                    os.path.isdir("{}{}".format(input_folder, name))]
    
    return text_classes


def prepare_input():
    """
    Read the input train and test data andreturn the dictionaries
    """
    
    # The paths are fixed in this example
    # We could include them in a config path
    data_path = "{}{}20news-bydate{}".format(os.getcwd(), os.sep, os.sep)
    train_path = "{}20news-bydate-train{}".format(data_path, os.sep)
    test_path = "{}20news-bydate-test{}".format(data_path, os.sep)
    
    text_classes = get_text_classes(train_path)
    
    train_dataset = file_structure_to_dataset(train_path, text_classes)
    test_dataset  = file_structure_to_dataset(test_path, text_classes)

    return train_dataset, test_dataset, text_classes