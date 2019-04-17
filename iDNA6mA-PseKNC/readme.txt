
model_cross_validation.py
Usage: python model_cross_validation.py positive_dataset negative_dataset
This script trains the iDNA6mA-PseKNC model.
5-fold cross-validation was used to evaluate the performance of the classifier.

model_training.py
Usage: python model_training.py training_positive_dataset training_negative_dataset model_file scale_file
This script was used to implement the iDNA6mA-PseKNC tool.
Outputs:
     1--a model file, iDNA6mA-PseKNC.pkl, which can be directly used for prediction.
     2--a normalized file, normalization.pkl, which can be used to normalized the input data.
     
model_indepedent_testing.py
Usage: python model_indepedent_testing.py test_positive_dataset test_negative_dataset model_file scale_file
Evaluate the performance of iDNA6mA-PseKNC on the indepedent testing dataset.
The model_file and scale_file generated in the training process must be involved.

feature_selection.py
#This script implements the feature encoding scheme and can be auto involved in other scripts.