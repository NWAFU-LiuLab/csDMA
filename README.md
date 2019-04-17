# csDMA
an improved machine-learning based prediction tool for identifying DNA 6mA modifications

classifier_selection.py

#This script is used to select the best classifier.

Usage: python classifier_selection.py train_positive_dataset train_negative_dataset matrix_motif1 matrix_motif2

This script trains five different 6mA prediction models.
Including:
     1--RandomForest
     2--GradientBoosting
     3--AdaBoost
     4--ExtraTrees
     5--SVM
     6--a ensemble classifier with the above five methods. 
5-fold cross-validation was used to evaluate the performance of different classifier.

csDMA_training.py
#This script generates csDMA model file and normalized model file.
Usage: python csDMA_training.py training_positive_dataset training_negative_dataset matrix_motif1 matrix_motif2 model_file scale_file
The csDMA trained with the ExtraTrees classifier.
Outputs:
     1--a model file, csDMA.pkl, which can be directly used for prediction.
     2--a normalized file, normalization.pkl, which can be used to normalized the input data.

csDMA_indepedent_testing.py
#This script evaluates the model performance on the indepedent testing dataset.
Usage: python csDMA_indepedent_testing.py test_positive_dataset test_negative_dataset matrix_motif1 matrix_motif2 model_file scale_file
Evaluate the performance of csDMA on the indepedent testing dataset.
The model_file and scale_file generated in the training process must be involved.

leaveoneout.py
#This script evaluates the model performance by using LeaveOneOut method.
#The csDMA trained with the ExtraTrees classifier.
Usage: python {0} training_positive_dataset training_negative_dataset matrix_motif1 matrix_motif2
Output: a LeaveOneOut score file will be generate.

feature_selection.py
#This script implements the feature encoding scheme and can be auto involved in other scripts.
