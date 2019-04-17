# csDMA
an improved machine-learning based prediction tool for identifying DNA 6mA modifications<br>
csDMA was implemented in python 2.7

classifier_selection.py<br>
#This script is used to select the best classifier.<br>
Usage: python classifier_selection.py train_positive_dataset train_negative_dataset matrix_motif1 matrix_motif2<br>
This script trains five different 6mA prediction models.<br>
Including:<br>
     1--RandomForest<br>
     2--GradientBoosting<br>
     3--AdaBoost<br>
     4--ExtraTrees<br>
     5--SVM<br>
     6--a ensemble classifier with the above five methods. <br>
5-fold cross-validation was used to evaluate the performance of different classifier.<br>

csDMA_training.py<br>
#This script generates csDMA model file and normalized model file.<br>
Usage: python csDMA_training.py training_positive_dataset training_negative_dataset matrix_motif1 matrix_motif2 model_file scale_file<br>
The csDMA trained with the ExtraTrees classifier.<br>
Outputs:<br>
     1--a model file, csDMA.pkl, which can be directly used for prediction.<br>
     2--a normalized file, normalization.pkl, which can be used to normalized the input data.<br>

csDMA_indepedent_testing.py<br>
#This script evaluates the model performance on the indepedent testing dataset.<br>
Usage: python csDMA_indepedent_testing.py test_positive_dataset test_negative_dataset matrix_motif1 matrix_motif2 model_file scale_file<br>
Evaluate the performance of csDMA on the indepedent testing dataset.<br>
The model_file and scale_file generated in the training process must be involved.<br>

leaveoneout.py<br>
#This script evaluates the model performance by using LeaveOneOut method.<br>
#The csDMA trained with the ExtraTrees classifier.<br>
Usage: python leaveoneout.py training_positive_dataset training_negative_dataset matrix_motif1 matrix_motif2<br>
Output: a LeaveOneOut score file will be generate.<br>

feature_selection.py<br>
#This script implements the feature encoding scheme and can be auto involved in other scripts.<br>
