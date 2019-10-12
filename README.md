# Deep Learning: Imbalanced Dataset Sampling Methodology Comparision

## Project Summary
The goal of this project was to compare different data sampling methodologies on an imbalanced dataset to determine the technique that results in the highest f1 score and recall rate of a deep learning model.  The three sampling methodologies that were tested are undersampling, oversampling, and Synthetic Minority Oversampling (SMOTE).  I also ran tests on unsampled data as a control.  This project contains three notebooks located in the "notebooks" directory that contains all of the code used to preprocess the data, perform exploratory data analysis (EDA), and train and evalutate the deep learning models.  It also contains three final models located in the "models" directory as well as hyperparameter tuning results in the "tuning" directory.

## Dataset
For this project, I used the [credit card fraud detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) found on Kaggle.  This dataset is highly imbalanced where only 0.172% of the data is of the positive fraudulent class.  Thus, this dataset is a good fit for the purposes of this project.  The dataset includes 30 numerical features not including the target label and these features are the result of a PCA transformation.  28 of these features are unknown due to confidentiality and are labeled "v1", "v2" ... "v28".  The other two features are "time" and "amount".

## Notebooks
**preprocessing.ipynb**
This notebook contains all of the code used to preprocess the raw original dataset.  This notebooks first loads the data into a Pandas dataframe and renames all columns to lowercase.  It then performs missing value imputation, checking for NaN values and zero values.  Though no NaN values are found, 27 cases of fraudulent data with a zero amount transaction were discovered.  All data with zero amount transactions were removed such that the model would only classify non-zero amount transactions.  The notebook then contains code for checking the datatypes of the dataset which are all float64 except for the binary label which is of int64 datatype.  Duplicates are then dropped and the data is split into three groups: training data, validation data, and test data.  The test data is 5% of the original dataset while the validation data is 3% of the remaining data after the test data is split off.  The test data contains 25 fraudulent cases.  Features are then scaled using the Standard Scaler module of sklearn.preprocessing and the statistics of the training data is used to scale the testing data so that the same scaling is done for both training and testing data.  The notebook then performs undersampling, oversampling, and SMOTE on the training data only and the resulting data is written to file.

**eda.ipynb**
This notebook contains all the exploratory data analysis done in this project.  The preprocessed data is first loaded into four Pandas dataframes, one for each sampling methodology.  Simple univariate analysis is then performed on each dataframe to determine the count, mean, standard deviation, minimum, quarties, and maximum.  Skewness of each variable is then plotted in barplot for each dataset and histograms for the 4 most skewed variables are displayed for each sampling methodology.  Correlation heatmaps and barplots are then shown as well as side-by-side boxplots for features with the highest absolute value correlation.  Outlier counts and proportions are also computed using the IQR and plotted as bargraphs.

**model and tuning.ipynb**
This notebook contains the code for the deep learning model architecture as well as functions for hyperparameter tuning.  The model contains two sets of hidden layers where each set contains a dense relu layer followed by a dropout layer.  The model is then compiled and fitted.  The optimizer used is the Adam optimizer and binary crossentropy is used as the loss function.  Hyperparameters tuned are learning rate, dropout regularization, neuron count, and batch size with the number of epochs fixed at 20.  For each combination, models are trained 20 times (5 times for each sampling dataset).  The mean f1, std f1, mean recall, std recall, mean precision, std precision, mean accuracy, and std accuracy are then computed using the testing data and stored in tuning_results.csv under the tuning directory.  Barplots are graphed for learning rate, neuron count, dropout regularization, and batch size with respect to the mean f1.  Each of the sampling techniques are also plotted with respect to mean f1, mean recall, and mean precision.  The hyperparameters and sampling methodologies with the maximum f1, recall, and precision are found and three models are then saved: two with high f1 scores and one with a high recall score.

## Models
**f1_model_smote.h5**
This model is trained from SMOTE data using a learning rate of 0.001, a dropout rate of 0.1, a neuron count of 32 per layer, and a batch size of 32.  It performed at a 0.82 f1 score on the testing data.

**f1_model_unsampled.h5**
This model is trained from unsampled data using a learning rate of 0.001, a dropout rate of 0.1, a neuron count of 32 per layer, and a batch size of 32.  It performed at a 0.85 f1 score on the testing data.

**recall_model_undersampled.h5**
This model is train from undersampled data using a learning rate of 0.001, a dropout rate of 0.0, a neuron count of 32 per layer, and a batch size of 32.  It performed at a 1.00 recall score on the testing data.

## Results
**Unsampling**
The unsampled data performed surprisingly well, having the highest precision score out of all 4 datasets.  It was also the dataset that produced the model with the highest f1 score.  However, it also had the highest spread of all the models trained and the model with the highest average f1 being SMOTE.  It also underperformed with regards to recall score, having the lowest recall score out of the 4 datasets.  That being said, it still performed fairly well given the high imbalance of the raw data.

**Undersampling**
The undersampled data had an extremely high (over 0.99 for the best model) recall score but also a very low precision score, resulting in the lowest f1 score out of all 4 models.  It was able to identify almost all of the fraudulent transactions but was often identifying non-fraudulent transactions as fraudulent also.  This is likely caused by the loss of non-fraudulent data due to the random undersampling of the majority class.  Still, it is the best choice out of all four sampling methodologies for finding fraudulent transactions.

**Oversampling**
The oversampled data had a decent f1 score, recall score, and precision score.  However, the undersampled data outperformed oversampling with regards to recall score and the unsampled data outperformed oversampling with regards to f1 score.  It still, however had a much higher precision than undersampling and a significantly higher recall score than unsampling.

**SMOTE**
Like the oversampled data, the SMOTE data had a decent recall and precision score.  Its recall score is slightly lower than oversampling but has a higher precision score.  As mentioned before, it also has the highest average f1 score, which still makes it a good candidate when choosing sampling techniques.

## Improvements
Several improvements could have been made to improve the final model and results of this project.  First, the zero amount data should have been kept such that the model would also be able to identify fraudulent 0 amount transactions.  Additional improvements could be made with cross validation.  Specifically, the validation data (not the testing data) should also be scaled during feature normalization and should be used to evaluate the models during hyperparameter tuning instead of the testing set.  The testing set would be used exclusively for evaluating the three final models.
