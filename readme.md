# Privacy-Preserving Data Mining in Banking Applications
## Classification performance with k-Anonymity

### Mini Project 4 - Applied Machine Learning [COMP-551] - McGill - Winter 2017

- The jupyter notebook summarizes the second part of my participation in the 4th mini project. It presents the _classifiaction algorithms_ for evaluation of performance _before_ and _after_ applying anonymization to the data.
- K-Anonymity has been applied in the first part using freeware software, ARX which could be downloaded from [here](http://arx.deidentifier.org/downloads/), please refer to the [report](Report.pdf) and the [presentation](Project_4_updated) for further details on the anonymization operation
- Anononymized data could be found in csv format for diffenet values of k in [k_anonymity](data/secured_files/k_anonymity) folder.
- I am testing out 3 common machine learning algorithms, namely __Logistic regression__, __Naive Bayes__ and __Random Forest__, for data classification before and after applying data privacy technique k-Anonymity. For this purpose, the dataset is divided into train and test sets. _Stratified sampling_ is used as our target value is unbalanced. In each set, I am maintain the ratio of zeros over ones, the same ratio as it is in the full dataset (About 3/4). Accuracy (ACC), area-under-curve (AUC), Precision (PRE) and Recall (REC) are used as performance metrics.
