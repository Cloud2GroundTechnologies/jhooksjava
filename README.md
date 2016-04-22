Machine Learning Algorithms using Spark: Naive Bayes
====================================================


Naive Bayes Solution
====================
This package shows how to use Spark's NaiveBayesModel for predicting diabetes.

This package has 4 classes:

1. ````BuildDiabetesModel````: builds a model from a client test data (includes 
   features and classification for every data point)

 
2. ````PredictDiabetes````: use the built model and new query data to predict diabetes


3. ````TestAccuracyOfModel````: use the built model and test data (which we know the 
   classifications) to find the accuracy of the model


4. ````Util````: some common methods used in these example classes


Build Model
===========
The class ````BuildDiabetesModel```` reads the client test data 
and builds a ````NaiveBayesModel```` and saves 
it in HDFS.

