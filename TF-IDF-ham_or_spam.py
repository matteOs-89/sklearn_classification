import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

"""
In this tutorial we will carry out classification task, in which we will train a model to detect spam texts from our text dataset using skearn framework. 
We will carry out such task by reading the dataset with Pandas library and data visualization using Matplotlib before training a few models in order to select the best performing model.
Few EDA steps to be applied out includes converting the data labels type into integers, 
implement TF IDF Vectorization on the training data and splitting out dataset for training and validation set. 
These process helps the model to perform better.
"""

df = pd.read_csv("../NLP/UPDATED_NLP_COURSE/TextFiles/smsspamcollection.tsv", sep="\t")


df.head()


" EDA process to understand the data"
print(df.isnull().sum())
print(len(df))

print(df["label"].unique())


bins = 1.20**(np.arange(0,50))
plt.hist(df[df["label"] == "ham"]["length"], bins=bins)
plt.hist(df[df["label"] == "spam"]["length"], bins=bins)
plt.legend(["ham", "spam"])
plt.xscale("log")
plt.show()


bins = 1.20**(np.arange(0,50))
plt.hist(df[df["label"] == "ham"]["punct"], bins=bins)
plt.hist(df[df["label"] == "spam"]["punct"], bins=bins)
plt.legend(["ham", "spam"])
plt.xscale("log")
plt.show()


print(df["label"].value_counts())

# 4825 Ham 
#747 Spam

""" The Dataset is not balanced as there are far more ham labels than spam. Therefore we will not rely on accuracy score of the model. Other scoring metrics will
be used in order to have a better feel of how the model is truly performing. These metrics are Confusion Metrix and Classification Report."""

X = df["message"]
y = df["label"]

le = LabelEncoder()
y = le.fit_transform(y)

classes = ["ham", "spam"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=101)


tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

X_train_tfidf




model = {"svc": LinearSVC(),
        "r_forrest": RandomForestClassifier(),
        "nbm": MultinomialNB()}



def train_model(model, X_train, y_train, X_test, y_test):
    
    """
    This function take in models and both training and validation dataset, and fit and scores the model's performance.
    It gives us an efficient way to evaluate various model performance.
    """
    
    result = {}
    
    for key, values in model.items():
        
        model[key].fit(X_train, y_train)
        acc = model[key].score(X_test, y_test)
    
        result[key] = acc
    print(result)
        


train_model(model, X_train_tfidf,y_train, X_test_tfidf, y_test)


# Pipeline allows us to carry out all preprocessing and train steps together, this helps to improve inference time on unseen data.

classifier_pipe = Pipeline(steps=[("tfidf_vect", TfidfVectorizer()),("model", LinearSVC())])


classifier_pipe.fit(X_train,y_train)

prediction = classifier_pipe.predict(X_test)


print(confusion_matrix(y_test,prediction))


    [[976      4]
     [ 13    122]]
   
  

print(classification_report(y_test, prediction))


              precision    recall  f1-score   support

           0       0.99      1.00      0.99       980
           1       0.97      0.90      0.93       135

   micro avg       0.98      0.98      0.98      1115
   macro avg       0.98      0.95      0.96      1115
weighted avg       0.98      0.98      0.98      1115


# Testing 

print(classes[int(classifier_pipe.predict(["50% off"]))])

# Spam


print(classes[int(classifier_pipe.predict(["Its a happy day"]))])

# Ham


