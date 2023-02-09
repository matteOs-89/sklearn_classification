
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
import spacy

nlp = spacy.load("en_core_web_lg")

"""

In this tutorial we will carry out classification task, in which we will train a model to detect spam texts from our text dataset using skearn framework.
We will carry out such task by reading the dataset with Pandas library and data visualization using Matplotlib before training a few models in order to select the best performing model.

Few EDA steps to be applied out includes converting the data labels type into integers, implement Word2Vec on the training data and splitting out dataset for training and validation set.
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



df["word2vec"] = df["message"].apply(lambda x: nlp(x).vector)


Xw_train, Xw_test, yw_train, yw_test = train_test_split(df.word2vec.values, y, test_size=0.2, 
                                                    random_state=101)


"""Convert training data from 1dim to 2dim, this allows the model to compute 
calculations. we will also implement MinMaxScaler to remove the negative values, as MultinomialNB does. not allow such values."""

ndim_train = np.stack(Xw_train)
ndim_test = np.stack(Xw_test)
print(ndim_test.shape)
print(ndim_train.shape)



scaler = MinMaxScaler()
scaled_ndim_train = scaler.fit_transform(ndim_train)
scaled_ndim_test = scaler.transform(ndim_test)

classes = ["ham", "spam"]

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
        

model = {"svc": LinearSVC(),
        "r_forrest": RandomForestClassifier(),
        "nbm": MultinomialNB()}



train_model(model, scaled_ndim_train,yw_train, scaled_ndim_test, yw_test)


y_pred = model["svc"].predict(scaled_ndim_test)


print(confusion_matrix(yw_test,y_pred))


print(classification_report(yw_test, y_pred))

