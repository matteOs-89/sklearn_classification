import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#models from scikit learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Model Evaluations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix

"""The aim of this project is to build a reliable ML model, that is capable of 
predicting whether a patient has a heart related disease, this can be described as a classification problem.
We will need to import and explore the dataset, clean it (if needed) and present it in a manner acceptable
for the chosen ML algorithm. """

df = pd.read_csv("/Users/eseosa/Downloads/heart-disease.csv")

df.head()  #Shows the first five rows of the dataset.
# attached is a text file that defining columns name (Data_dict).

"""Here we will explore the dataset, check for null features,
 fill or drop columns if needed and also explore the dataset by visualisation"""

print(df.isnull().sum()) # dataset has 0 null values

"""in this classification task there are a few important visual explorations needed, these include:
 
* Choosing what part of the dataset will be used as the dependent variable or independent feature. 
These explorations will help provide a better understanding of feature importance

* Explore the dependant variable for example, is it evenly spread? Do more patients have heart issues than others?

* The correlation between sex and disease i.e, is the heart disease more prominent in the male sex or otherwise
 
* The correlation between age and disease
  
* How different conditions stated correlate with potential heart issues (read 'Data_dict' if needed).
  
 """

df["target"].value_counts() #As this is what we are trying to predict, it will be  "y" (dependent variable)
#165 Yes %55
#138 No %45

plt.figure(figsize=(6, 4), dpi = 200)
sns.countplot(data=df, x="target")
plt.show()

df.sex.value_counts() # 1-MALE 0-FEMALE
# Male    207
# Female     96

plt.figure(figsize=(6, 4), dpi = 200)
sns.countplot(data=df, x="target", hue="sex");

plt.title("Heart Disease Frequency for sex")
plt.xlabel("0= No Disease, 1 = Disease")
plt.ylabel("Amount")

plt.legend(["Female", "Male"])
plt.show()


"""
As the graph shows above, it's interesting to note that although there are over double the
  amount of men than women in the dataset, the ratio in the positive heart disease results between male and female 
  is closer, in fact the difference is as little as 12.7%. This tells me that the probability of a women
  being diagnosed with this condition is higher than that of men."""

"""Let's plot a heatmap to visualise the possibility of multicollinearity,
 and its relationship to the dependant"""

plt.figure(figsize=(10, 6), dpi= 200)
sns.heatmap(data=df.corr(), annot=True )
plt.show()

# Exang has strongest negative corrolation with targets,
#Negative correlation means as one decreases the other increases. here the decress of .

"""
exang - exercise induced angina (1 = yes; 0 = no) ...from Data_dict...

cp - chest pain type 
0: Typical angina: chest pain related decrease blood supply to the heart 
1: Atypical angina: chest pain not related to heart 
2: Non-anginal pain: typically esophageal spasms (non heart related) 
3: Asymptomatic: chest pain not showing signs of disease

As stated above exang is related to chest pains and there are various types 

"""

"LETS explore further into type of chest pains"


plt.figure(figsize=(6, 4), dpi = 200)
sns.countplot(data=df, x="target", hue="cp")

plt.xlabel("0= No Disease, 1 = Disease")
plt.ylabel("Amount")

plt.legend(["Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic"])
            


"""Non-anginal pain: typically esophageal spasms (non heart related) 
 has highest CP type correlation with all positive targets, this is interesting to me
 as a non heart related pain has the highest correlation, perhaps this is some 
 some one with a better domain understanding could explain."""

"""Age vs max heart rate for heart disease (Thalach)


#shatter with negative examples
sns.scatterplot(data=df.age[df.target==0],
                #df.thalach[df.target==0],
                color="blue"
                )


#shatter with positive example
sns.scatterplot(data=df.age[df.target==1],
                #df.thalach[df.target==1],
               color="red"
                )



plt.title("Heart Disease in relation to age and  Max Heart Rate (thalach)")
plt.ylabel("Max Heart Rate (thalach)")

plt.legend(["No Disease", "Disease"], loc="best")
plt.show()

"""

# There is no clear correlation between age, max heart rate and heart disease
"""however I should state that I noticed a potential outlier where the patient is below 30 years old. 
In this case, the patient had a very high heart rate and a heart disease. Although this was an isolated incident, 
it is possible that this could occur and as such, I have opted to retain the outlier"""


"""Split dataset into Independant(X) and Dependant (y)"""

X = df.drop("target", axis=1)

y = df["target"]

#spliting dataset for train and test

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=101)
#Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


"""TIME TO CHOOSE A MODEL"""

"""
1 GBOOST
2 KKN 
3 Logistic Regression 
4 RandomForrestClassifier 
5 SVC"""

"""First I will take the baseline scoring of these models to see how they perform, then further evaluate the models """

models = {
    "GBOOST": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC()
    }


def fit_and_score(model, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates provided machine learning models
    provided in a dictionary format.

    Return model scores in a dictionary format

    X_train: training data(no label)
    X_test: testing dat (no labels)
    y_train: training labels
    y_test: test labels
    """

    # setting random seed rto return same sample
    np.random.seed(101)

    # Creating a dictionary to keep model scores

    model_scores = {}

    # loop through models
    for name, model in model.items():
        # fit and evaluate models

        model.fit(X_train, y_train)

        model_scores[name] = model.score(X_test, y_test)

   #Return model scores
    return model_scores





model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)

"""Score shows that all models performed over the 80% mark """
print(model_scores)

"""Although the score accuracy is in it's high 80s in some models, 
with the highest coming at 86% accuracy (Logistic Regression) lets see if we can improve on this 
by tuning the hyperparameters of these models. This will use RandomizedearchCV"""

# creating a hyperparameter grid for Xgboost

x_grid = {'n_estimators': np.arange(10,1000,50),
          "min_samples_split": np.arange(2,20,2),
          "learning_rate": [0.1,0.2,0.3,0.4],
          "max_depth":[None, 3, 5, 6, 10]
          }
# creating a hyperparameter grid for K Neighbour

k_grid = {"n_neighbors": [1,2,3,4,5,6,7,8,9]}

# creating a hyperparameter grid for logistic regression

log_reg_grid = {"C": np.logspace(-4,4, 20),
               "solver": ["liblinear"]}

# creating a hyperparameter grid for Random forest

rf_grid = {"n_estimators": np.arange(10,1000,50),
          "max_depth":[None, 3, 5, 10],
          "min_samples_split": np.arange(2,20,2),
          "min_samples_leaf": np.arange(1,20,2)}

# creating a hyperparameter grid for SVC
svc_grid = {'C': [1, 10, 100, 1000],
            'kernel': ['linear','rbf','poly'],
            "degree":[1, 2, 3,4]}


"""randomized Hyperparameter CVsearch for each models, with fitting and scorings to cross check"""

xgb_Rgrid = RandomizedSearchCV(GradientBoostingClassifier(random_state=101),
                                param_distributions= x_grid,
                                cv=5,
                               n_iter=30,
                             random_state=101)

xgb_Rgrid.fit(X_train,y_train)

xgbY_score = xgb_Rgrid.score(X_test, y_test)




knn_Rgrid = RandomizedSearchCV(KNeighborsClassifier(),
                                param_distributions= k_grid,
                                cv=5,
                               n_iter=30,
                              random_state=101)

knn_Rgrid.fit(X_train,y_train)
knnY_score = knn_Rgrid.score(X_test, y_test)

lr_Rgrid = RandomizedSearchCV(LogisticRegression(random_state=101),
                                param_distributions= log_reg_grid,
                                cv=5,
                               n_iter=30,
                             random_state=101)

lr_Rgrid.fit(X_train,y_train)
lrY_score = lr_Rgrid.score(X_test, y_test)


rf_Rgrid = RandomizedSearchCV(RandomForestClassifier(random_state=101),
                                param_distributions=rf_grid,
                                cv=5,
                               n_iter=30,
                             random_state=101)

rf_Rgrid.fit(X_train,y_train)
rfY_score = rf_Rgrid.score(X_test, y_test)


svc_Rgrid = RandomizedSearchCV(SVC(random_state=101),
                                param_distributions= svc_grid,
                                cv=5,
                               n_iter=30,
                             random_state=101)

svc_Rgrid.fit(X_train,y_train)
svcY_score = svc_Rgrid.score(X_test, y_test)


randomCV_score = {
    "GBOOST": xgbY_score,
    "KNN": knnY_score,
    "Logistic Regression": lrY_score,
    "Random Forest": rfY_score,
    "SVC": svcY_score
    }

print(randomCV_score)

""""CONFUSION MATRIX and CLASSIFICATION REPORT"""

"""I will be comparing the two highest scorers (KNN and Random Forrest) from the randomised section to decide which
 of the models is more suitable for this task, I will do this by taking readings for the following:
 
* ROC curve and AUC score
* Confusion Matrix
* Classification report
* Precision (True Positive vs False Positive)
* Recall (True Positive vs False Negative)
* F1-score
 """

k_pred = knn_Rgrid.predict(X_test)
#print(f"KNN_CLASS: {classification_report(y_test, k_pred)}")
plot_confusion_matrix(knn_Rgrid, X_test, y_test)
plot_roc_curve(knn_Rgrid, X_test, y_test)
plt.show()


r_pred = rf_Rgrid.predict(X_test)
#print(f"Random Forrest CLASS: {classification_report(y_test, r_pred)}")
plot_confusion_matrix(rf_Rgrid, X_test, y_test)
plot_roc_curve(rf_Rgrid, X_test, y_test)
plt.show()


""""
KNN_CLASS:               precision    recall  f1-score   support

                    0       0.90      0.84      0.87        31
                    1       0.84      0.90      0.87        30

             accuracy                           0.87        61
            macro avg       0.87      0.87      0.87        61
         weighted avg       0.87      0.87      0.87        61
         
         
KNN AUC(area under curve) = 0.90 (90%)

Random Forrest CLASS:               precision    recall  f1-score   support

                                 0       0.92      0.77      0.84        31
                                 1       0.80      0.93      0.86        30

                          accuracy                           0.85        61
                         macro avg       0.86      0.85      0.85        61
                      weighted avg       0.86      0.85      0.85        61
                      
Random Forrest AUC (area under curve) = 0.91 (91%)
                      
From the plot and above, it shows that although both scores are very similar, I would perhaps alt to move forward using 
Random Forest model as it gives a better recall scoring. My reason is that I would rather the model
performs better diagnosing patients with heart diseases with a risk of false diagnoses than to use a model that clears  
more patients with heart issues as normal.

"""

'''I SHOULD STATE THAT PERHAPS A BETTER SCORE COULD BE FOUND USING A MORE ROBUST CV
MODEL LIKE GRIDSEARCHCV, HOWEVER DUE TO COMPUTING TIME IT WILL NOT BE USED ON THIS OCCASION AND
 I HAVE OPTED TO USE ramdomizedsearch, with a possible less effective tradeoff t GRIDSEARCHCV.
'''


"-----FEATURE IMPORTANCE----"



#print(rf_Rgrid.best_params_)
"""{'n_estimators': 60, 'min_samples_split': 4,
 'min_samples_leaf': 9, 'max_depth': 10}"""

feat_rf = RandomForestClassifier(n_estimators=60,
                            min_samples_split=4,
                            min_samples_leaf=9,
                            max_depth=10)

feat_rf.fit(X_train, y_train)


print(feat_rf.feature_importances_)










