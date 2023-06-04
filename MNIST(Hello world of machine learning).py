#!/usr/bin/env python
# coding: utf-8

# # MNIST

# import sys
# assert sys.version_info >= (3,5)
# 
# #Is this notebook running on Colab or Kaggle?
# IS_COLAB = "google.colab" in sys.modules
# IS_KAGGLE = "kaggle_secrets" in sys.modules
# 
# #Scikit-Learn >= 0.20 is required
# import sklearn
# assert sklearn.__version__ >= "0.20"
# 
# #Common imports
# import numpy as np
# import os
# 
# #To make this notebook output stable across runs
# np.random.seed(42)
# 
# #To plot pretty figures
# %matplotlib inline
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize = 14)
# mpl.rc('xtick', labelsize = 12)
# mpl.rc('ytick', labelsize = 12)
# 
# #Where to save the figures
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "classification"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok = True)
# 
# def save_fig(fig_id, tight_layout = True, fig_extension ="png", resolution = 300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi = resolution)

# In[4]:


#MNIST 
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1, as_frame = False)
mnist.keys()


# In[5]:


X, y = mnist["data"], mnist["target"]
X.shape


# In[6]:


y.shape


# In[7]:


28*28


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()


# In[18]:


y[0]


# In[19]:


y = y.astype(np.uint8)


# In[20]:


def plot_digit(data):
    image = data.reshape(28,28)
    plt.imshow(image, cmap = mpl.cm.binary,
              interpolation = "nearest")
    plt.axis("off")


# In[26]:


#Extra
def plot_digits(instances, images_per_row = 10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    #This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1
    
    #Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)
    
    #Reshape the array so it's organized as a grid containing 28x28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))
    
    
    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    #Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


# In[27]:


plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row =10)
save_fig("more_digits_plot")
plt.show()


# In[28]:


y[0]


# In[29]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# # Training a Binary Classifier

# In[30]:


#Training a Binary Classifier
y_train_5 = (y_train ==5) #True for all 5s, False for all other digits
y_test_5 = (y_test == 5)


# In[31]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter =1000, tol=1e-3, random_state =42)
sgd_clf.fit(X_train, y_train_5)


# In[32]:


sgd_clf.predict([some_digit]) #The classifier guesses that this image represents a 5(True).
#Looks like it guesses right in this particular case! Now, let's evaluate this model's performance.


# In[34]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv =3, scoring= "accuracy")


# # Peformance Measures

# # Measuring Accuracy Using Cross-Validation

# In[38]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits =3, shuffle = True, random_state =42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
#Has a cumulative score of about 93%


# In[39]:


#A dumb classifier to check the accuracy
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype = bool)


# In[40]:


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")
#The cumulative score is aproximately 90%
#This is because about 10% of all images are 5s, so this means if 
#you guess that an image isn't a 5, then you will be right about 90% of the time.


# In[41]:


#The two diffrent accuracies show that accuracy isn't generally the
#perferred performance measure for classifiers, especially when dealing with skewed datasets


# # Confusion Matrix 

# In[42]:


#Confusion Matrixes are used to check the number of times the classifier confused 5s and another number.
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv =3)


# In[43]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
#Each row represents an actual class, while each column represents a predicted class.
#53,892 represents all the non 5s, while 687 represents all wrongly classifed 5s(false positives).
#The second row considers the images of 5s(the positive class):
    #1,891 were wrongly classified as 5s. While 3,530 were correctly
#A positive class(The second row) will only have non-zero values.


# In[44]:


y_train_perfect_predictions = y_train_5 #pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)
#The first row represents the number of non 5s in the first row and all the false positives.
#The second row represents all the false postives and all the true positives. 


# In[45]:


#Precision equation is
    #*Precision = TP / (TP + FP)
    #TP is the number of true positives, and FP is the number of false positives
    
    #*Recall = TP / (TP + FN)
    #FN is of course the number of false negatives


# # Precision and Recall

# In[46]:


from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
#The precision deteckts aprroximately 84% of the 5s.


# In[47]:


recall_score(y_train_5, y_train_pred)
#The recall score deteckts approximately 65% of the 5s.


# In[48]:


#The cumulative score for precision and recall score is 74%.


# In[49]:


#The Harmonic mean(F1 score) is:
    #F1 = ( 2 / (1/precision) + (1/recall)) = 2 * ((precision * recall) / (precision + recall)) = TP / TP + ((FN + FP)/2)


# In[50]:


#To compute the F1 score
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# In[51]:


#Increasing precison reduces recall, and increasing recall reduces precision.


# In[52]:


y_scores = sgd_clf.decision_function([some_digit])
y_scores


# In[53]:


threshold = 0
y_some_digit_pred = (y_scores > threshold)


# In[54]:


y_some_digit_pred


# In[55]:


threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
#This confirms that raising the threshold decreas recall. 
#The image actually represents a 5, and the classifier detects it when 
#the threshold is 0, but it misses it when the threshold is increased to 8,000


# In[56]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3,
                             method = "decision_function")


# In[58]:


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[66]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label = "Recall")
    plt.legend(loc= "center right", fontsize = 16) 
    plt.xlabel("Threshold", fontsize =16)
    plt.grid(True)
    plt.axis([-50000,50000,0,1])
    #highlight the threshold, add the legend, axis label and grid

recall_90_precision = recalls[np.argmax(precisions >= 0.90)]    
threshold_90_precision = thresholds[np.argmax(precisions >= 90)] 

plt.figure(figsize = (8,4))
plot_precision_recall_vs_threshold(precisions,recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0.,0.9], "r:")
plt.plot([-50000, threshold_90_precision], [0.9,0.9], "r:")
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9],"ro")
plt.plot([threshold_90_precision], [recall_90_precision], "ro")
plt.show()


# In[73]:


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
save_fig("precision_vs_recall_plot")
plt.show()


# In[67]:


threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


# In[74]:


threshold_90_precision


# In[68]:


#To make the predictions(on the training set for now), instead of 
#calling the classifiers predict() method, you can just run this code:
y_train_pred_90 = (y_scores >= threshold_90_precision)


# In[69]:


precision_score(y_train_5, y_train_pred_90)


# In[70]:


recall_score(y_train_5,y_train_pred_90)


# In[71]:


#A high-precison classifier is not very useful if its recall is too low!
#Therefore the precision classfier isn't too useful for this project.
#Always make sure to check the recall value


# # The ROC Curve

# In[75]:


#The Receiver operating charateristic(ROC)curve is used to plot for 
#plotting the true positive rate(another name for recall) against the false positive rate.
#Before you compute the ROC curve, you first need to compute True Positive rate(TPR) and False Positive Rate(FPR)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label =None):
    plt.plot(fpr, tpr, linewidth = 2, label=label)
    plt.plot([0,1],[0,1], 'k--')#dashed diagonal
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate(Fall-Out)', fontsize =16)
    plt.ylabel('True Positive Rate(Recall)', fontsize =16)
    plt.grid(True)

plt.figure(figsize = (8,6))
plot_roc_curve(fpr,tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:") 
plt.plot([fpr_90], [recall_90_precision], "ro")               
save_fig("roc_curve_plot")                                    
plt.show()
    


# In[76]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# In[77]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state =42)
y_probas_forest = cross_val_predict(forest_clf, X_train,y_train_5, cv =3,
                                    method = "predict_proba")


# In[78]:


y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# In[79]:


plt.plot(fpr,tpr, "b:", label = "SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()


# In[80]:


#The RandomForestClassifier's ROC curve looks much better than SGDClassifier's:
#It comes much closer to the top-left corner. As a result, its ROC AUC score is also significantly better:
roc_auc_score(y_train_5, y_scores_forest)


# # Multiclass Classification

# In[81]:


sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])


# In[82]:


some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores
#Instead or returning on score per instance
#This method returns 10 scores, one per class


# In[83]:


#The highes score is indeed one corresponding to class 5:
np.argmax(some_digit_scores)
sgd_clf.classes_
sgd_clf.classes_[5]


# In[84]:


from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state =42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)


# In[85]:


#Training a RandomForestClassifier is just as easy:
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])


# In[86]:


forest_clf.predict_proba([some_digit])


# In[87]:


cross_val_score(sgd_clf, X_train,y_train, cv=3, scoring = "accuracy")


# In[88]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring = "accuracy")


# # Error Analysis

# In[91]:


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv = 3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# In[95]:


plt.matshow(conf_mx, cmap = plt.cm.gray)
save_fig("confusion_matrix_plot", tight_layout = False)
plt.show()


# In[99]:


row_sums = conf_mx.sum(axis =1, keepdims = True)
norm_conf_mx = conf_mx / row_sums


# In[97]:


np.fill_diagonal(norm_conf_mx,  0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_errors_plot", tight_layout=False)
plt.show()


# In[103]:


cl_a, cl_b = 3,5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]


plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
save_fig("error_analysis_digits_plot")
plt.show()


# # Multilabel Classification

# In[104]:


from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7) #If the digit is bigger than 7
y_train_odd = (y_train % 2 == 1) #If the digit is even or odd
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# In[106]:


#Now I can make a prediction
knn_clf.predict([some_digit])
#Basically the output show us that that digit 5 is indeed not large(false) and odd(True)


# In[108]:


y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")


# # Multioutput Classification

# In[ ]:


#This section of code cleans the digital images
noise = np.random.randint(0,100,(len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0,100,(len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


# In[112]:


#Plotting the difference two digits: The left is the messy one, the right is the cleaned image.
some_index = 0
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
save_fig("noisy_digit_example_plot")
plt.show()


# In[113]:


#This final piece of code concludes the tutorial on classisification
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
save_fig("cleaned_digit_example_plot")


# # Tackling the titanic dataset

# In[118]:


import os
import urllib.request

TITANIC_PATH = os.path.join("datasets", "titanic")
DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/"

def fetch_titanic_data(url=DOWNLOAD_URL, path=TITANIC_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)
    for filename in ("train.csv", "test.csv"):
        filepath = os.path.join(path, filename)
        if not os.path.isfile(filepath):
            print("Downloading", filename)
            urllib.request.urlretrieve(url + filename, filepath)

fetch_titanic_data()   


# In[130]:


import pandas as pd
def load_titanic_data(filename, titanic_path = TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


# In[131]:


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")


# In[132]:


train_data.head()


# In[133]:


#Looking at the passenger ID
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")


# In[135]:


#Looking at the catergories in the data
train_data.info()


# In[136]:


#Calculating the median age for all females on the titanic
train_data[train_data["Sex"] == "female"]["Age"].median()


# In[137]:


train_data.describe()
#Approximately 38% percent of the people on the titanic survived
#The Fare(Cost) was about 32.20 pound Sterling
#The average age for all passengers on the ship was 29.7 or 30 years


# In[138]:


train_data["Survived"].value_counts()
#So 549 people died, while 342 people survived


# In[139]:


#Were going to look at how many people were in each class
train_data["Pclass"].value_counts()
#There was 491 people in 3rd class, 184 in second class, and 216 in first class.


# In[140]:


#Find the number of men and women
train_data["Sex"].value_counts()


# In[141]:


#Looking at where the people on the ship embarked from
train_data["Embarked"].value_counts()
#S = Southampton
#Q = Queenstown
#C = Cherbourg


# In[142]:


#Now it's time to build our preprocessing pipelines, starting with the pipeline 
#for numberical attributes
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])
#Numerical attributes


# In[143]:


#Now we can build a categorical attributes
from sklearn.preprocessing import OneHotEncoder


# In[144]:


cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("cat_encoder", OneHotEncoder(sparse = False))
])
#Catergorical attributes


# In[146]:


#Joining/combining the numberical and categorical pipelines:
from sklearn.compose import ColumnTransformer
num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])


# In[147]:


#Now that we have a pipeline that can process numerical and categorical data
#It's time to feed the data to any machine learning model we want
X_train = preprocess_pipeline.fit_transform(
    train_data[num_attribs + cat_attribs]) #We're preparing to train our model
X_train


# In[148]:


y_train = train_data["Survived"]


# In[150]:


#We're now all set up to train our classifier. 
#We're gonna start off with RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
forest_clf.fit(X_train, y_train)


# In[ ]:


#The model is now trained, now time to make predictions on the test set.


# In[151]:


X_test = preprocess_pipeline.transform(test_data[num_attribs + cat_attribs])
y_pred = forest_clf.predict(X_test)


# In[152]:


#Cross validation time(Random Forest)
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv =10)
forest_scores.mean()


# In[168]:


#Support Vector classifier
from sklearn.svm import SVC
svm_clf = SVC(gamma = "auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv = 10)
svm_scores.mean()


# In[169]:


#Plotting the accuracy of the Support Vector Machine and Random Forest Classifier
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2] *10, forest_scores , ".")
plt.boxplot([svm_scores, forest_scores], labels = ("SVM", "Random Forests"))
plt.ylabel("Accuracy", fontsize = 14)
plt.show()


# In[170]:


#This section of code finds the average propability of survival for 6 diffrent ages
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()


# In[172]:


train_data["Class_bucket"] = train_data["Pclass"]
train_data[["Class_bucket", "Survived"]].groupby(['Class_bucket']).mean()


# In[173]:


train_data["gender_classification"] = train_data["Sex"]
train_data[["gender_classification", "Survived"]].groupby(['gender_classification']).mean()


# In[ ]:




