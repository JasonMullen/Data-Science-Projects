#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas ')


# In[2]:


get_ipython().system('pip install matplotlib')


# In[3]:


get_ipython().system('pip install numpy')


# In[4]:


get_ipython().system('pip install tarfile')


# In[5]:


get_ipython().system('pip install sklearn')


# In[6]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[7]:


import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[8]:


fetch_housing_data()


# In[9]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[10]:


housing = load_housing_data()
housing.head()


# In[11]:


housing.info()


# In[12]:


housing["ocean_proximity"].value_counts()


# In[13]:


housing.describe()


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize = (20,15))
plt.show()


# In[15]:


import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[16]:


train_set, test_set = split_train_test(housing, 0.2)
len(train_set)


# In[17]:


len(test_set)


# In[18]:


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 **32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[19]:


housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[20]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[21]:


housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0.,1.5,3.0,4.5,6.,np.inf],
                              labels = [1,2,3,4,5])
housing["income_cat"].hist()


# In[22]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits =1, test_size = 0.2, random_state =42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[23]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat",axis = 1, inplace = True)


# In[24]:


housing = strat_train_set.copy()


# In[25]:


housing.plot(kind = "scatter", x="longitude", y="latitude")


# In[26]:


housing.plot(kind = "scatter", x="longitude", y="latitude", alpha = 0.1)


# In[27]:


housing.plot(
            kind = "scatter", x = "longitude", y ="latitude", alpha = 0.4,
            s = housing["population"]/100, label = "population", figsize = (10,7),
            c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True
            )
plt.legend


# In[28]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[29]:


from pandas.plotting import scatter_matrix

attributes =["median_house_value", "median_income","total_rooms",
            "housing_median_age"]
scatter_matrix(housing[attributes],figsize = (12,8))


# In[30]:


#The median income has the highest correlation to the the median house value
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)


# In[31]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# In[32]:


housing = strat_train_set.drop("median_house_value", axis =1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[33]:


#Data Cleaning
#housing.dropna(subset=["total_bedrooms"]) #Data cleaning:Option 1
housing.drop("total_bedrooms", axis =1) #Data cleaning: Option 2
#median = housing["total_bedrooms"].median() #Data cleaning: Option 3
housing["total_bedrooms"].fillna(median, inplace = True)


# In[34]:


#This section of code takes care of the missing vlaues.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
housing_num = housing.drop("ocean_proximity", axis =1)


# In[35]:


#Now you can fit the imputer instance to the training data using the fit()method:
imputer.fit(housing_num)


# In[36]:


imputer.statistics_


# In[37]:


housing_num.median().values


# In[38]:


X = imputer.transform(housing_num)


# In[39]:


housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)


# In[40]:


#Looking at the value for the first 10 instances.
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[41]:


#Converting the texts to to numbers inorder to input the information
#into a machine learning algorithm
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[42]:


#Retreving the list of categories
ordinal_encoder.categories_


# In[43]:


#One hot encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[44]:


housing_cat_1hot.toarray()


# In[45]:


cat_encoder.categories_


# In[46]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X, y = None):
        return self #nothing else to do
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[47]:


#Pipeline class that helps with transformations.
#This example is a small pipeline for the numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[48]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat",OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)


# In[49]:


#Machine Learning linear Regression training model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#Now it's time to test the model out
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:",lin_reg.predict(some_data_prepared))


# In[50]:


print("Labels:",list(some_labels))


# In[51]:


#mean squared error model
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[52]:


from sklearn.tree import DecisionTreeRegressor
#trained model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

#Time to evaluate the training set
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[53]:


#Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv = 10)
tree_rmse_scores = np.sqrt(-scores)

#Time to evalueate the training set
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
display_scores(tree_rmse_scores)


# In[54]:


#Computing the same scores for the linear regression model just to be sure
lin_scores = cross_val_score(lin_reg,housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv =10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[55]:


#Random forest regressor models
print("start")
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
#Time to evaluate the data
print("midpoint")
forest_reg.fit(housing_prepared, housing_labels)
forest_mse = mean_squared_error(housing_labels,housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
print("midpoint past")
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                               scoring = "neg_mean_squared_error", cv =10 )
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
print("end")


# In[56]:


#Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap': [False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, 
                          scoring = 'neg_mean_squared_error',
                          return_train_score =True)
grid_search.fit(housing_prepared, housing_labels)


# In[57]:


grid_search.best_params_


# In[58]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score),params)


# In[59]:


#Esemble/combine models.
#Analyze the best models and their errors
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[60]:


#displaying the important scores next to their corresponding attributes names:
extra_attribs = ["rooms_per_hhold","pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse = True)


# In[61]:


#Evaluate your system on the test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis =1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[67]:


#Want to compute a 95% confidence interval for the genralization error
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1,
                        loc=squared_errors.mean(), 
                        scale = stats.sem(squared_errors)))


# In[68]:


#Exercise Solutions


# In[76]:


#Exercise 1
#Support Vector Machines regressor
print("Start")
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
param_grid = [ 
    {'kernal': ['linear'], 'C': [10.,30.,100.,300.,1000.,3000.,10000.,]},
    {'kernal':['rbf'], 'C': [1.0,3.0,10.,30.,100.,300.,1000.0],
    'gamma': [0.01,0.03,0.1,0.3,1.0,3.0]},
]

print("middle")

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5,scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(housing_prepared, housing_labels)
#print("end")


# In[77]:


#Will take 45-mins to run
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# In[82]:


from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])
class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances,self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]


# In[83]:


#Trying to create a single pipeling that does the full data preparation plus the final prediction.
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances,k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])


# In[ ]:





# In[ ]:




