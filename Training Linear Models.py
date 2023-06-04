#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# # Linear Regression

# # The Normal Equation

# In[3]:


import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100,1)


# In[4]:


plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.axis([0,2,0,15])
save_fig("generated_data_plot")
plt.show()


# In[5]:


#Time to compute Theta hat using the Normal Equation. We will use the inv()
#function from NumPy's Linear Algebra model(np.linalg) to compute the inverse
#of a matrix, and the dot() method for matrix multiplication:
X_b = np.c_[np.ones((100,1)), X] #add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


# In[6]:


#The actual function that was used to generate the data is y = 4 +3x1 + Gaussian noise.
#Theta zero = 4.21
#Theta one = 2.77
theta_best


# In[7]:


#Making predictions with Theta hat
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict


# In[8]:


#Plotting the model's predictions
plt.plot(X_new, y_predict, "r-")
plt.plot(X,y, "b.")
plt.axis([0,2,0,15])
plt.show()


# In[9]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_


# In[10]:


theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b,y, rcond = 1e-6)
theta_best_svd


# In[11]:


#This section of code will computer the pseudoinverse Moore-Penrose inverse
np.linalg.pinv(X_b).dot(y)


# In[12]:


#The pseduoinverse is computed using a standard Matrix factorization
#The SVD decompses the trainig set Matrix X into the matrix mulitplication
#Of three matrices  U Σ VT(seenumpy.linalg.svd()). 


# # Gradient Descent
# 

# # Batch Gradient Descent

# In[14]:


eta = 0.1 #Learning Rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) #Random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients


# In[15]:


theta


# In[16]:


X_new_b.dot(theta)


# In[20]:


theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path = None):
    m = len(X_b)
    plt.plot(X,y,"b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta)- y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize = 18)
    plt.axis([0,2,0,15])
    plt.title(r"$\eta = {}$".format(eta), fontsize = 16)


# In[21]:


np.random.seed(42)
theta = np.random.randn(2,1) #random initialization

plt.figure(figsize = (10,4))
plt.subplot(131); plot_gradient_descent(theta,eta = 0.02)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)

save_fig("gradient_descent_plot")
plt.show()


# In[22]:


#*n=0.02 IS too slow
#*n = 0.1 is good
#*n = 0.5 The learning rate is too high.


# # Stochastic Gradient Descent

# In[23]:


theta_path_sgd = []
m = len(X_b)
np.random.seed(42)


# In[24]:


n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    # not shown in the book
            y_predict = X_new_b.dot(theta)           # not shown
            style = "b-" if i > 0 else "r--"         # not shown
            plt.plot(X_new, y_predict, style)        # not shown
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)                 # not shown

plt.plot(X, y, "b.")                                 # not shown
plt.xlabel("$x_1$", fontsize=18)                     # not shown
plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
plt.axis([0, 2, 0, 15])                              # not shown
save_fig("sgd_plot")                                 # not shown
plt.show()               


# In[25]:


theta


# In[26]:


from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())


# In[27]:


sgd_reg.intercept_, sgd_reg.coef_


# In[ ]:




