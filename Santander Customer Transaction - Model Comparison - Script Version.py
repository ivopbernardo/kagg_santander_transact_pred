#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Matplotlib Library
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#SkLearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn import metrics


import os
print(os.listdir("../input"))

#Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import tensorflow as tf
from keras import regularizers

# Any results you write to the current directory are saved as output.


# In[2]:


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# The following Kernel is organized in the following topics:
#     * EDA
#     * Data cleaning
#     * Model Developing:
#         * Random Forest
#         * Light Gradient Boosting
#         * Neural Network
#     * Model comparison        
#         

# In[3]:


#Let's check the correlation with target and find out which variables are mostly correlated with the target var
target_correlation = pd.DataFrame(training_data.corr()).iloc[1:,0]

target_correlation.sort_values(ascending=False)


# The variable with the higher positive correlation (pearson) has a value of 0.06731 (var_6) so we do not have many hopes for linear models.
# Var_81 is the most negative one with a value -0.080917. Mostly the correlation between variables range between these values higher than -0.1 and lower that 0.1

# In[ ]:


#Removing outliers from variables using interquartile range
Q1 = training_data.iloc[:,2:].quantile(0.25)
Q3 = training_data.iloc[:,2:].quantile(0.75)
IQR = Q3-Q1

train_data_outliers = training_data[~((training_data.iloc[:,2:] < (Q1 - 1.5 * IQR)) |(training_data.iloc[:,2:] > (Q3 + 1.5 * IQR))).any(axis=1)]
print("Number of outliers removed was : {} rows".format(len(training_data)-len(train_data_outliers)))


# In[ ]:


#Being a binary classification problem, class imbalance might be a problem. Let's check it:
print("% of target variable is {} with {} positive cases.".format(train_data_outliers.target.sum()/len(train_data_outliers),train_data_outliers.target.sum()))

#Only 9.7% of the target is a positive variable. Let's undersample negative cases and construct the train and test datasets
sc_X = StandardScaler()


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def construct_Train_Test(weight, full_sample):  
    if full_sample == 1:
        a, b, c, d = train_test_split(training_data.iloc[:,2:], training_data['target'], test_size = 0.2)
        a = sc_X.fit_transform(a)
        b = sc_X.transform(b) 
    elif weight == 0:
        sample_data = training_data.sample(frac = 0.2)
        print("% of positive target values on training data is {}".format(training_data['target'].sum()/len(training_data)))
        a, b, c, d = train_test_split(sample_data.iloc[:,2:], sample_data['target'], test_size = 0.2)
        a = sc_X.fit_transform(a)
        b = sc_X.transform(b)
    else: 
        positive_cases = training_data.loc[training_data['target']==1]
        negative_cases = training_data.loc[training_data['target']==0].sample(int(weight*len(positive_cases)))
        full_table = positive_cases.append(negative_cases)
        a, b, c, d = train_test_split(full_table.iloc[:,2:], full_table['target'], test_size = 0.2)
        a = sc_X.fit_transform(a)
        b = sc_X.transform(b)
    return a, b, c, d


# In[5]:


#Baseline Random Forest
def Random_Forest_Baseline():
    baseline = RandomForestClassifier()
    return baseline

#Neural Network Parameters
def Neural_Network_Sigmoid(activ):
    #Initializing the Neural Network 
    ann = Sequential()
    #Adding the input layer and the two initial hidden layers 
    ann.add(Dense(output_dim = 128, activation = activ, input_dim = 200))
    ann.add(Dense(output_dim = 128, activation = activ))
    ann.add(Dense(output_dim = 128, activation = activ))
    #Adding the Output Layer
    ann.add(Dense(output_dim = 1, activation = 'sigmoid'))
    #Compilar a Rede Neuronal - Stochastic Gradient Descent
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[auc])
    return ann

#Neural Network Parameters
def Neural_Network_Relu(activ):
    #Initializing the Neural Network 
    ann = Sequential()
    #Adding the input layer and the two initial hidden layers 
    ann.add(Dense(output_dim = 128, activation = activ, input_dim = 200))
    ann.add(Dense(output_dim = 128, activation = activ))
    ann.add(Dense(output_dim = 128, activation = activ))
    #Adding the Output Layer
    ann.add(Dense(output_dim = 1, activation = 'sigmoid'))
    #Compilar a Rede Neuronal - Stochastic Gradient Descent
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[auc])
    return ann

#Neural Network Parameters - With Regularization
def Neural_Network_Relu_Regul(activ):
    #Initializing the Neural Network 
    ann = Sequential()
    #Adding the input layer and the two initial hidden layers 
    ann.add(Dense(output_dim = 124, activation = activ, input_dim = 200))
    ann.add(Dropout(0.5))
    ann.add(Dense(output_dim = 124, activation = activ))
    ann.add(Dropout(0.5))
    ann.add(Dense(output_dim = 24, activation = activ))
    ann.add(Dropout(0.5))
    ann.add(Dense(output_dim = 24, activation = activ ,
                kernel_regularizer=regularizers.l2(0.01)))
    #Adding the Output Layer
    ann.add(Dense(output_dim = 1, activation = 'sigmoid'))
    #Compilar a Rede Neuronal - Stochastic Gradient Descent
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[auc])
    return ann

#Random Forest Randomized Search Function Definition
def Random_Forest_Model_Random_Search():
    #Initializing the Neural Network 
    # Number of parameters for randomized grid search
    rf = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 400, num = 5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    max_depth.append(None)
    min_samples_split = [5,50,500]
    min_samples_leaf = [4,10,100]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=20, random_state=100, n_jobs = -1, scoring='roc_auc')
    return rf_random


#Random Forest with Best Parameters selected by randomized search
def Random_Forest_Model_Params(params):
        rf = RandomForestClassifier(**params)
        return rf


# In[8]:


#Model Class Implementation
models = {}

class model():
    
    def __init__(self, model_type, model_object, weight_train, full_sample):
        self.model_type = model_type
        self.model_object = model_object
        self.weight_train = weight_train
        self.full_sample = full_sample
    
    def get_train_test(self):
        a, b, c, d = construct_Train_Test(self.weight_train, self.full_sample)
        return a,b,c,d
    
    def train_model(self, epochs):
        self.epochs = epochs
        if self.model_type == 'Neural Network Sigmoid' or self.model_type == "Neural Network Relu":
            X_train, X_test, y_train, y_test = construct_Train_Test(self.weight_train, self.full_sample)
            self.history = self.model_object.fit(X_train, y_train, epochs=epochs, validation_split=0.2, shuffle=True)
        else:
            X_train, X_test, y_train, y_test = construct_Train_Test(self.weight_train, self.full_sample)
            print("Size of X_train object is {}".format(X_train.shape))
            self.model_object.fit(X_train, y_train)
            self.history = None
        return self.history

    def get_auc(self):
        if self.model_type == 'Neural Network Sigmoid':
            y_pred = self.model_object.predict(sc_X.transform(training_data.iloc[:,2:]))
            model_key = 'For Model Type {}, number of epochs {} and weight of {} for each positive value on training'.format(self.model_type, self.weight_train) 
            models[model_key] = roc_auc_score(training_data.iloc[:,1], y_pred)
        elif self.model_type == 'Neural Network Relu':
            y_pred = self.model_object.predict(sc_X.transform(training_data.iloc[:,2:]))
            model_key = 'For Model Type {}, number of epochs {} and weight of {} for each positive value on training'.format(self.model_type , self.epochs, self.weight_train) 
            models[model_key] = roc_auc_score(training_data.iloc[:,1], y_pred)
            
    def plot_learning_curve(self):
        if self.model_type == 'Neural Network Sigmoid' or self.model_type == "Neural Network Relu":
            plt.plot(np.arange(0,self.epochs), self.history.history['loss'], '--', color="#111111",  label="Training Accuracy - Relu Activation")
            plt.plot(np.arange(0,self.epochs), self.history.history['val_loss'], color="#111111", label="Cross-validation Accuracy - Activation")
            # Create plot
            plt.title("Learning Curve")
            plt.xlabel("Nb. Epochs"), plt.ylabel("Binary Cross Entropy"), plt.legend(loc="best")
            plt.tight_layout()
            plt.show()
        


# **First Model: Neural Network with Rectified Linear Unit activation function. - Train it with the entire sample first - plot Learning Curve to check bias and variance**

# In[7]:


#Train Neural Network with Relu Activation function on full training data
annr = Neural_Network_Relu('relu')
artificial_nn_relu = model('Neural Network Relu', annr, 0, 1)
neural_model = artificial_nn_relu.train_model(100) 
artificial_nn_relu.plot_learning_curve()
artificial_nn_relu.get_auc()


# By the loss function, we can see that we are overfitting our hypothesis. We will add regularizations to the Neural Network.

# In[ ]:


#Train Neural Network with Relu Activation function on full training data
annr_reg = Neural_Network_Relu_Regul('relu')
artificial_nn_relu_reg = model('Neural Network Relu', annr_reg, 1, 0)
neural_model = artificial_nn_relu_reg.train_model(100) 
artificial_nn_relu_reg.plot_learning_curve()
artificial_nn_relu_reg.get_auc()


# In[ ]:


models


# In[ ]:



#Train same Neural Network with relu on an "artificial" data set with balanced classes - Looping through several weights
artificial_nn_relu_reg = model('Neural Network Relu', annr_reg, i, 0)
neural_model = artificial_nn_relu_reg.train_model(100) 
artificial_nn_relu_reg.plot_learning_curve()
artificial_nn_relu_reg.get_auc()
print(models)


# For 50 epochs, Relu activation function seems to work best to this problem. We will train the model with 100 epochs just to check the validity of this statement.

# In[ ]:


#Train Neural Network with Relu Activation function on full training data - 100 epochs
annr = Neural_Network_Relu('relu')
artificial_nn_relu = model('Neural Network Relu', annr, 0, 1)
neural_model_history = artificial_nn_relu.train_model(100) 


# In[ ]:


plt.plot(np.arange(0,100), neural_model_history.history['auc'], '--', color="#111111",  label="Training AUC - Relu Activation")
plt.plot(np.arange(0,100), neural_model_history.history['val_auc'], color="#111111", label="Cross-validation AUC - Activation")
plt.plot(np.arange(0,100), neural_model_s.history['auc'], '--', color='red',  label="Training AUC - Sigmoid Activation")
plt.plot(np.arange(0,100), neural_model_s.history['val_auc'], color='red', label="Cross-validation AUC - Sigmoid Activation")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Nb. Epochs"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# Relu and Sigmoid activation functions start to converge but still have a difference in AUC. And the learning curve stabilizes with lesser epochs for Relu function so we will choose that as baseline model.
# Let's check if balancing the dataset (remember that we only have 1 in 10 cases of positive target) can boost the generalization of the model.

# In[ ]:


#Train Neural Network with Relu Activation function on full training data - 100 epochs - 3 positive cases for each negative case on the dataset
annr_balanced = Neural_Network_Relu('relu')
artificial_nn_relu_balanced = model('Neural Network Relu', annr_balanced, 3, 0)
neural_model_balanced = artificial_nn_relu_balanced.train_model(100) 


# In[ ]:


plt.plot(np.arange(0,100), neural_model_history.history['auc'], '--', color="#111111",  label="Training AUC - Relu Activation - Full Sample")
plt.plot(np.arange(0,100), neural_model_history.history['val_auc'], color="#111111", label="Cross-validation AUC - Relu Activation - Full Sample")
plt.plot(np.arange(0,100), neural_model_balanced.history['auc'], '--', color='red',  label="Training AUC - Relu Activation - Balanced 3/1")
plt.plot(np.arange(0,100), neural_model_balanced.history['val_auc'], color='red', label="Cross-validation AUC - Relu Activation - Balanced 3/1")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Nb. Epochs"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# Wow! Very promising.. 
# Not so fast. We are calculating the AUC on a "balanced" data set and this does not mean that the scoring on the real data will work as good.

# In[ ]:


#Updating AUC on dictionary for each model
artificial_nn_relu.get_auc()
artificial_nn_relu_balanced.get_auc()


# Weight 0 means full sample. There is not a big difference between AUC's for both methods but full sample wins.
# Let's see how a Random Forest might fit the data - We will use a randomized search to find the best parameters.
# To find them we use a sample of 20% of the full data set as training a lot of forests on 1 millions X 200 features data is computationally expensive.

# In[ ]:


#Random Forest - We have trained a random forest randomized search and the parameters that we have got were {'n_estimators':400, 'min_samples_split':10,'min_samples_leaf':10, 'max_features':'sqrt', 'max_depth':60}
best_params = {'n_estimators':400, 'min_samples_split':10,'min_samples_leaf':10, 'max_features':'sqrt', 'max_depth':60}


# In[ ]:


rf_best_params = Random_Forest_Model_Params(best_params)
X_train, X_test, y_train, y_test = construct_Train_Test(0, 1)
train_sizes, train_scores, test_scores = learning_curve(rf_best_params, X_train, y_train,
                                                        cv=2,
                                                            scoring='roc_auc',
                                                        n_jobs=-1, 
                                                        train_sizes=np.linspace(0.01, 1.0, 30),
                                                        verbose = 10)


# In[ ]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, '--', color="red",  label="Training ROC AUC")
plt.plot(train_sizes, test_mean, color="black", label="Cross-validation ROC AUC")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# Awful overfitting! Made a rookie error in the randomized search and possibly will have to train a random forest with different parameters. We will modify min_samples_split, first.

# In[ ]:


#Random Forest - We have trained a random forest randomized search and the parameters that we have got were {'n_estimators':400, 'min_samples_split':10,'min_samples_leaf':10, 'max_features':'sqrt', 'max_depth':60}
best_params_alternative = {'n_estimators':100, 'min_samples_split':1000,'min_samples_leaf':1000, 'max_features':'sqrt', 'max_depth':100}


# In[ ]:


rf_best_params_alternative = Random_Forest_Model_Params(best_params_alternative)
X_train, X_test, y_train, y_test = construct_Train_Test(0, 1)
train_sizes, train_scores, test_scores = learning_curve(rf_best_params_alternative, X_train, y_train,
                                                        cv=2,
                                                            scoring='roc_auc',
                                                        n_jobs=-1, 
                                                        train_sizes=np.linspace(0.01, 1.0, 10),
                                                        verbose = 10)                    
                                                    


# In[ ]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, '--', color="red",  label="Training ROC AUC")
plt.plot(train_sizes, test_mean, color="black", label="Cross-validation ROC AUC")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

