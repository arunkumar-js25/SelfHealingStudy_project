#!/usr/bin/env python
# coding: utf-8

import pandas
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

df = pandas.read_csv("D:\ArunKumar\Library\Machine Learning\DataSet_14L_1.5P.csv")
X = df[['Load','Capsule','InitialFrequency','Days']]
y = df['Frequency']

#TrainingSet
X_test = [[14,1.5,106.2,5]]


#Training Set
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
print(X_test)
print(y_test)


#BRR : Bayesian Ridge Regression : Very effective when the size of the dataset is small.
from sklearn.linear_model import BayesianRidge

# Creating and training model
model = BayesianRidge()
model.fit(X_train, y_train)

# Model making a prediction on test data
y_pred = model.predict(X_test)
print(y_pred)


#GBR : Gradient Boosting Regression - WORKS ACCURATE
from sklearn.ensemble import GradientBoostingRegressor

# with new parameters
model = GradientBoostingRegressor(n_estimators=600, max_depth=5, learning_rate=0.01, min_samples_split=3)
# with default parameters
model = GradientBoostingRegressor()

model.fit(X, y)

y_pred = model.predict(X_test)
print(y_pred)


#Ridge Regression : Ridge regression is a method we can use to fit a regression model when multicollinearity is present in the data.
from numpy import arange
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold


#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#define Ridge model
#model = Ridge(alpha = 0.5, tol = 0.001, solver ='auto', random_state = 42)
model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')

#fit model
model.fit(X, y)

#display lambda that produced the lowest test MSE
print("Alpha : ",model.alpha_)

#predict hp value using ridge regression model
y_pred = model.predict(X_test)
print(y_pred)


#Lasso Ridge Regression
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold

#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#define LassoCV model
model = LassoCV(alphas=arange(0.1, 1, 0.01), cv=cv, n_jobs=-1)

#fit model
model.fit(X, y)

#predict hp value using ridge regression model
y_pred = model.predict(X_test)
print(y_pred)


# Standard Scaler
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scaledX = scale.fit_transform(X)
#print(scaledX)

model = linear_model.LinearRegression()
model.fit(scaledX, y)

scaled = scale.transform(X_test)
y_pred = model.predict(scaled)
print(y_pred)



#DTR : Decision Tree Regression
from sklearn import tree

model = tree.DecisionTreeRegressor(max_depth=5)
model.fit(X, y)

y_pred = model.predict(X_test)
print(y_pred)


#Support Vector Regression
from sklearn import svm
model = svm.SVR()
model.fit(X, y)

y_pred = model.predict(X_test)
print(y_pred)


# LinearSVR - Support Vector Regression
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline

model = make_pipeline(LinearSVR(random_state=0, tol=1e-5))
model.fit(X, y)

y_pred = model.predict(X_test)
print(y_pred)


# ANN - Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

model = MLPRegressor(max_iter=500)
model.fit(X, y)

y_pred = model.predict(X_test)
print(y_pred)



# Evaluation of r2 score of the model against the test set
from sklearn.metrics import r2_score
print(f"r2 Score Of Test Set : {r2_score(y_test, y_pred)}")



from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print("MSE: %.2f" % mse)


# Original Vs Predicted Values
x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# Plot the results
plt.figure()
plt.scatter(X['Days'], y, s=5, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test['Days'], y_pred, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.title("GBR")
plt.legend()
plt.show()




# Implementation of gradient descent in linear regression 
import numpy as np 
import matplotlib.pyplot as plt 
  
class Linear_Regression: 
    def __init__(self, X, Y): 
        self.X = X 
        self.Y = Y 
        self.b = [0, 0] 
      
    def update_coeffs(self, learning_rate): 
        Y_pred = self.predict() 
        Y = self.Y 
        m = len(Y) 
        self.b[0] = self.b[0] - (learning_rate * ((1/m) *
                                np.sum(Y_pred - Y))) 
  
        self.b[1] = self.b[1] - (learning_rate * ((1/m) *
                                np.sum((Y_pred - Y) * self.X))) 
  
    def predict(self, X=[]): 
        Y_pred = np.array([]) 
        if not X: X = self.X 
        b = self.b 
        for x in X: 
            Y_pred = np.append(Y_pred, b[0] + (b[1] * x)) 
  
        return Y_pred 
      
    def get_current_accuracy(self, Y_pred): 
        p, e = Y_pred, self.Y 
        n = len(Y_pred) 
        return 1-sum( 
            [ 
                abs(p[i]-e[i])/e[i] 
                for i in range(n) 
                if e[i] != 0] 
        )/n 
    #def predict(self, b, yi): 
  
    def compute_cost(self, Y_pred): 
        m = len(self.Y) 
        J = (1 / 2*m) * (np.sum(Y_pred - self.Y)**2) 
        return J 
  
    def plot_best_fit(self, Y_pred, fig): 
                f = plt.figure(fig) 
                plt.scatter(self.X, self.Y, color='b') 
                plt.plot(self.X, Y_pred, color='g') 
                f.show()
def main(): 
    X = np.array([i for i in range(11)]) 
    Y = np.array([2*i for i in range(11)]) 
  
    regressor = Linear_Regression(X, Y) 
  
    iterations = 0
    steps = 100
    learning_rate = 0.01
    costs = [] 
      
    #original best-fit line 
    Y_pred = regressor.predict() 
    regressor.plot_best_fit(Y_pred, 'Initial Best Fit Line') 
      
  
    while 1: 
        Y_pred = regressor.predict() 
        cost = regressor.compute_cost(Y_pred) 
        costs.append(cost) 
        regressor.update_coeffs(learning_rate) 
          
        iterations += 1
        if iterations % steps == 0: 
            print(iterations, "epochs elapsed") 
            print("Current accuracy is :", 
                regressor.get_current_accuracy(Y_pred)) 
  
            stop = input("Do you want to stop (y/*)??") 
            if stop == "y": 
                break
  
    #final best-fit line 
    regressor.plot_best_fit(Y_pred, 'Final Best Fit Line') 
  
    #plot to verify cost function decreases 
    h = plt.figure('Verification') 
    plt.plot(range(iterations), costs, color='b') 
    h.show() 
  
    # if user wants to predict using the regressor: 
    regressor.predict([i for i in range(10)])
if __name__ == '__main__': 
    main()


from sklearn import tree
dot_data = tree.export_graphviz(model, out_file=None,feature_names=['Load','Capsule','InitialFrequency','Days'], class_names=['Frequency'])

import pydotplus
import graphviz
from IPython.display import Image
graph = pydotplus.graph_from_dot_data(dot_data) # Show graph
Image(graph.create_png())

