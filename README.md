#Study Title: Prediction of Self-Healing Capacity of Cracks using Sodium Silicate Capsule by comparing advanced Machine Learning Approaches.

#Study Introduction: 
Advances in machine learning (ML) methods are important in industrial engineering and attract great attention in recent years. However, a comprehensive 
comparative study of the most advanced ML algorithms is lacking. As Crack in concrete structure is a major problem affecting the durability of the structures and which may lead to failure of the structure. Repair and rehabilitation of concrete building or structures is expensive and it is difficult to access the damage after completing the construction of the structure. The solution for this type of problem is selfhealing concretes. Capsule based self-healing concrete is a new type of method used to decrease the damage and increase the service life and performance of a concrete structure. Healing agent like sodium silicate being capsulated and introduced into the concrete while casting. To observe the self-healing mechanism in the concrete sample, cracks were induced in the concrete sample by three point bending test. The healing process of concrete sample was observed using ultrasonic non-destructive concrete tester. Self-healing concept was developed from wound healing mechanism in human being, in which certain level of wound can heal by itself. Self-healing agents have the ability to improve the properties of concrete even after damage. 
![image](https://user-images.githubusercontent.com/53009824/188303239-200ed20b-2543-45d3-9cbf-6fea601c8167.png)

Thus, considering an existing study paper of self-healing concretes where reacting sodium silicate with the calcium hydroxide present in the concrete to form crystal like substance. Our study to understand and predict the frequency of the ultra sonic concrete tester after nth day using the advanced regression ML algorithms such as Support Vector Regression, Decision Tree Regression, Artificial Neural Network, Bayesian Ridge Regression and Ridge Regression, Gradient Boosting Regression and others.

Keywords: Self-healing concrete, Sodium silicate capsules, Crack depth, Ultra sonic concrete tester, Durability, crack closure percentage, machine learning,
prediction.

#Study targeted ML algorithms:
1. Support Vector Regression: 
Support Vector Regression supports both linear and non-linear regressions. This method works on the principle of the Support Vector Machine. SVR differs from SVM in the way that SVM is a classifier that is used for predicting discrete categorical labels while SVR is a regressor that is used for predicting continuous ordered variables. SVR acknowledges the presence of non-linearity in the data and provides a proficient prediction model. When we are moving on with SVR, is to basically consider the points that are within the decision boundary line and Our best fit line is the hyperplane that has a maximum number of points, same approach in SVM.
![image](https://user-images.githubusercontent.com/53009824/188302748-e49c246c-49e9-490e-a31a-e0320b3b57c8.png)

2. Decision tree regression: 
Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output. Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values.
DTR is a non-parametric procedure for predicting continuous dependent variable where the data is partitioned into nodes on the basis of conditional binary responses. Models use a binary tree to recursively partition the predictor space into subsets in which the distribution of y is successively more homogenous. A decision tree P with t terminal nodes is used for communicating the decision.

3. ANN Regression: 
Regression ANNs predict an output variable as a function of the inputs. The input features (independent variables) can be categorical or numeric types, however, for regression ANNs, we require a numeric dependent variable. If the output variable is a categorical variable (or binary) the ANN will function as a classifier.
ANN is a mathematical technique using an analogy to biological neurons to generate a general solution to a problem. All neural functions are stored in the neurons and the connections between them. After learning historical data, ANN can be used effectively to predict new data. The training of ANNs is considered as the establishment of new connections between neurons. ANN architecture may have one or more hidden layers between the input and output layers. Each layer constitutes neurons, which are connected with other neurons by the weights passing signals to others. When the amount of signals received by one neuron overtakes its threshold, the activation function is awoken and the outcome is treated as the input of next neuron. It can approximate an arbitrary nonlinear function with satisfactory accuracy. They learn from examples by building an input–output mapping without explicit derivation of the model equation. They have been widely used in pattern classification, function approximation, optimization, prediction and automatic control and in many different domains, such as load forecasting and strength forecast.

4. Bayesian ridge regression: 
BRR allows a natural mechanism to survive insufficient data or poorly distributed data by formulating linear regression using probability distributors rather than point estimates. The output or response ‘y’ is assumed to drawn from a probability distribution rather than estimated as a single value.

5. Gradient Boosting Regression: 
A machine learning technique for regression problems, typically on the basis of decision trees. It builds the model in a stage-wise fashion and allows optimization of an arbitrary differentiable loss function. The goal in GBR is to find a loss function F*(x) and minimize the expected value of it over the joint distribution of all (y, x) values. Boosting evaluates F*(x) by an additive expansion. The gradient boosting algorithm improves F*(x) by adding an estimator h to provide a better model. A eneralization of this idea to loss functions other than squared error is that residuals for a given model are the negative gradients of the squared error loss function. Hence, gradient boosting is a gradient descent algorithm by adding a different loss.

6. Multiple Regression: 
MR also known as multiple linear regression (MLR), is a statistical technique that uses two or more explanatory variables to predict the outcome of a response variable. In other words, it can explain the relationship between multiple independent variables against one dependent variable.

#Study Conclusion: 
Gradient Boosting Regression and Multiple Linear Regression predicted the frequency change closely then the others and some of the other algo is not best suited to predict industry oriented usage based on observations (small datasets).
