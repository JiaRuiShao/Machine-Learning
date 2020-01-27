## Classification(Logistic Regression) AND Regularization

## A. Classification

Logistic regression is a method for classifying data into discrete outcomes. For example, we might use logistic regression to classify an email as spam or not spam. In this module, we introduce the notion of classification, the cost function for logistic regression, and the application of logistic regression to multi-class classification.

### I. Classification and Representation

1. **Binary Classification**

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. 

![w3.1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.1.PNG?raw=true)

For now, we will focus on the **binary classification problem** in which y can take on only two values, 0 and 1. For instance, if we are trying to build a spam classifier for email, then x<sup>(i)</sup> may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, y∈{0,1}. **0 is also called the negative class, and 1 the positive class**, and they are sometimes also denoted by the symbols “\-” and “\+.” Given x<sup>(i)</sup>, **the corresponding y<sup>(i)</sup> is also called the label for the training example**.

For binary classification, the assignment of which one is zero and which one is one doesn't matter at all.

**How do we develop a classification algorithm**?

Set the threshold as c, if H<sub>θ</sub>(x) >= c, predict "Y = 1"; if H<sub>θ</sub>(x) < c, predict "Y = 0"

\*\*Implementing linear regression to a binary dataset is **a BAD idea** b/c the predicted results could be largely greater than 1 or smaller than 0. That's why we use logistic Regression in which 0 <= H<sub>θ</sub>(x) <= 1.

2. **Hypothesis Representation**

In order to make 0 <= H<sub>θ</sub>(x) <= 1, we need to change our form for our hypothesis H<sub>θ</sub>(x). This is accomplished by plugging θ<sup>T</sup>x into the Logistic Function. 

Logistic/Sigmoid function:

![sigmoid function](https://enlight.nyc/img/sigmoid.png)

**z = θ<sup>T</sup>x**

**g(z) = 1/(1+e<sup>-z</sup>)**

**H<sub>θ</sub>(x) = g(θ<sup>T</sup>x)**

Interpretation of Hypothesis Output

The function g(z), shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

h<sub>θ</sub>(x) will give us the probability that our output is 1. For example, h<sub>θ</sub>(x) = 0.7 gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

P(y=0|x;θ) + P(y=1|x;θ) = 1

3. **Decision Boundary**

We found that g(z) >= 0.5 when z >= 0, thus H<sub>θ</sub>(x) = g(θ<sup>T</sup>x) >= 0.5 when z = θ<sup>T</sup>x >= 0

Predict "y = 1" if θ<sup>T</sup>x >= 0; Predict "y = 0" if θ<sup>T</sup>x < 0

Decision Boundary is the line that separaters the region/area where y = 0 and where y = 1. It is created by our hypothesis function.

Decision boundary is a property of the hypothesis rather than the data:

Later on, when we talk about how to fit the parameters and there we'll end up using the training dataset to determine the value of the parameters. But once we have particular values for the parameters θ0, θ1, ..., θn that completely defines the decision boundary and we don't actually need to plot a training set in order to plot the decision boundary.

Linear Decision Boundaries

![w3.3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.3.PNG?raw=true)

Non-linear Decision Boundaries

![w3.4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.4.PNG?raw=true)

### II. Logistic Regression Model

1. **Cost Function**

Recall: the cost function for linear regression: 

![cost function for linear regression](http://www.ebc.cat/wp-content/uploads/2017/02/cost_linear_regression.png)

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

![convex vs. non-convex](https://cdn-images-1.medium.com/max/1600/1*27O7lZEDfBEqSFtTRAIMvA.jpeg)

![local vs global minimum](https://dchandra.com/images/Non_convex.png)

Instead, our cost function for logistic regression looks like:

![w3.5](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0lJmLU2hAg1reJK46-wO5A3cI8FF1oDh5L2s09eXHzecunWKy)

![Cost Function For Logistic Regression](https://i.stack.imgur.com/0CaHb.png)

+ If y = 0, the cost J(h<sub>θ</sub>(x),y) → ∞ as h<sub>θ</sub>(x) → 1
+ If y = 1, the cost J(h<sub>θ</sub>(x),y) → ∞ as h<sub>θ</sub>(x) → 0
+ If h<sub>θ</sub>(x) = y, the cost J(h<sub>θ</sub>(x),y) = 0 for both y = 0 and y = 1

2. **Simplified Cost Function and Gradient Descent**

We can compress our cost function's two conditional cases into one case:

![Simplified Cost Function For Logistic Regression](https://i.stack.imgur.com/XbU4S.png)

This cost function can be derived from statistics using the principle of maximum likelihood estimation(MLE). 

Notice that when y is equal to 1, then the second term (1 - y) log(1 - h<sub>θ</sub>(x)) will be zero and will not affect the result. If y is equal to 0, then the first term -ylog(h<sub>θ</sub>(x)) will be zero and will not affect the result.

A vectorized implementation is:

h = g(Xθ)

J(θ) = - 1/m\*(y<sup>T</sup>\*log(h) + (1-y)<sup>T</sup>\*log(1-h))

![w3.6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.6.PNG?raw=true)

**Gradient Descent**

Remember that the general form of gradient descent is:

![w3.7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.7.PNG?raw=true)

We can work out the derivative part using calculus to get:

![w3.8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.8.PNG?raw=true)

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in θ.

A vectorized implementation is:

![w3.9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.9.PNG?raw=true)

3. **Advanced Optimization**

Using some of these ideas, we'll be able to get logistic regression to run much more quickly than it's possible with gradient descent. And this will also let the algorithms scale much better to very large machine learning problems when there's a very large number of features.

Other Optimization Algorithms(other than gradient descent):

- Conjugate gradient
- BFGS
- L-BFGS

Pros | Cons
-----|-----
No need to manually pick alpha | More complex
Often compute faster than gradient descent |  

We first need to provide a function that evaluates the following two functions for a given input value θ: J(θ); ∂J(θ)/∂θ<sub>j</sub>

We can write a single function that returns both of these:

```
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

![w3.10](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.10.PNG?raw=true)

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()". 

```
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

![w3.11](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.11.PNG?raw=true)

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

![w3.12](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.12.PNG?raw=true)

Notice that the index of Octave/Matlab starts from 1 rather than 0!!

### III. Multiclass Classification

**Multiclass Classification: One-vs-all**

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

![w3.13](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.13.PNG?raw=true)

One-vs-all(one-vs-rest):

![One-vs-all](https://houxianxu.github.io/images/logisticRegression/4.png)

Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

![w3.14](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.14.PNG?raw=true)

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest h<sub>θ</sub>(x) as our prediction.

Q: Suppose you have a multi-class classification problem with kk classes (so y∈{1,2,…,k}). Using the 1-vs.-all method, how many different logistic regression classifiers will you end up training?

A: k

## B. Regularization -- Solve the problem of overfitting

Machine learning models need to generalize well to new examples that the model has not seen in practice. In this module, we introduce regularization, which helps prevent models from overfitting the training data.

### I. Solving the Problem of Overfitting

High Bias vs. High Variance

![Bias vs. Variance](https://elitedatascience.com/wp-content/uploads/2017/06/Bias-vs.-Variance-v5-2-darts.png)

**Overfitting**: If we have too many features, the learned hypothesis may fir the training set very well, but fail to generalize to new examples.

A model that has learned the noise instead of the signal is considered “overfit” because it fits the training dataset but has poor fit with new datasets.

While the black line fits the data well, the green line is overfit.

![overfit](https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Overfitting.svg/320px-Overfitting.svg.png)

Example:

Consider the problem of predicting y from x ∈ R. The leftmost figure below shows the result of fitting a y = θ<sub>0</sub> + θ<sub>1</sub>x to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good. 

![example](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0cOOdKsMEeaCrQqTpeD5ng_2a806eb8d988461f716f4799915ab779_Screenshot-2016-11-15-00.23.30.png?expiry=1559174400000&hmac=ZWZSuLGo1Gn5R38JCgBPJOsb71kGNEyOtQrY5jI7HNY)

Instead, if we had added an extra feature x<sup>2</sup>, and fit y = θ<sub>0</sub> + θ<sub>1</sub>x + θ<sub>2</sub>x<sup>2</sup>, then we obtain a slightly better fit to the data (See middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5th order polynomial y = ∑ θ<sub>j</sub>x<sup>j</sup>. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for different living areas (x). 

Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of **underfitting**—in which the data clearly shows structure not captured by the model—and the figure on the right is an example of **overfitting**.

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. **There are two main options to address the issue of overfitting**:

1) **Reduce the number of features**:

- Manually select which features to keep.
- Use a model selection algorithm (studied later in the course).

2) **Regularization**

- Keep all the features, but reduce the magnitude of parameters θsub>j</sub>.
- Regularization works well when we have a lot of slightly useful features.

### II. Regularized Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:

θ<sub>0</sub> + θ<sub>1</sub>x + θ<sub>2</sub>x<sup>2</sup> + θ<sub>3</sub>x<sup>3</sup> + θ<sub>4</sub>x<sup>4</sup>

We'll want to eliminate the influence of θ<sub>3</sub>x<sup>3</sup> + θ<sub>4</sub>x<sup>4</sup>. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our **cost function**:

![w3.16](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.16.PNG?raw=true)

We've added two extra terms at the end to inflate the cost of θ<sub>3</sub> and θ<sub>4</sub>. Now, in order for the cost function to get close to zero, we will have to reduce the values of θ<sub>3</sub> and θ<sub>4</sub> to near zero. This will in turn greatly reduce the values of θ<sub>3</sub>x<sup>3</sup> and θ<sub>4</sub>x<sup>4</sup> in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms θ<sub>3</sub>x<sup>3</sup> and θ<sub>4</sub>x<sup>4</sup>.

![image](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/j0X9h6tUEeawbAp5ByfpEg_ea3e85af4056c56fa704547770da65a6_Screenshot-2016-11-15-08.53.32.png?expiry=1559174400000&hmac=b9rftZyUMhDEfRztTPgNA8vnl75JnvZ-fN28_XQo_Ms)

**We could also regularize all of our theta parameters in a single summation as**:

![w3.15](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.15.PNG?raw=true)

The **λ**, or lambda, **is the regularization parameter**. It balance (our two objectives) between fitting the training data and preventing overfitting.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. 

Q: What if lambda λ is set to an extremely large value (perhaps too large for our problem, say λ = 10<sup>10</sup>?

A: Algorithm results in underfitting (fails to fit even the training set).

### III. Regularized Linear Regression

![w3.17](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.17.PNG?raw=true)

1. **Gradient Descent**

previously: ![w3.18](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.18.PNG?raw=true)

We will modify our gradient descent function to separate out θ<sub>0</sub> from the rest of the parameters because we do not want to penalize θ<sub>0</sub>.

Updated euqation to calculate θ:

![w3.19](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.19.PNG?raw=true)

With some manipulation our update rule can also be represented as:

![w3.20](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.20.PNG?raw=true)

The first term in the above equation, 1 - α\*λ/m will always be less than 1. Intuitively you can see it as reducing the value of θ<sub>j</sub> by some amount on every update. Notice that the second term is now exactly the same as it was before.

2. **Normal Equation**

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

![w3.21](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.21.PNG?raw=true)

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including x<sub>0</sub>, multiplied with a single real number λ.

Regularization also take sure of non-invertiblity issue in that X<sup>T</sup> * X +  λ⋅L becomes invertible when we add the term λ⋅L.

### IV. Regularized Logistic Regression

1. **Gradient Descent**

Cost Function for Logistic Regression Before Regularization:

![w3.22](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.22.PNG?raw=true)

Regularized Logistic Regression Cost Function:

![w3.23](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.23.PNG?raw=true)

The second sum, ∑ θ<sup>2</sup><sub>j</sub> means to explicitly exclude the bias term, θ<sub>0</sub>. I.e. the θ vector is indexed from 0 to n (holding n+1 values, θ<sub>0</sub> through θ<sub>n</sub>), and this sum explicitly skips θ<sub>0</sub> by running from 1 to n, skipping 0. Thus, when computing the equation, we should continuously update the two following equations:

![Equation](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dfHLC70SEea4MxKdJPaTxA_306de28804a7467f7d84da0fe3ee9c7b_Screen-Shot-2016-12-07-at-10.49.02-PM.png?expiry=1559260800000&hmac=jIezpdugnAajCbOqi6O1zg4pWjaCOL8YW5C-41YXzrM)

2. **Advanced Optimization**

![w3.24](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W3/w3.24.PNG?raw=true)
