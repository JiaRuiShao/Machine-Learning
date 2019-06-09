## Support Vector Machine(SVM) Algorithm -- Separation of classes

SVMs are considered by many to be the most powerful 'black box' learning algorithm, and by posing a cleverly-chosen optimization objective, one of the most widely used learning algorithms today.

### 0. Introduction to SVM

	A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

Suppose you are given plot of two label classes on graph as shown in image below. Can you decide a separating line for the classes?

![A](https://cdn-images-1.medium.com/max/1600/1*BpeH5_M58kJ5xXfwzxI8yA.png)

You might have come up with something similar to following image. It fairly separates the two classes. Any point that is left of line falls into black circle class and on right falls into blue square class. **Separation of classes. That’s what SVM does**. It finds out a line/ hyper-plane (in multidimensional space that separate outs classes). 

![B](https://cdn-images-1.medium.com/max/1600/1*Sg6wjASoZHPphF10tcPZGg.png)

One more parameter is **kernel**. It **defines whether we want a linear of linear separation**. This is also discussed in next section.

Kernel can be divided by linear and polynomial kernel.

For **linear kernel** the equation for prediction for a new input using the dot product between the input (x) and each support vector (xi) is calculated as follows:

```
f(x) = B(0) + sum(ai * (x,xi))
```

This is an equation that involves calculating the inner products of a new input vector (x) with all support vectors in training data. The coefficients B0 and ai (for each input) must be estimated from the training data by the learning algorithm.

The **polynomial kernel** can be written as 

```
K(x,xi) = 1 + sum(x * xi)^d 
```

and exponential kernel as 

```
K(x,xi) = exp(-gamma * sum((x — xi²))
```

Polynomial and exponential kernels calculates separation line in higher dimension. This is called __kernel trick__.

### I. Large Margin Classification

1. **Optimization Objective**

Alternative view of logistic regression

![w7.1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.1.PNG?raw=true)

Support Vector Machine(SVM): an alternative to logistic regression

![w7.2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.2.PNG?raw=true)

- We replace the first and second terms of logistic regression with the respective cost functions

- We remove (1 / m) because it does not matter

- Instead of A + λB, we use CA + B
	+ Parameter C similar to the role (1 / λ)
	+ When C = (1 / λ), the two optimization equations would give same parameters θ

- Unlike logistic regression, we don't get probabilities, instead:
	+ We get a direct prediction of 1 or 0:
		* if Xθ >= 1, h<sub>θ</sub>(x) = 1
		* if Xθ < -1, h<sub>θ</sub>(x) = 0

![w7.4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.4.PNG?raw=true)

Q: Consider the following minimization problems:

![w7.3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.3.PNG?raw=true)

These two optimization problems will give the same value of θ (i.e., the same value of θ gives the optimal solution to both problems) if:

- [ ] C = λ
- [ ] C = -λ
- [X] C = 1/λ
- [ ] C = 2/λ

2. **Large Margin Classifer**

Sometimes people talk about support vector machines, as large margin classifiers.

Here's the cost function for the support vector machine:

![w7.5](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.5.PNG?raw=true)

SVM Decision Boundary -- large margin classifier

If C is huge, we would want A = 0 to minimize the cost function.

To make A equals to 0, when y<sup>(i)</sup> = 1, z = Xθ >= 1; when y<sup>(i)</sup> = 0, z = Xθ <= -1.

![SVM Decision Boundary](https://cdn-images-1.medium.com/max/1600/1*nUpw5agP-Vefm4Uinteq-A.png)

Note: margin is the distance between blue and black line

![w7.6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.6.PNG?raw=true)

If C is very large

- Decision boundary would change from black to magenta line

If C is not very large

- Decision boundary would be the black line

- SVM being a large margin classifier is only relevant when you have no outliers

Q: Consider the training set to the right, where "x" denotes positive examples (y=1) and "o" denotes negative examples (y=0). Suppose you train an SVM (which will predict 1 when θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> >= 0). What values might the SVM give for θ<sub>0</sub>, θ<sub>1</sub>, and θ<sub>2</sub>?

![Q](http://spark-public.s3.amazonaws.com/ml/images/12.2-quiz-1-q.png)

- θ<sub>0</sub> = -3, θ<sub>1</sub> = 1, and θ<sub>2</sub> = 0

3. **Mathematics of Large Margin Classification**

**Vector inner product**

![w7.7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.7.PNG?raw=true)

- u<sup>T</sup> \* v is also called inner product

- length of u = hypotenuse calculated using Pythagoras’ Theorem

If we project vector v on vector u (green line):

- p = length of vector v onto u
	+ p can be positive or negative
	+ p would be negative when angle between v and u more than 90
	+ p would be positive when angle between v and u is less than 90

- u<sup>T</sup> \* v = p \* ||u|| = u1 v1 + u2 v2 = v<sup>T</sup> \* v

**SVM decision boundary: projections**

To simplify, we assume features n to be 2 and θ<sub>0</sub> to be 0.

θ<sup>T</sup> \* x<sup>(i)</sup> = p<sup>(i)</sup> * ||θ|| = θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub>

![w7.8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.8.PNG?raw=true)

When θ0 = 0, this means the vector passes through the origin

θ projection will always be 90 degrees to the decision boundary

![w7.9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.9.PNG?raw=true)

**Decision boundary choice 1: graph on the left**

p<sub>1</sub> is projection of x<sub>1</sub> example on θ (red)

- p<sub>1</sub> \* ||θ|| >= 1

p<sub>2</sub> is projection of x<sub>2</sub> example on θ (magenta)

- p2 \* ||θ|| <= -1

For these to be true, ||θ|| has to be large

However, our purpose is to minimize ||θ||<sup>2</sup>. So this decision boundary choice does not appear to be suitable.

**Decision boundary choice2: graph on the right**

p<sub>1</sub> is projection of x<sub>1</sub> example on θ (red)

- p<sub>1</sub> is much bigger, and ||θ|| can be smaller

p<sub>2</sub> is projection of x<sub>2</sub> example on θ (magenta)

- p<sub>1</sub> is much larger, and ||θ|| can be smaller

SVM would end up with a large margin because it tries to maximize the margin to minimize the squared norm of θ, ||θ||<sup>2</sup>.

Q: ![w7.10](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.10.PNG?raw=true)

![Q](http://spark-public.s3.amazonaws.com/ml/images/12.3-quiz-1-q.png)

where p<sub>(i)</sub> is the (signed - positive or negative) projection of x<sub>(i)</sub> onto θ. Consider the training set above. At the optimal value of θ, what is ||θ||?

A: 1/2

### II. Kernels -- a powerful way to learn complex non-linear examples

1. **Non-linear decision boundary**

![w7.11](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.11.PNG?raw=true)

Given the data, is there a different or better choice of the features f1, f2, f3 … fn?

2. **Similarity function -- Gaussian kernel** 

![w7.12](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.12.PNG?raw=true)

We will manually pick 3 landmarks (points) l<sup>(1)</sup>, l<sup>(2)</sup> and l<sup>(3)</sup>.

Given an example x, we will define the features as a measure of similarity between x and the landmarks.

- f1 = similarity(x, l(1))
- f2 = similarity(x, l(2))
- f3 = similarity(x, l(3))

The different similarity functions are Gaussian Kernels and this kernel is often denoted as k(x, l(i)).

3. **Kernels and similarity**

compute the kernel with similarity between X and a landmark

![w7.13](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.13.PNG?raw=true)

![w7.14](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.14.PNG?raw=true)

As you increase sigma square, the value of the feature falls away much more slowly with moving away from l1.

![w7.15](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.15.PNG?raw=true)

Where to get l<sup>(1)</sup>, l<sup>(2)</sup> and l<sup>(3)</sup>?

![w7.16](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.16.PNG?raw=true)

![w7.17](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.17.PNG?raw=true)

When we solve the following optimization problem, we get the features

![w7.18](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.18.PNG?raw=true)

Notice that we do not regularize θ<sub>0</sub>, so it starts from θ<sub>1</sub>.

Choose SVM parameters C(=1/λ)：

![w7.19](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.19.PNG?raw=true)

### III. SVMs in Practice

We would normally use an SVM software package (liblinear, libsvm etc.) to solve for the parameters θ.

You need to specify the following:

- Choice of parameter C

- Choice of kernel (similarity function)
	+ 1. No kernel / "linear kernel”
		* Predict “y = 1” if θ_transpose * x >= 0
		* Use this when n is large & m is small
	+ 2. Gaussian kernel
		* For this kernel, we have to choose σ<sup>2</sup>
		* Use this when n is small and/or m is large

If you choose a Gaussian kernel:

![w7.20](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.20.PNG?raw=true)

Note: do feature scaling before using Gaussian kernel.

Other choices of kernel:

![w7.21](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.21.PNG?raw=true)

Multi-class classification:

![w7.22](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.22.PNG?raw=true)

**Logistic Regression vs SVMs**

n = \# of features, m = \# of training examples

- if n is large(relative to m): use logistic regression, or SVM without a kernel("linear kernel")

- if n is small, m is intermediate: use SVM with Gaussian kernal

- if n is small, m is large: add/create more features, then use logistic regression or SVM without a kernel("linear kernel")

Neural Network likely to work well for most of the settings, but may be slow to train.

The key thing to note is that if there is a huge number of training examples, a Gaussian kernel takes a long time.

The optimization problem of an SVM is a convex problem, so you will always find the global minimum while for neural networks, you may find local optima b/c it's non-convex.


Q: Suppose you are trying to decide among a few different choices of kernel and are also choosing parameters such as C, σ<sup>2</sup>, etc. How should you make the choice?

A: Choose whatever performs best on the cross-validation data.
