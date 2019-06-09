## Support Vector Machine(SVM) Algorithm -- Separation of classes

Which points should influence optimality? 

+ All points? 
	+ Linear regression 
	+ Neural nets 

- Or only “difficult points” close to decision boundary?
	+ **Support vector machines**

SVMs are considered by many to be the most powerful 'black box' learning algorithm, and by posing a cleverly-chosen optimization objective, one of the most widely used learning algorithms today.

### 0. Introduction to SVM

What is Support Vector Machine?

	The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

_“Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However,  it is mostly used in classification problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well_

![SVM](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_1.png)

![Possible hyperplanes](https://cdn-images-1.medium.com/max/960/0*0o8xIA4k3gXUDCFU.png)

**Hyperplane**

To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. 

![Hyperplanes in 2D and 3D feature space
](https://cdn-images-1.medium.com/max/1280/1*ZpkLQf2FNfzfH4HXeMw4MQ.png)

**Hyperplanes are decision boundaries that help classify the data points**. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. It becomes difficult to imagine when the number of features exceeds 3.

![Support Vectors](https://cdn-images-1.medium.com/max/1280/0*ecA4Ls8kBYSM5nza.jpg)

**Support Vectors**

- Support vectors are the data points that lie closest to the decision boundary (or hyperplane) and influence the position and orientation of the hyperplane

- Support vectors are the critical elements of the training set 

- They are the data points most difficult to classify

- Moving a support vector moves the decision boundary
while moving the other vectors has no effect


Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane b/c the optimization algorithm to generate the weights proceeds in such a way that only the support vectors determine the weights and thus the boundary. Support Vectors are the points that help us build our SVM.

**Support Vector Machine (SVM)** finds an optimal solution.

- SVMs maximize the margin

- The decision function is fully specified by a (usually very small) subset of training samples, the support vectors. 

- 

**How can we identify the right hyper-plane?**

**Identify the right hyper-plane (Scenario-1)**: Here, we have three hyper-planes (A, B and C). Now, identify the right hyper-plane to classify star and circle.

![1](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_21.png)

**Identify the right hyper-plane (Scenario-2)**: Here, we have three hyper-planes (A, B and C) and all are segregating the classes well. Now, How can we identify the right hyper-plane?

![2](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_3.png) 

Here, maximizing the distances between nearest data point (either class) and hyper-plane will help us to decide the right hyper-plane. This distance is called as __Margin__.

![3](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_4.png)

Above, you can see that the margin for hyper-plane C is high as compared to both A and B. Hence, we name the right hyper-plane as C. Another lightning reason for selecting the hyper-plane with higher margin is robustness. If we select a hyper-plane having low margin then there is high chance of miss-classification.

**Identify the right hyper-plane (Scenario-3)**

![4](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_5.png)

Some of you may have selected the hyper-plane B as it has higher margin compared to A. But, here is the catch, SVM selects the hyper-plane which classifies the classes accurately prior to maximizing margin. Here, hyper-plane B has a classification error and A has classified all correctly. Therefore, the right hyper-plane is A.

**Find the hyper-plane to segregate to classes (Scenario-4)**: In the scenario below, we can’t have linear hyper-plane between the two classes, so how does SVM classify these two classes? Till now, we have only looked at the linear hyper-plane.

![5](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_8.png)

SVM can solve this problem. Easily! It solves this problem by introducing additional feature. Here, we will add a new feature z=x^2+y^2. Now, let’s plot the data points on axis x and z:

![6](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_9.png)

In SVM, it is easy to have a linear hyper-plane between these two classes. But, another burning question which arises is, should we need to add this feature manually to have a hyper-plane. No, SVM has a technique called the **kernel trick**. These are functions which takes low dimensional input space and transform it to a higher dimensional space i.e. it **converts not separable problem to separable problem, these functions are called kernels**. It is mostly useful in non-linear separation problem. Simply put, it does some extremely complex data transformations, then find out the process to separate the data based on the labels or outputs you’ve defined.

When we look at the hyper-plane in original input space it looks like a circle:

![7](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_10.png)

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

Large Margin Intuition

In logistic regression, we take the output of the linear function and squash the value within the range of [0,1] using the sigmoid function. If the squashed value is greater than a threshold value(0.5) we assign it a label 1, else we assign it a label 0. In SVM, we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values([-1,1]) which acts as margin.


Q: Consider the following minimization problems:

![w7.3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.3.PNG?raw=true)

These two optimization problems will give the same value of θ (i.e., the same value of θ gives the optimal solution to both problems) if:

- [ ] C = λ
- [ ] C = -λ
- [X] C = 1/λ
- [ ] C = 2/λ

2. **Large Margin Classifer**

In the SVM algorithm, we are looking to maximize the margin between the data points and the hyperplane. sO Sometimes people call support vector machines as large margin classifiers.The loss function that helps maximize the margin is the loss function J.

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

![w7.7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.7.png?raw=true)

- u<sup>T</sup> \* v is also called inner product

- length of u = hypotenuse calculated using Pythagoras’ Theorem

If we project vector v on vector u (green line):

- p = length of vector v onto u
	+ p can be positive or negative
	+ p would be negative when angle between v and u more than 90
	+ p would be positive when angle between v and u is less than 90

- u<sup>T</sup> \* v = p \* ||u|| = u1 v1 + u2 v2 = v<sup>T</sup> \* v

**SVM decision boundary: Hyperplanes**

To simplify, we assume features n to be 2 and θ<sub>0</sub> to be 0.

θ<sup>T</sup> \* x<sup>(i)</sup> = p<sup>(i)</sup> * ||θ|| = θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub>

![w7.8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.8.png?raw=true)

When θ0 = 0, this means the vector passes through the origin

θ projection will always be 90 degrees to the decision boundary.

![w7.9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.9.png?raw=true)

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

![w7.11](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.11.png?raw=true)

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

![w7.14](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.14.png?raw=true)

As you increase sigma square, the value of the feature falls away much more slowly with moving away from l1.

![w7.15](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.15.PNG?raw=true)

Where to get l<sup>(1)</sup>, l<sup>(2)</sup> and l<sup>(3)</sup>?

![w7.16](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.16.png?raw=true)

![w7.17](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.17.png?raw=true)

When we solve the following optimization problem, we get the features

![w7.18](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.18.png?raw=true)

Notice that we do not regularize θ<sub>0</sub>, so it starts from θ<sub>1</sub>.

Choose SVM parameters C(=1/λ)：

![w7.19](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.19.png?raw=true)

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

![w7.20](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.20.png?raw=true)

Note: do feature scaling before using Gaussian kernel.

Other choices of kernel:

![w7.21](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.21.png?raw=true)

Multi-class classification:

![w7.22](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W7/w7.22.png?raw=true)

**Pros and Cons associated with SVM**

**Pros**:

- Because it is quadratic, the surface is a paraboloid, with just a single global minimum
- It works really well with clear margin of separation
- It is effective in high dimensional spaces
- It is effective in cases where number of dimensions is greater than the number of samples
- It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient

**Cons**:

- It doesn’t perform well, when we have large data set because the required training time is higher
- It also doesn’t perform very well, when the data set has more noise i.e. target classes are overlapping
- SVM doesn’t directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. It is related SVC method of Python scikit-learn library

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
