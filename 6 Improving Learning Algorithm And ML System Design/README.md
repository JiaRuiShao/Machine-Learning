## Improving Learning Algorithm And ML System Design

### I. Improving learning algorithm

A. **Evaluating a Learning Algorithm**

1. **Deciding What to Try Next**

How to improve the performance of a machine learning system?

To explain this, let's continue using our example of learning to predict housing prices. 

Say you've implement and regularize linear regression to minimize cost function J. Now suppose that after you take your learn parameters, if you test your hypothesis on the new set of houses, suppose you find that this is making huge errors in this prediction of the housing prices.

The question is what should you then try mixing in order to improve the learning algorithm?

(1) Get more training examples (but sometimes it doesn't help)

(2) Try smaller sets of features

(3) Try getting additional features

(4) Try adding polynomial features

(5) Try decreasing λ

(6) Try increasing λ

Don't know which method to choose? 

Use **Machine Learning diagnostic**, which is a test that you can run to gain insight what is/isn't working with a learning algorithm, and gain guidance as to how to improve its performance.

- Diagnostics can give guidance as to what might be more fruitful things to try to improve a learning algorithm.

- Diagnostics can be time-consuming to implement and try, but they can still be a very good use of your time instead of just go with gut feeling and see what works

- A diagnostic can sometimes rule out certain courses of action (changes to your learning algorithm) as being unlikely to improve its performance significantly.

2. **Evaluating a Hypothesis**

hypothesis overfit:

![w6.1]()

How to tell?

- plot the hypothesis h of x 

- for problems with a large number of features, we need to use the following method

Randomly split the data into two portion, training set and test set (usually 70% 30% split)

![w6.2]()

Training/Testing procedure for linear regression

![Testing procedure for linear regression](https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w6_ml_design/test_lg.png)

Training/Testing procedure for logistic regression \-\- misclassification error

![Testing procedure for logistic regression](https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w6_ml_design/test_logrg.png)

3. **Model Selection and Train/Validation/Test Sets**

Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. It could over fit and as a result your predictions on the test set would be poor. The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set.

Given many models with different polynomial degrees, we can use a systematic approach to identify the 'best' function. In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

Note: d = degree of polynomial

minimize the training error, which would give you some parameter θ. Use these θ to measure the performance on the test set and see which model has the lowest test set error

![w6.4]()

Problem: the performance of the hypothesis on that test set may not be a fair estimate of how well the hypothesis is likely to do on examples we haven't seen before. So it isn't such a good idea to select your model using the test set and then using the same test set to report the error.

So we're going to **split the data set into three pieces: training set (60%), cross validation set (20%), test set (20%)**.

Training/validation/testing error

![w6.5]()

PS: validation error is just like the training error on the validation data set.

Now instead of measuring the performance of a model on the test set, we're going to test them on the validation set. Pick the hypothesis(model) with the lowest cross validation error and evaluate it on the 
test set.

![w6.6]()

Steps:

(1) Optimize the parameters in θ using the training set for each polynomial degree.

(2) Find the polynomial degree d with the least error using the cross validation set.

(3) Estimate the generalization error using the test set with J<sub>test</sub>(θ<sup>(d)</sup>), (d = theta from polynomial with lower error).

This way, the degree of the polynomial d has not been trained using the test set.


B. **Bias vs. Variance**

1. **Diagnosing Bias vs. Variance**

If you run a learning algorithm and it doesn't do as long as you are hoping, almost all the time, it will be because you have either a **high bias problem** or a **high variance problem**, in other words, either an underfitting problem or an overfitting problem.

- **high bias(underfitting)**: both J<sub>train</sub>(θ<sup>(d)</sup>) and J<sub>cv</sub>(θ<sup>(d)</sup>) will be high, J<sub>cv</sub>(θ<sup>(d)</sup>) ≈ J<sub>train</sub>(θ<sup>(d)</sup>)

- **high variance(overfitting)**: J<sub>train</sub>(θ<sup>(d)</sup>) will be low, and J<sub>cv</sub>(θ<sup>(d)</sup>) will be much greater than J<sub>train</sub>(θ<sup>(d)</sup>)

In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis.

![w6.7]()

![summary](https://www.learnopencv.com/wp-content/uploads/2017/02/Bias-Variance-Tradeoff-In-Machine-Learning-1.png)

The training error will tend to decrease as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to decrease as we increase d up to a point, and then it will increase as d is increased, forming a convex curve.

2. **Regularization and Bias/Variance**

We've known how regularization can help prevent over-fitting. But how does it affect the bias and variances of a learning algorithm? Here we'll go deeper into the issue of bias and variances and talk about how they interact with and are affected by the regularization of your learning algorithm.

![w6.9](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/3XyCytntEeataRJ74fuL6g_3b6c06d065d24e0bf8d557e59027e87a_Screenshot-2017-01-13-16.09.36.png?expiry=1559952000000&hmac=l8fiA3g68wMa-H7Uo3iWu8sRwqjl75iJeAAA8ItlWas)

![relationship betw λ and J(θ)](http://spark-public.s3.amazonaws.com/ml/images/10.5-quiz-1-option4.png)

- Large λ, high bias(underfit)
- Small λ, high variance(overfit)
- Intermediate λ, just alright

How to automatically choose a good value for the regularization parameter?

(1) Define  J<sub>train</sub>(θ), J<sub>cv</sub>(θ) and J<sub>test</sub>(θ) as cost function **without regularization**.

![w6.10]()

(2) Try some range of value of λ

![w6.11]()

(3) Choose the λ with the lowest cross validation error

![w6.12]()

(4) Apply the best combo θ and λ on J<sub>test</sub>(θ) to see if it has a good generalization of the problem

3. **Learning curves**

Learning curves is a tool that can diagnose if a physical learning algorithm may be suffering from bias, sort of variance problem or a bit of both.

![learning curve](https://image.slidesharecdn.com/practicalml-161021194209/95/practical-machine-learning-31-638.jpg?cb=1477079466)

To plot learning curve, plot J<sub>train</sub>(θ) and J<sub>cv</sub>(θ) as a function of m<sub>training</sub>

**Experiencing high bias**:

![learning curve -- high bias](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bpAOvt9uEeaQlg5FcsXQDA_ecad653e01ee824b231ff8b5df7208d9_2-am.png?expiry=1559952000000&hmac=lEZ_0ipQ7u1eAV3hYeAHX2KnthAWpqOhZltAs0wB22Q)

- Low training size: causes J<sub>train</sub>(θ) to be low, and J<sub>cv</sub>(θ) to be high

- Large training size: causes both J<sub>train</sub>(θ) and J<sub>cv</sub>(θ) to be high

If a learning algorithm is suffering from **high bias**, getting more training data will not help much (by itself).

**Experiencing high variance**:

![learning curve -- high variance](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/vqlG7t9uEeaizBK307J26A_3e3e9f42b5e3ce9e3466a0416c4368ee_ITu3antfEeam4BLcQYZr8Q_37fe6be97e7b0740d1871ba99d4c2ed9_300px-Learning1.png?expiry=1559952000000&hmac=Xf9C11X3VBoaZAqe6SWtBXi2LNBBMClmGPSXViiFVTs)

- Low training size: causes J<sub>train</sub>(θ) to be low, and J<sub>cv</sub>(θ) to be high

- Large training size: J<sub>train</sub>(θ) increases with training set size and J<sub>cv</sub>(θ) continues to decrease with leveling off. Also, J<sub>train</sub>(θ) < J<sub>cv</sub>(θ) but the difference between them remains significant.

If a learning algorithm is suffering from high variance, getting more training data will is likely to help.

**Summary**:

![summary](https://sebastianraschka.com/images/faq/ml-solvable/bias-variance.png)

4. **Debugging neural network learning algorithm**

Our decision process can be broken down as follows:

(1) Get more training examples -- doesn't works for high bias model

(2) Try smaller sets of features -- works for the high variance model

(3) Try getting additional features -- works for the high bias model

(4) Try adding polynomial features -- works for the high bias model

(5) Try decreasing λ -- works for the high bias model

(6) Try increasing λ -- works for the high variance model

Diagnosing Neural Networks

Small neural network vs. Large neural network

![w6.13]()

- A neural network with fewer parameters is **prone to underfitting**. It is also computationally cheaper.

- A large neural network with more parameters is **prone to overfitting**. It is also computationally expensive. In this case you can use regularization (increase λ) to address the overfitting.

Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

Model Complexity Effects:

- Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.

- Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.

- In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

Q: Suppose you fit a neural network with one hidden layer to a training set. You find that the cross validation error J<sub>cv</sub>(θ) is much larger than the training error J<sub>train</sub>(θ).  Is increasing the number of hidden units likely to help?

A: No, because it is currently suffering from high variance, so adding hidden units is unlikely to help.


### II. Machine Learning System Design

To optimize a machine learning algorithm, you’ll need to first understand where the biggest improvements can be made. In this module, we discuss how to understand the performance of a machine learning system with multiple parts, and also how to deal with skewed data.

A. **Example -- Building a Spam Classifier**

1. **Prioritizing What to Work On**

Supervised learning.

x = features of email

y = spam (1) or not spam(0)

Features x: Choose 100 words indicative of spam/not spam

x<sub>j</sub> = {
	1 if word j appears in the email
	0 otherwise
}

Note: In practice, take most frequently occurring n words (10,000 to 50,000) in training set, rather than manually pick 100 words.

Improvement methods:

- Collect more data (for example "honeypot" project but doesn't always work)
- Develop sophisticated features based on email routing information(from header)
- Develop sophisticated features for message body
- Develop sophisticated algorithm to detec misspellings

It is difficult to tell which of the options will be most helpful.

2. **Error Analysis**

Recommended approach:

- Start with a simple algorithm that you can implement quickly. Implement it and test it on your cross validation data

- Plot learning curves to decide if more data and features, etc are likely to help

- Error Analysis: Manually examine the examples in cross validation set that your algorithm made errors on. See if you can spot any systematic trend in what type of examples it is making errors on.

Spam Classifer Example:

m<sub>cv</sub> = 500 examples in cross validation set

Algorithm misclassifies 100 emails

Manually examine the 100 errors and categorize them based on: 

(1) what type of email it is

(2) what features you think would have helped the algorithm classify them correctly

3. **Numerical evaluation**

When developing learning algorithms, one other useful tip is to make sure that you have a numerical evaluation of your learning algorithm.

If you have a way of evaluating your learning algorithm that just gives you back a single real number, maybe accuracy, maybe error that tells you how well your learning algorithm is doing.

Example:

Let's say we're trying to decide whether or not we should treat words like discount, discounts, discounted, discounting as the same word.

In natural language processing, the way that this is done is actually using a type of software called **stemming software**. (search porter stemmer if you're interested)

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/kky-ouM6EeacbA6ydECl3A_01b1fa64fcc9a7eb5da8e946f6a12636_Screenshot-2017-01-25-12.08.23.png?expiry=1559952000000&hmac=2PaA0SDXBigH9_sMlucRY4KNjihc1fozUlM10ROb-_I)

For example if we use stemming, which is the process of treating the same word with different forms (fail/failing/failed) as one word (fail), and get a 3% error rate instead of 5%, then we should definitely add it to our model. However, if we try to distinguish between upper case and lower case letters and end up getting a 3.2% error rate instead of 3%, then we should avoid using this new feature. 

B. **Handling Skewed Data --Precision/Recall**

Skewed Classes: when the ratio of positive to negative examples is very close to one of two extremes. In this case, the number of positive examples is much, much smaller than the number of negative examples.

Example: Cancer Classification

Train logistic model h<sub>θ</sub>(x) (y=1 if cancer, y=0 if not)

Find that you got 1% error on test set (99% correct)

However, we found that only 0.5% of patients have cancer, which makes 99% correct rate not so impressive.

```matlab
function y = predictCancer(x)
	y = 0; % ignore x!
return
```

Let's say you have one joining algorithm that's getting 99.2% accuracy (0.8% error). Let's say you make a change to your algorithm and you now are getting 99.5% accuracy (0.5% error).

If you have very skewed classes it becomes much harder to use just classification accuracy, because you can get very high classification accuracies or very low errors, and it's not always clear if doing so is really improving the quality of your classifier.

When we're faced with such a skewed classes therefore we would want to come up with a different error metric or a different evaluation metric. One such evaluation metric are what's called **precision recall**.

**precision/recall**

![precision recall](https://cdn-images-1.medium.com/max/1600/1*pOtBHai4jFd-ujaNXPilRg.png)

![w6.14]()

Precision = true positive / predicted positive

Recall = true positive / actual positive

If a classifier has high precision and high recall, then we are confident that the algorithm is to be doing well, even if we have very skewed classes.

**Trading off between precision and recall**

![precision and recall](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNiZbsyvDfIouSR5W7L3oKFUaUH-wGbrBAt3heCT2lKMMpLoYB)

(1) Suppose we want to predict y = 1 (cancer) only when we're very confident 

Set higher threshold

Predict 1 if h<sub>θ</sub>(x) >= 0.7 instead of 0.5

Predict 0 if h<sub>θ</sub>(x) < 0.7

- higher precision
- lower recall

(2) Suppose we want to avoid missing too many cases of cancer (avoid false negatives)

Set lower threshold

Predict 1 if h<sub>θ</sub>(x) >= 0.3 instead of 0.5

Predict 0 if h<sub>θ</sub>(x) < 0.3

- higher recall
- lower precision

**F1 Score (F score)**

Average of precision and recall is not a good method to evaluate the performance of the learning algorithm.

![w6.15]()

Instead, we use F score.

F score = 2\*Precision\*Recall/(Precision+Recall)

C. **Using Large Data Sets**

![Comparison of several machine learning algorithms](https://www.researchgate.net/profile/Pavel_Kordik/publication/321987431/figure/fig4/AS:669621732536328@1536661670896/Comparison-of-several-machine-learning-algorithms-in-H2Oai-trained-on-samples-with.png)

Comparison of several machine learning algorithms:

- Preception (logistic regression)
- Winnow
- Memory-based
- Navive Bayes
- KNN
- Decision Trees

In machine learning that it's not who has the best algorithm that wins, it's who has the most data.

**Large data rationale**

Using a learning algorithm with many parameters to avoid high bias(underfitting)

- logistic regression/linear regression with many features
- neural network with many hidden units

Using a very large training set to avoid high variance(overfitting)

If you can do both, that will give you a high performance of your learning algorithm.