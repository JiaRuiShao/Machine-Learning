## Machine Learning

### 1 Introduction, Linear Regression with One Variable

We are going to start by covering linear regression with one variable. Linear regression predicts a real-valued output based on an input value. We discuss the application of linear regression to housing price prediction, present the notion of a cost function, and introduce the gradient descent method for learning.

We’ll also have optional lessons that provide a refresher on linear algebra concepts. Basic understanding of linear algebra is necessary for the rest of the course, especially as we begin to cover models with multiple variables.

### 2 Linear Regression with Multiple Variable, Matlab Tutorial

This week we’re covering linear regression with multiple variables. we’ll show how linear regression can be extended to accommodate multiple input features. We also discuss best practices for implementing linear regression.

We’re also going to go over how to use Octave. You’ll work on programming assignments designed to help you understand how to implement the learning algorithms in practice. To complete the programming assignments, you will need to use Octave or MATLAB.

[Note](https://github.com/JiaRuiShao/Machine-Learning/blob/master/2%20Linear%20Regression%20with%20Multiple%20Variables/README.md)

[Multiple Linear Regression Slides](https://github.com/JiaRuiShao/Machine-Learning/blob/master/2%20Linear%20Regression%20with%20Multiple%20Variables/Multivariate%20Linear%20Regression%20Slides.pdf)

[Matlab Tutorial Slides](https://github.com/JiaRuiShao/Machine-Learning/blob/master/2%20Linear%20Regression%20with%20Multiple%20Variables/Octave%2C%20Matlab%20Tutorial%20Slides.pdf)

[Exercise](https://github.com/JiaRuiShao/Machine-Learning/tree/master/2%20Linear%20Regression%20with%20Multiple%20Variables/Exercise/MATLAB)

### 3 Classification(Logistic Regression), Regularization

This week, we’ll be covering logistic regression. Logistic regression is a method for classifying data into discrete outcomes. For example, we might use logistic regression to classify an email as spam or not spam. In this module, we introduce the notion of classification, the cost function for logistic regression, and the application of logistic regression to multi-class classification.

We are also covering regularization. Machine learning models need to generalize well to new examples that the model has not seen in practice. We’ll introduce regularization, which helps prevent models from overfitting the training data.

[Note](https://github.com/JiaRuiShao/Machine-Learning/blob/master/3%20Classification(Logistic%20Regression)%20AND%20Regularization/README.md)

[Classification Slides](https://github.com/JiaRuiShao/Machine-Learning/blob/master/3%20Classification(Logistic%20Regression)%20AND%20Regularization/Classification(Logistic%20Regression)%20Slides.pdf)

[Regularization Slides](https://github.com/JiaRuiShao/Machine-Learning/blob/master/3%20Classification(Logistic%20Regression)%20AND%20Regularization/Regularization%20Slides.pdf)

[Exercise](https://github.com/JiaRuiShao/Machine-Learning/tree/master/3%20Classification(Logistic%20Regression)%20AND%20Regularization/Classification%20and%20Regularization%20Exercise/Exercise)

### 4 Neural Networks: Representation

This week, we are covering neural networks. Neural networks is a model inspired by how the brain works. It is widely used today in many applications: when your phone interprets and understand your voice commands, it is likely that a neural network is helping to understand your speech; when you cash a check, the machines that automatically read the digits also use neural networks.

[Note](https://github.com/JiaRuiShao/Machine-Learning/blob/master/4%20Neural%20Networks%20--%20Representation/README.md)

[Neural Networks -- Representation Slides](https://github.com/JiaRuiShao/Machine-Learning/blob/master/4%20Neural%20Networks%20--%20Representation/Neural%20Networks%20--%20Representation%20Slides.pdf)

[Exercise](https://github.com/JiaRuiShao/Machine-Learning/tree/master/4%20Neural%20Networks%20--%20Representation/Multi-class%20Classification%20AND%20Neural%20Networks%20Exercise)


### 5 Neural Networks: Learning

In Week 5, you will be learning how to train Neural Networks. The Neural Network is one of the most powerful learning algorithms (when a linear classifier doesn't work, this is what I usually turn to), and this week's videos explain the 'backpropagation' algorithm for training these models. In this week's programming assignment, you'll also get to implement this algorithm and see it work for yourself.

[Note](https://github.com/JiaRuiShao/Machine-Learning/blob/master/5%20Neural%20Networks%20--%20Learning/README.md)

[Neural Networks Slides](https://github.com/JiaRuiShao/Machine-Learning/blob/master/5%20Neural%20Networks%20--%20Learning/Neural%20Network%20Slides.pdf)

[Logistic Neural Network Exercise](https://github.com/JiaRuiShao/Machine-Learning/tree/master/5%20Neural%20Networks%20--%20Learning/Logistic%20Neural%20Network%20Exercise)

### 6 Advice for Applying Machine Learning, Mahine Learning System Design

In Week 6, you will be learning about systematically improving your learning algorithm. The videos for this week will teach you how to tell when a learning algorithm is doing poorly, and describe the 'best practices' for how to 'debug' your learning algorithm and go about improving its performance.

We will also be covering machine learning system design. To optimize a machine learning algorithm, you’ll need to first understand where the biggest improvements can be made. In these lessons, we discuss how to understand the performance of a machine learning system with multiple parts, and also how to deal with skewed data.

### 7 Support Vector Machines

This week, you will be learning about the support vector machine (SVM) algorithm. SVMs are considered by many to be the most powerful 'black box' learning algorithm, and by posing a cleverly-chosen optimization objective, one of the most widely used learning algorithms today.

### 8 Unsupervised Learning, Dimensionality Reduction

This week, you will be learning about unsupervised learning. While supervised learning algorithms need labeled examples (x,y), unsupervised learning algorithms need only the input (x). You will learn about clustering—which is used for market segmentation, text summarization, among many other applications.

We will also be introducing Principal Components Analysis, which is used to speed up learning algorithms, and is sometimes incredibly useful for visualizing and helping you to understand your data.

### 9 Anomaly Detection, Recommender Systems

This week, we will be covering anomaly detection which is widely used in fraud detection (e.g. ‘has this credit card been stolen?’). Given a large number of data points, we may sometimes want to figure out which ones vary significantly from the average. For example, in manufacturing, we may want to detect defects or anomalies. We show how a dataset can be modeled using a Gaussian distribution, and how the model can be used for anomaly detection.

We will also be covering recommender systems, which are used by companies like Amazon, Netflix and Apple to recommend products to their users. Recommender systems look at patterns of activities between different users and different products to produce these recommendations. In these lessons, we introduce recommender algorithms such as the collaborative filtering algorithm and low-rank matrix factorization.

### 10 Large Scale Machine Learning

This week, we will be covering large scale machine learning. Machine learning works best when there is an abundance of data to leverage for training. With the amount data that many websites/companies are gathering today, knowing how to handle ‘big data’ is one of the most sought after skills in Silicon Valley.

### 11 Application Example: Photo OCR

This week, we will walk you through a complex, end-to-end application of machine learning, to the application of Photo OCR. Identifying and recognizing objects, words, and digits in an image is a challenging task. We discuss how a pipeline can be built to tackle this problem and how to analyze and improve the performance of such a system.