# Unsupervised Learning

While supervised learning algorithms need labeled examples (x,y), unsupervised learning algorithms need only the input (x). You will learn about clustering—which is used for market segmentation, text summarization, among many other applications.

We will also be introducing Principal Components Analysis, which is used to speed up learning algorithms, and is sometimes incredibly useful for visualizing and helping you to understand your data.

## **I. Clustering**

### **Unsupervised Learning: Introduction**

Difference between supervised learning and unsupervised learning:

- supervised learning algorithms have labels
- unsupervised learning algorithms don't have labels

In unsupervised learning, you are given an unlabeled dataset and are asked to find "structure" in the data.

One main type of unsupervised learning algorithm: `clustering`

Applications of clustering:

- Market segmentation
- Social network analysis
- Organize computing clusters
- Astronomical data analysis
- etc

### **K-Means Algorithm**


**K-Means Algorithm** is by far the most popular and widely used clustering algorithm. K Means is an iterative algorithm and it does two things. First is a cluster assignment step, and second is a move centroid step. 

![K-Means Clustering](https://ds055uzetaobb.cloudfront.net/brioche/uploads/y4KGN92h7r-screen-shot-2016-05-05-at-43007-pm.png?width=1200)

**Procedure**:

1. randomly initialize k points(cluster centroids)

2. cluster assignment step: going through each of the examples and assign each of the data points to the cloest cluster centroid

3. move centroid step to the new means

4. repeat the previous two steps till reach convergence

![1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/1.png?raw=true)

### **Optimization Objective**

understanding what is the optimization objective (of k means) will help us to: 

1. debug the learning algorithm and make sure that (k-means) algorithm is running correctly

2. find better costs for this and avoid the local ultima

![2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/2.png?raw=true)

Question:

![3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/3.png?raw=true)

What does this mean?

[ ] The learning rate is too large.

[ ] The algorithm is working correctly.

[ ] The algorithm is working, but kk is too large.

[X] It is not possible for the cost function to sometimes increase. There must be a bug in the code.

### **Random Initialization**

![4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/4.png?raw=true)

1. cluster # K < training example # m

2. randomly pick K training examples

3. set u<sub>1</sub>, ..., u<sub>k</sub> equal to these examples

For small number of clusters(2<=K<=10), different random initialization could make a huge difference. 
Multiple random initializations can sometimes, help you find much better clustering of the data. 

![5](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/5.png?raw=true)


### **Choosing the Number of Clusters**

To be honest, there actually isn't a great way of answering this or doing this automatically and by far the most common way of choosing the number of clusters is still choosing it manually by looking at visualizations or by looking at the output of the clustering algorithm or something else.

A large part of why it might not always be easy to choose the number of clusters is that it is often generally ambiguous how many clusters there are in the data.

**Elbow method**:

![Elbow method](https://www.datanovia.com/en/wp-content/uploads/dn-tutorials/004-cluster-validation/figures/015-determining-the-optimal-number-of-clusters-k-means-optimal-clusters-wss-silhouette-1.png)

Very often people are running K-means to use for some later/downstream purpose. Evaluate k-means based on a metric for how well it performs for that later purpose.

Question:

Suppose you run k-means using k = 3 and k = 5. You find that the cost function J is much higher for k = 5 than for k = 3. What can you conclude?

[ ] This is mathematically impossible. There must be a bug in the code.

[ ] The correct number of clusters is k = 3.

[X] In the run with k = 5, k-means got stuck in a bad local minimum. You should try re-running k-means with multiple random initializations.

[ ] In the run with k = 3, k-means got lucky. You should try re-running k-means with k = 3 and different random initializations until it performs no better than with k = 5.

## **II. Dimensionality Reduction**

### **Motivation I: Data Compression**

A second type of unsupervised learning problem called **dimensionality reduction**

There are a couple of different reasons why one might want to do dimensionality reduction. One is **data compression**, and data compression not only allows us to compress the data and therefore use up less computer memory or disk space, but it will also allow us to speed up our learning algorithms.

**Reduce data from 2D to 1D**:

![6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/6.png?raw=true)

project all data onto a 1D line

**Reduce data from 3D to 2D**:

![7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/7.png?raw=true)

project all data onto a 2D plane


### **Motivation II: Data Visualization**

The second application of dimensionality reduction is to visualize the data. For a lot of machine learning applications, it really helps us to develop effective learning algorithms, if we can understand our data better by visualizing the data better. 

## **III. Principle Component Analysis(PCA)**

For the problem of dimensionality reduction, by far the most popular, by far the most commonly used algorithm is something called principle components analysis, or PCA.

### **Principal Component Analysis Problem Formulation**

What PCA does is it tries to find a vector(surface) onto which to project the data so as to minimize the projection error.

Reduce from n-dimension to k-dimensions: Find k vectors u<sup>(1)</sup>, u<sup>(2)</sup>, ..., u<sup>(k)</sup> onto which to project the data so as to minimize the projection error.

PS: before applying PCA, it's standard practice to first perform mean normalization at feature scaling so that the features should have zero mean, and should have comparable ranges of values. 

PCA is not linear regression!!!

- Minimize different things

![PCA vs LR](http://efavdb.com/wp-content/uploads/2018/06/pca_vs_linselect.jpg)

- In PCA, there is no special variable y that we're trying to predict

### **Principal Component Analysis Algorithm**

Before PCA, perform feature scaling/mean normalization first.

![8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/8.png?raw=true)

Then compute vectors u<sup>(1)</sup>, u<sup>(2)</sup>, ..., u<sup>(k)</sup>.

![9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/9.png?raw=true)

After that, compute new representations, the z1 and z2 of the data.

![11](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/11.png?raw=true)

Summary:

![12](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/12.png?raw=true)

## **IV. Applying PCA**

### **Reconstruction from Compressed Representation**

How to go back to the original high-dimensional data.

![13](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/13.png?raw=true)

Question:

![14](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/14.png?raw=true)

[X] U<sub>reduce</sup> will be an n×n matrix.

[X] x<sub>approx</sup> = x for every example x.

[X] The percentage of variance retained will be 100%.

[ ] The percentage of variance retained will always above 1.

### **Choosing the Number of Principal Components**

In the PCA algorithm we take N dimensional features and reduce them to some K dimensional feature representation. This number K is a parameter of the PCA algorithm. This number K is also called the number of principle components or the number of principle components that we've retained.

In order to choose k, that is to choose the number of principal components, here are a couple of useful concepts.

What PCA tries to do is it tries to minimize the `average squared projection error`(distance between x and it's projection onto that lower dimensional surface). 

The total variation in the data is the average of my training sets of the length of each of my training examples.

Typically, choose k to be smallest value so that:

```
the average squared projection error/the total variation <= 0.01 (1%)
```

Another way to say this to use the language of PCA is that 99% of the variance is retained.

![15](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/15.png?raw=true)

### **Advice for Applying PCA**

How PCA could hlep to speed up the learning algorithm.

Say you have a supervised learning problem with inputs X and labels Y, and xi are very high dimensional. 

Fortunately with PCA we'll be able to reduce the dimension of this data and so make our algorithms run more efficiently.

First extract just the inputs and temporarily put aside the Y's. Then we're going to apply PCA and this will give me a reduced dimension representation of the data. 

And we'll have a new sort of training example, which is Z1 paired with y1, up to ZM, YM.

Finally, we take this reduced dimension training set and feed it to a learning algorithm.

Note: what PCA does is it defines a mapping from x to z and this mapping from x to z should be defined by running PCA only on the training sets (not vaidation or testing set!!). Then apply that mapping to your cross-validation set and your test set.

![17](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/17.png?raw=true)

**Application of PCA:**

- Compressopn (choose k by % of variance retained)
	+ Reduce memory/disk needed to store data
	+ Speed up learning algorithm

- Visualization of high dimensional data (by choosing k=2,3)

**Bad use of PCA: to prevent overfitting**

![18](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W8/18.png?raw=true)

You might throw away some valuable information without knowing what the values of y are.

Before implementing PCA, first try running whatever you want to do with your original/raw data x<sup>(i)</sup>. Only if that doesn't do what you want, then implement PCA and consider using z<sup>(i)</sup>.

