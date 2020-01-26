# Anomaly Detection & Recommender Systems

## I. **Anomaly Detection**

Given a large number of data points, we may sometimes want to figure out which ones vary significantly from the average. For example, in manufacturing, we may want to detect defects or anomalies. We show how a dataset can be modeled using a Gaussian distribution, and how the model can be used for anomaly detection.

### 1. **Density Estimation**


#### **Problem Motivation**

Given the training set, build a new model p(x). Use the p(x) to detect unusual data(anomaly)

![1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/1.png?raw=true)

Example: 

- Fraud detection
- Manufacturing
- Monitoring computers in a data center

#### **Gaussian/Normal Distribution**

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/340px-Normal_Distribution_PDF.svg.png)

E(X) = μ

Var(X) = σ<sup>2</sup>

σ standard deviation

![2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/2.png?raw=true)

![](http://www.visiondummy.com/wp-content/uploads/2014/03/gaussiandensity.png)

#### **Anomaly Detection Algorithm**

Density estimation

![3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/3.png?raw=true)

![4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/4.png?raw=true)

### 2. **Building an Anomaly Detection System**

Unsupervised machine learning algorithms, however, learn what normal is, and then apply a statistical test to determine if a specific data point is an anomaly. A system based on this kind of anomaly detection technique is able to detect any type of anomaly, including ones which have never been seen before.

Once the anomalies are found by these interacting machine learning algorithms, a whole other layer of machine learning – one that utilizes deep neural networks, among other clustering and similarity algorithms – works to discover the relationships between metrics so that the flood of discovered anomalies can be distilled down to a much more manageable number of correlated incidents, which can then be investigated by human experts.

By filtering out the massive amount of insurmountable data and pinpointing the issues at hand, we can extract actionable insights effortlessly from the anomalies, which empowers us to turn issues into opportunities and errors into learning curves.

#### **Developing and Evaluating an Anomaly Detection System**

The importance of real-number evaluation:

When developing a learning algorithm (choosing features, etc.), making decisions is much easier if we have a way of evaluating our learning algorithm.

Assume we have some labeled data, of anomalous and nonanomalous examples. (y = 0 if normal, y = 1 if anomalous).

Training set: x<sup>(1)</sup>, x<sup>(2)</sup>, ..., x<sup>(m)</sup> (assume normal examples/not anomalous)

Cross validation set: (x<sub>cv</sub><sup>(1)</sup>, y<sub>cv</sub><sup>(1)</sup>), ..., (x<sub>cv</sub><sup>(m<sub>cv</sub>)</sup>, y<sub>cv</sub><sup>(m<sub>cv</sub>)</sup>)

Test set: (x<sub>test</sub><sup>(1)</sup>, y<sub>test</sub><sup>(1)</sup>), ..., (x<sub>test</sub><sup>(m<sub>test</sub>)</sup>, y<sub>test</sub><sup>(m<sub>test</sub>)</sup>)

![5](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/5.png?raw=true)

pick the value of epsilon ε that maximizes f1 score, or that otherwise does well on your cross validation set

Question:

![6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/6.png?raw=true)

#### **Anomaly Detection vs. Supervised Learning**

When to use anomaly detection versus supervised learning:

![7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/7.png?raw=true)

In many anomaly detection applications, we have very few positive(abnormal) examples and lots of negative(normal) examples. And when we're doing the process of estimating p(x), affecting all those Gaussian parameters, we need only negative examples to do that. So if you have a lot negative data, we can still fit p(x) pretty well. 

In contrast, for supervised learning, more typically we would have a reasonably large number of both positive(abnormal) and negative(normal) examples. 

A key difference really is that in anomaly detection, often we have such a small number of positive examples that it is not possible for a learning algorithm to learn that much from the positive examples. And so what we do instead is take a large set of negative examples and have it just learn a lot, learn p(x) from just the negative examples.

Applications:

![8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/8.png?raw=true)

Question:

Which of the following problems would you approach with an anomaly detection algorithm (rather than a supervised learning algorithm)? Check all that apply.

[X] You run a power utility (supplying electricity to customers) and want to monitor your electric plants to see if any one of them might be behaving strangely.

[ ] You run a power utility and want to predict tomorrow’s expected demand for electricity (so that you can plan to ramp up an appropriate amount of generation capacity).

[X] A computer vision / security application, where you examine video images to see if anyone in your company’s parking lot is acting in an unusual way.

[ ] A computer vision application, where you examine an image of a person entering your retail store to determine if the person is male or female.

#### **Choosing What Features to Use**

When you're applying anomaly detection, one of the things that has a huge effect on how well the model does is what features you choose to use to give the anomaly detection algorithm.

Deal with non-gaussian features:

![9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/9.png?raw=true)

Try X.^0.2, X.^0.1, X.^0.05, log(X) to to make your data more gaussian.

Error analysis Procedure:

![10](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/10.png?raw=true)

Look at the anomaly that the algorithm is failing to flag, and see if that inspires you to create some new feature, with this new feature it becomes easier to distinguish the anomalies from your good examples. 

Question:

Suppose your anomaly detection algorithm is performing poorly and outputs a large value of p(x) for many normal examples and for many anomalous examples in your cross validation dataset. Which of the following changes to your algorithm is most likely to help?

[ ] Try using fewer features.


[X] Try coming up with more features to distinguish between the normal and the anomalous examples.

[ ] Get a larger training set (of normal examples) with which to fit p(x).

[ ] Try changing ϵ.

### 3. **Multivariate Gaussian Distribution (Optional)**

#### **Multivariate Gaussian/Normal Distribution**

![11](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/11.png?raw=true)

An anomaly detection algorithm will fail to flag this green point as an anomaly. 

In order to fix this problem, we could use Multivariate Gaussian Distribution

![12](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/12.png?raw=true)

![13](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/13.png?raw=true)

![14](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/14.png?raw=true)

![15](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/15.png?raw=true)

![16](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/16.png?raw=true)

![17](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/17.png?raw=true)

#### **Anomaly Detection using the Multivariate Gaussian Distribution**

![18](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/18.png?raw=true)

Procedure:

![19](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/19.png?raw=true)

Relationship of Multivariate Gaussian Distribution and original Gaussian Distribution

In original Gaussian Distribution, the p(x) does not have angle when ploted.

![20](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/20.png?raw=true)

Comparasion:

![21](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/21.png?raw=true)

Reasonable rule of thumb of using Multivariate Gaussian Model: m >= 10n

## II. Recommender Systems

When you buy a product online, most websites automatically recommend other products that you may like. Recommender systems look at patterns of activities between different users and different products to produce these recommendations. In this module, we introduce recommender algorithms such as the collaborative filtering algorithm and low-rank matrix factorization.

### 1. **Predicting Movie Ratings**


#### **Problem Formulation**

Example: Predicting movie rating: user rates movies using one to five stars

n<sub>u</sub> = # of users
n<sub>m</sub> = # of movies
r(i,j) = 1 if user j has rated movie i
y(i,j) = rating given by user j to movie i (defined only if r(i,j) = 1)

Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4)
---|---|---|---|---
A | 5 | 5 | 0 | 0
B | 5 | ? | ? | 0
C | ? | 4 | 0 | ?
D | 0 | 0 | 5 | 4
E | 0 | 0 | 5 | ?

In this case, n<sub>u</sub> = 4, n<sub>m</sub> = 5. Build a recommendation system to try predict the missing ratings values given r(i,j) and y(i,j).

#### **Content Based Recommendations**

The first approach to build a recommender system which is called **content based recommendations**.

Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) | x1(romance) | x2(action)
---|---|---|---|---|---|---
A | 5 | 5 | 0 | 0 | 0.9 | 0
B | 5 | ? | ? | 0 | 1.0 | 0.01
C | ? | 4 | 0 | ? | 0.99 | 0
D | 0 | 0 | 5 | 4 | 0.1 | 1.0
E | 0 | 0 | 5 | ? | 0 | 0.9

feature vectors: x<sup>(A)</sup> = [1; 0.9; 0]
n<sub>f</sub> = 2

![22](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/22.png?raw=true)

![23](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/23.png?raw=true)

Optimization objective:

![24](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/24.png?raw=true)

![25](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/25.png?raw=true)

### 2. **Collaborative Filtering**

Another approach to building a recommender system that's called collaborative filtering that can start to learn for itself what features to use.

#### **Collaborative Filtering**

Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) | x1(romance) | x2(action)
---|---|---|---|---|---|---
A | 5 | 5 | 0 | 0 | ? | ?
B | 5 | ? | ? | 0 | ? | ?
C | ? | 4 | 0 | ? | ? | ?
D | 0 | 0 | 5 | 4 | ? | ?
E | 0 | 0 | 5 | ? | ? | ?

![26](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/26.png?raw=true)

Question:

![27](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/27.png?raw=true)

Answer: 0.5

[0; 3].T * [1; ?] = 1.5
[0; 5].T * [1; ?] = 2.5

? = 0.5

Formalize this algorithm:

![28](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/28.png?raw=true)

![29](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/29.png?raw=true)

#### **Collaborative Filtering Algorithm**

Collaborative Filtering Optimization Objective:

![30](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/30.png?raw=true)

Process:

![31](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/31.png?raw=true)


### 2. **Low Rank Matrix Factorization**

#### **Vectorization: Low Rank Matrix Factorization**

![32](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/32.png?raw=true)

![33](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/33.png?raw=true)

Finding related movies:

![34](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/34.png?raw=true)

#### **Implementational Detail: Mean Normalization**

If a user doesn't have any movie rating, then according to our algprithm, all predicted movies ratings for that user are zeros.

![35](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/35.png?raw=true)

To avoid that, use mean normalization:

![36](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W9/36.png?raw=true)

Questions:

Q2: In which of the following situations will a collaborative filtering system be the most appropriate learning algorithm (compared to linear or logistic regression)?

[ ] You manage an online bookstore and you have the book ratings from many users. You want to learn to predict the expected sales volume (number of books sold) as a function of the average rating of a book.

[ ] You're an artist and hand-paint portraits for your clients. Each client gets a different portrait (of themselves) and gives you 1-5 star rating feedback, and each client purchases at most 1 portrait. You'd like to predict what rating your next customer will give you.

[X] You run an online bookstore and collect the ratings of many users. You want to use this to identify what books are "similar" to each other (i.e., if one user likes a certain book, what are other books that she might also like?)

[X] You own a clothing store that sells many styles and brands of jeans. You have collected reviews of the different styles and brands from frequent shoppers, and you want to use these reviews to offer those shoppers discounts on the jeans you think they are most likely to purchase

Q3: You run a movie empire, and want to build a movie recommendation system based on collaborative filtering. There were three popular review websites (which we'll call A, B and C) which users to go to rate movies, and you have just acquired all three companies that run these websites. You'd like to merge the three companies' datasets together to build a single/unified system. On website A, users rank a movie as having 1 through 5 stars. On website B, users rank on a scale of 1 - 10, and decimal values (e.g., 7.5) are allowed. On website C, the ratings are from 1 to 100. You also have enough information to identify users/movies on one website with users/movies on a different website. 

**Which of the following statements is true?**

[ ] You can combine all three training sets into one as long as your perform mean normalization and feature scaling after you merge the data.

[ ] It is not possible to combine these websites' data. You must build three separate recommendation systems.

[X] You can merge the three datasets into one, but you should first normalize each dataset's ratings (say rescale each dataset's ratings to a 0-1 range).

[ ]  Assuming that there is at least one movie/user in one database that doesn't also appear in a second database, there is no sound way to merge the datasets, because of the missing data.

Q4: Which of the following are true of collaborative filtering systems? Check all that apply.

[ ] For collaborative filtering, the optimization algorithm you should use is gradient descent. In particular, you cannot use more advanced optimization algorithms (L-BFGS/conjugate gradient/etc.) for collaborative filtering, since you have to solve for both the x(i)'s and Theta(j)'s simultaneously.

[X] For collaborative filtering, it is possible to use one of the advanced optimization algoirthms (L-BFGS/conjugate gradient/etc.) to solve for both the x(i)'s and Theta(j)'s simultaneously.

[ ] Suppose you are writing a recommender system to predict a user's book preferences. In order to build such a system, you need that user to rate all the other books in your training set.

[X] Even if each user has rated only a small fraction of all of your products (so r(i,j)=0 for the vast majority of (i,j) pairs), you can still build a recommender system by using collaborative filtering.

Q5: Suppose you have two matrices A and B, where A is 5x3 and B is 3x5. Their product is C=AB, a 5x5 matrix. Furthermore, you have a 5x5 matrix R where every entry is 0 or 1. 
You want to find the sum of all elements C(i,j) for which the corresponding R(i,j) is 1, and ignore all elements C(i,j) where R(i,j)=0. 

One way to do so is the following code:

```
C = A * B;
total = 0;
for i = 1:5
  for j = 1:5
    if(R(i, j) == 1)
	  total = total + C(i, j);
	end
  end
end
```

Which of the following pieces of Octave code will also correctly compute this total? Check all that apply. Assume all options are in code.

[X] total = sum(sum((A * B) .* R))

[X] C = A * B; total = sum(sum(C(R == 1)))

[ ] C = (A * B) * R; total = sum(C(:))

[ ] total = sum(sum(A(R == 1) * B(R == 1))
