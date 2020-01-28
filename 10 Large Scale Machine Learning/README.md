# Large Scale Machine Learning

Machine learning works best when there is an abundance of data to leverage for training. In this module, we discuss how to apply the machine learning algorithms with large datasets.

## **I. Gradient Descent with Large Datasets**

### **Learning With Large Datasets**

Q: Suppose you are facing a supervised learning problem and have a very large dataset (m = 100,000,000). How can you tell if using all of the data is likely to perform much better than using a small subset of the data (say m = 1,000)?

[ ] There is no need to verify this; using a larger dataset always gives much better performance.

[ ] Plot J<sub>train</sub>(θ) as a function of the number of iterations of the optimization algorithm (such as gradient descent).

[ ] Plot a learning curve J<sub>train</sub>(θ) and J<sub>cv</sub>(θ), plotted as a function of m) for some range of values of m (say up to m = 1,000) and verify that the algorithm has bias when m is small.

[X] Plot a learning curve for a range of values of m and verify that the algorithm has high variance when m is small.

![1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/1.png?raw=true)

### **Stochastic Gradient Descent**

A modification to the basic gradient descent algorithm -- Stochastic gradient descent, which will allow us to scale these algorithms to much bigger training sets

Batch gradient descent takes a long time for one gradient iteration, and get the algorithm to converge.

In contrast to Batch gradient descent, what we are going to do is come up with a different algorithm called Stochastic gradient descent that scales better to large data sets because it doesn't need to look at all the training examples in every single iteration, only needs to look at a single training example in one iteration. 

![2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/2.png?raw=true)

![3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/3.png?raw=true)

![4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/4.png?raw=true)

Q: Which of the following statements about stochastic gradient descent are true? Check all that apply.

[X] When the training set size m is very large, stochastic gradient descent can be much faster than gradient descent.

[X] The cost function J<sub>train</sub>(θ) should go down with every iteration of batch gradient descent (assuming a well-tuned learning rate \alphaα) but not necessarily with stochastic gradient descent.

[ ] Stochastic gradient descent is applicable only to linear regression but not to other models (such as logistic regression or neural networks).

[X] Before beginning the main loop of stochastic gradient descent, it is a good idea to "shuffle" your training data into a random order.

### **Mini-Batch Gradient Descent**

![5](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/5.png?raw=true)

![6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/6.png?raw=true)

Compared to batch gradient descent, mini-batch gradient descent also allows us to make progress much faster. 

We can start making progress in modifying the parameters after looking at just ten examples rather than needing to wait 'till you've scan through every single training example of 300 million of them. 

So, why do we want to look at b examples at a time rather than look at just a single example at a time as the Stochastic gradient descent? The answer is in vectorization. 

In particular, Mini-batch gradient descent is likely to outperform Stochastic gradient descent **only if you have a good vectorized implementation**. In that case, the sum over 10 examples can be performed in a more vectorized way which will allow you to partially parallelize your computation over the ten examples.

Q: Suppose you use mini-batch gradient descent on a training set of size m, and you use a mini-batch size of b. The algorithm becomes the same as batch gradient descent if:

A: b = m

### **Stochastic Gradient Descent Convergence**

Some techniques for making sure the algorithm is converging and for picking the learning rate alpha.

Check for convergence:

![7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/7.png?raw=true)

![8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/8.png?raw=true)

Learning rate:

Learning rate alpha is typically held constant. Can slowly decrease overtime if we want theta to converge. (Eg: alpha = const1/(iteration# + const2))

Q: Which of the following statements about stochastic gradient descent are true? Check all that apply.

[ ] Picking a learning rate α that is very small has no disadvantage and can only speed up learning.

[X] If we reduce the learning rate \alphaα (and run stochastic gradient descent long enough), it’s possible that we may find a set of better parameters than with larger \alphaα.

[ ] If we want stochastic gradient descent to converge to a (local) minimum rather than wander of "oscillate" around it, we should slowly increase \alphaα over time.

[X] If we plot J cost (averaged over the last 1000 examples) and stochastic gradient descent does not seem to be reducing the cost, one possible problem may be that the learning rate \alphaα is poorly tuned.

## **II. Advanced Topics**

### **Online Learning**

Online learning setting is a new large-scale machine learning setting. The online learning setting allows us to model problems where we have a continuous flood or a continuous stream of data coming in and we would like an algorithm to learn from that.

![9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/9.png?raw=true)

If you really have a continuous stream of data, then an online learning algorithm can be very effective.

And in particular, if over time because of changes in the economy maybe users start to become more price sensitive and willing to pay, you know, less willing to pay high prices. Or if they become less price sensitive and they're willing to pay higher prices. Or if different things become more important to users, if you start to have new types of users coming to your website. This sort of online learning algorithm can also adapt to changing user preferences and kind of keep track of what your changing population of users may be willing to pay for. 

Other examples: predicted click through rate (CTR), customized selection of new articles, product recommendation;...

### **Map Reduce and Data Parallelism**

Map reduce approach is a different approach to large scale machine learning.

![10](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/10.png?raw=true)

Map-reduce can be used when the learning algotithm can be expressed as sum of functions over the training set.

![11](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W10/11.png?raw=true)

Q: Suppose you apply the map-reduce method to train a neural network on ten machines. In each iteration, what will each of the machines do?

[ ] Compute either forward propagation or back propagation on 1/5 of the data.

[X] Compute forward propagation and back propagation on 1/10 of the data to compute the derivative with respect to that 1/10 of the data.

[ ] Compute only forward propagation on 1/10 of the data. (The centralized machine then performs back propagation on all the data).

[ ] Compute back propagation on 1/10 of the data (after the centralized machine has computed forward propagation on all of the data).
