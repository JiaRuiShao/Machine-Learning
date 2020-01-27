## Neural Networks: Representation

Neural networks is a model inspired by how the brain works. It is widely used today in many applications: when your phone interprets and understand your voice commands, it is likely that a neural network is helping to understand your speech; when you cash a check, the machines that automatically read the digits also use neural networks.

### I. Motivations

1. **Non-linear Hypotheses**

It's computationally expensive if you want to use the Non-linear Hypotheses(quadratic or cubic function) when there're many features.

2. **Neurons and the Brain**

Neural Networks

+ Origin: Algorithms that try to mimic the brain
+ Was very widely used in 80s and early 90s; 
+ Recent resurgence: State-of-the-art technique for many applications.

The "one learning algorithm" hypothesis

+ Auditory cortex learns to see
+ Somatosensory cortex learns to see

### II. Neural Networks

1. **Model Representation**

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons).

![neuron in the brain](https://d2jmvrsizmvf4x.cloudfront.net/gUTyOY9SJ2WX4BmLrSvF_the_neuron-14BBC6B30D64FABB7A6+%281%29.png)

![neuron in the brain](https://upload.wikimedia.org/wikipedia/commons/4/44/Neuron3.png)

In our model, our dendrites are like the input features x<sub>1</sub> ... x<sub>n</sub>, and the output is the result of our hypothesis function. 

![Neuron Model: Logistic Unit](http://img.blog.csdn.net/20160312223541214)

Terms:

We call θ the parameters or the weights, x<sub>0</sub> input node as bias unit, which always equal to 1.

![layers](http://img.blog.csdn.net/20160312223725767)

Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

a<sub>i</sub><sup>(j)</sup> = "activation" of unit i in layer j

If network has s<sub>j</sub> units in layer j, s<sub>j+1</sub> units in layer j+1, then θ<sup>(j)</sup> will be of dimension s<sub>j+1</sub> x (s<sub>j</sub> + 1)

This first hidden unit and the output unit here are computed as follows:

There's a is a<sub>1</sub><sup>(2)</sup> is equal to the sigmoid function of the sigmoid activation function, also called the logistics activation function, apply to this sort of linear combination of these inputs. And then this second hidden unit has this activation value computer as sigmoid of this. And similarly for this third hidden unit is computed by that formula.

![elborate](http://img.blog.csdn.net/20160312224012891)

![theta](http://img.blog.csdn.net/20160312224116595)

Q: Consider the following neural network:

![Q](http://spark-public.s3.amazonaws.com/ml/images/8.3-quiz-1-q.png)

What is the dimension of θ<sup>(1)</sup>

s<sub>1</sub>=2, s<sub>2</sub>=4,dimension of θ<sup>(1)</sup>=s<sub>2</sub>×(s<sub>1</sub>+1) = 4 × 3

2. **Model Representation II**

Forward propagation: Vectorized implementation 

![neural network](http://img.blog.csdn.net/20160313083341366)

![a1, a2, a3](http://img.blog.csdn.net/20160313083353726)

![a1, a2, a3](http://img.blog.csdn.net/20160313083540034)

This process of computing the activations from the input then the hidden then the output layer is called forward propagation.

Neural network is just like logistic regression using hidden units a<sup>(1)</sup>,..., a<sup>(l)</sup>.

Other neural network architectures

![](https://cv-tricks.com/wp-content/uploads/2018/09/Neural-Network-Architecures.jpg)

![](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

![](https://upload.wikimedia.org/wikipedia/commons/3/30/Multilayer_Neural_Network.png)

### III. Simple Non-linear Classification Examples

#### Binary Classification

X<sub>1</sub> and X<sub>2</sub> are binary (0 or 1).

![](http://img.blog.csdn.net/20160313092435809)

y = x<sub>1</sub> XOR x<sub>2</sub>

x<sub>1</sub> XNOR x<sub>2</sub> = NOT(x<sub>1</sub> XOR x<sub>2</sub>)

1. **Example: AND**

In order to build up to a network that fits the XNOR example we're going to start with a slightly simpler one and show a network that fits the AND function.

x<sub>1</sub>, x<sub>2</sub>∈{0,1}

y = x<sub>1</sub> AND x<sub>2</sub>

![](http://img.blog.csdn.net/20160313092545684)

H<sub>θ</sub>(x) = g(-30 + 20x<sub>1</sub> + 20x<sub>2</sub>)

Let's look at the four possible input values for x1 and x2 and look at what the hypotheses will output in that case.

![](http://img.blog.csdn.net/20160313092714107)

2. **Example: OR**

This network showed here computes the OR function.

![](http://img.blog.csdn.net/20160313092758461)

H<sub>θ</sub>(x) = g(-10 + 20x<sub>1</sub> + 20x<sub>2</sub>)

![output](http://img.blog.csdn.net/20160313092826638)

3. **Example: (NOT x<sub>1</sub>) AND (NOT x<sub>2</sub>)**

![](http://spark-public.s3.amazonaws.com/ml/images/8.6-quiz-1-option1.png)

4. **Putting it together: x<sub>1</sub> XNOR x<sub>2</sub>**

The weights matrices for AND, NOR, and OR are:

```
AND: θ<sub>1</sub> = [-30 20 20]
NOR: θ<sub>1</sub> = [10 -20 -20]
OR: θ<sub>1</sub> = [-10 20 20]
```

We can combine these to get the XNOR logical operator (which gives 1 if x<sub>1</sub> and x<sub>2</sub> are both 0 or 1).

For the transition between the first and second layer, we'll use a θ<sub>1</sub> matrix that combines the values for AND and NOR:

```
θ1 = [-30 20 20]
				[10 -20 -20]
```

For the transition between the second and third layer, we'll use a θ<sub>2</sub> matrix that uses the values for OR:

```
θ<sub>1</sub> = [-10 20 20]
```

The values for all our nodes:

```
a(2) = g(θ(1)⋅x)
a(3) = g(θ(1)⋅a(2))
hθ(x) = a(3)
```

![](http://img.blog.csdn.net/20160313093144562)

![](http://img.blog.csdn.net/20160313093320499)

Example: [Handwritten digit classification](https://youtu.be/yxuRnBEczUU)

#### Multiclass Classification

![](http://img.blog.csdn.net/20160313190410588)

![](http://img.blog.csdn.net/20160313190421807)

Q: Suppose you have a multi-class classification problem with 10 classes. Your neural network has 3 layers, and the hidden layer (layer 2) has 5 units. Using the one-vs-all method described here, how many elements does θ<sup>(2)</sup> have?

A: 10 x 6 = 60. Recall that if network has s<sub>j</sub> units in layer j, s<sub>j+1</sub> units in layer j+1, then θ<sup>(j)</sup> will be of dimension s<sub>j+1</sub> x (s<sub>j</sub> + 1)
