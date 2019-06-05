## 5 Neural Networks -- Learning

### I. Cost Function and Backpropagation

1. **Cost Function**

Let's first define a few variables that we will need to use:

- L = total number of layers in the network
- s<sub>l</sub> = number of units (not counting bias unit) in layer l
- K = number of output units/classes

Recall that the cost function for regularized logistic regression was:

![w5.1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W5/w5.1.PNG?raw=true)

For neural networks, it is going to be slightly more complicated:

![w5.2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W5/w5.2.PNG?raw=true)

Note:

- the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
- the triple sum simply adds up the squares of all the individual θs in the entire network.
- the i in the triple sum does not refer to training example i

2. **Backpropagation Algorithm**

"Backpropagation" is neural-network terminology for minimizing our cost function.

In order to minimize our cost function J(θ) we need to compute:

- J(θ)
- ∂J(θ)/∂θ<sub>i</sub><sub>j</sub><sup>(l)</sup>

(1) **Suppose we only have one training example**:

a<sup>(1)</sup> = x

z<sup>(2)</sup> = θ<sup>(1)</sup> * a<sup>(1)</sup>

a<sup>(2)</sup> = g(z<sup>(2)</sup>) [add a<sub>0</sub><sup>(2)</sup>]

z<sup>(3)</sup> = θ<sup>(2)</sup> * a<sup>(2)</sup>

a<sup>(3)</sup> = g(z<sup>(3)</sup>) [add a<sub>0</sub><sup>(3)</sup>]

...

z<sup>(L)</sup> = θ<sup>(L-1)</sup> * a<sup>(L-1)</sup>

a<sup>(L)</sup> = g(z<sup>(L)</sup>)


In order to compute the derivatives, we're going to use an algorithm called back propagation.

𝛿<sub>j</sub><sup>(l)</sup> = "error" of node j in layer l.


𝛿<sub>j</sub><sup>(L)</sup> = a<sub>j</sub><sup>(L)</sup> - y<sub>j</sub> [Note here a<sub>j</sub><sup>(L)</sup> is (h<sub>θ</sub>(x))<sub>j</sub><sup>(L)</sup>]

...

𝛿<sup>(3)</sup> = (θ<sup>(3)</sup>)<sup>T</sup>\*𝛿<sup>(4)</sup>.\*g'(z<sup>(3)</sup>) = (θ<sup>(3)</sup>)<sup>T</sup> \* 𝛿<sup>(4)</sup>.\*(a<sup>(3)</sup>.\*(1-a<sup>(3)</sup>))

𝛿<sup>(2)</sup> = (θ<sup>(2)</sup>)<sup>T</sup>\*𝛿<sup>(3)</sup>.\*g'(z<sup>(2)</sup>) = (θ<sup>(2)</sup>)<sup>T</sup> \* 𝛿<sup>(3)</sup>.\*(a<sup>(2)</sup>.\*(1-a<sup>(2)</sup>))

There's no 𝛿<sup>(1)</sup> term b/c the first layer, which is the input layer, we don't want to change the input features we observed in our training sets, so that doesn't have any error associated with it. 

The name **back propagation** comes from the fact that we start by **computing the delta term for the output layer and then we go back** a layer and compute the delta terms for the third hidden layer and then we go back another step to compute delta 2 and so, we're sort of back propagating the errors from the output layer to layer 3 to layer 2 to hence the name back complication.

if Δ = 0:

∂J(θ)/∂θ<sub>i</sub><sub>j</sub><sup>(l)</sup> = a<sub>j</sub><sup>(l)</sup>\*𝛿<sub>i</sub><sup>(l+1)</sup>

(2) **Suppose we have m training example**:

Given training set{(x<sup>(1)</sup>,y<sup>(1)</sup>),...,(x<sup>(m)</sup>,y<sup>(m)</sup>)}

set Δ<sub>i</sub><sub>j</sub><sup>(l)</sup> = 0 (for all l,i,j)

For i = 1 to m:

**1.set a<sup>(1)</sup> = x<sup>(i)</sup>**

**2.perform forward propagation to compute a<sup>(l)</sup> for l = 2,3,...,L**

![forward propagation](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bYLgwteoEeaX9Qr89uJd1A_73f280ff78695f84ae512f19acfa29a3_Screenshot-2017-01-10-18.16.50.png?expiry=1559433600000&hmac=DpZBT3xCzRZ8PIyhDRLatGgGQbj9YcXeXIH8HRqKxyk)

**3.using y<sup>(i)</sup>, compute 𝛿<sup>(L)</sup> = a<sup>(L)</sup> - y<sup>(i)</sup>**

Where L is our total number of layers and a<sup>(L)</sup> is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

**4.compute 𝛿<sup>(L-1)</sup>, 𝛿<sup>(L-2)</sup>, ..., 𝛿<sup>(2)</sup> using 𝛿<sup>(l)</sup> = ((θ<sup>(l)</sup>)<sup>T</sup> \* 𝛿<sup>(l+1)</sup>).\*g'(z<sup>l</sup>) = ((θ<sup>(l)</sup>)<sup>T</sup> \* 𝛿<sup>(l+1)</sup>).\*a<sup>(l)</sup>.\*(1-a<sup>(l)</sup>)**

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by z<sup>(l)</sup>

**5.Δ<sub>i</sub><sub>j</sub><sup>(l)</sup> := Δ<sub>i</sub><sub>j</sub><sup>(l)</sup> + a<sub>j</sub><sup>(l)</sup> \* 𝛿<sub>i</sub><sup>(l+1)</sup> or with vectorization, Δ<sup>(l)</sup> := Δ<sup>(l)</sup> + 𝛿<sup>(l+1)</sup>\*(a<sup>(l)</sup>)<sup>T</sup>**

Hence we update our new Delta Δ matrix:

D<sub>i</sub><sub>j</sub><sup>(l)</sup> := 1/m\*(Δ<sub>i</sub><sub>j</sub><sup>(l)</sup> + Δθ<sub>i</sub><sub>j</sub><sup>(l)</sup>) if j!=0

D<sub>i</sub><sub>j</sub><sup>(l)</sup> := 1/m\*Δ<sub>i</sub><sub>j</sub><sup>(l)</sup> if j==0

∂J(θ)/∂θ<sub>i</sub><sub>j</sub><sup>(l)</sup> = D<sub>i</sub><sub>j</sub><sup>(l)</sup>

3. **Backpropagation Intuition**

![w5.3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W5/w5.3.PNG?raw=true)

![w5.4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W5/w5.4.PNG?raw=true)

In the image above, to calculate 𝛿<sub>2</sub><sup>(2)</sup>, we multiply the weights 0<sub>12</sub><sup>(2)</sup> and 0<sub>22</sub><sup>(2)</sup> by their respective 𝛿 values found to the right of each edge. So we get 𝛿<sub>2</sub><sup>(2)</sup> = 0<sub>12</sub><sup>(2)</sup>\*𝛿<sub>1</sub><sup>(3)</sup> + 0<sub>22</sub><sup>(2)</sup>\*𝛿<sub>2</sub><sup>(3)</sup>. To calculate every single possible 𝛿<sub>j</sub><sup>(l)</sup>, we could start from the right of our diagram. Going from right to left, to calculate the value of 𝛿<sub>j</sub><sup>(l)</sup>, you can just take the over all sum of each weight times the \deltaδ it is coming from. Hence, another example would be 𝛿<sub>2</sub><sup>(3)</sup>
= 0<sub>12</sub><sup>(3)</sup>\*𝛿<sub>1</sub><sup>(4)</sup>

### II. Backpropagation in Practice

1. **Advanced Optimization**

In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

```
thetaVector = [Theta1(:); Theta2(:); Theta3(:);]
deltaVector = [D1(:); D2(:); D3(:)]
```

Have initial parameters θ<sup>(1)</sup>, θ<sup>(2)</sup>, θ<sup>(3)</sup>

Unroll to get `initialTheta` to pass to

```
optTheta = fminunc(@costFunction, initialTheta, options)
```

```
function[jVal, gradientVec] = costFunction(thetaVec)
```

From `thetaVec`, use `reshape` to get θ<sup>(1)</sup>, θ<sup>(2)</sup>, θ<sup>(3)</sup>.

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:

```
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

Use forward and back propagation to compute D<<sup>(l)</sup>, D<<sup>(2)</sup>, D<<sup>(3)</sup> and J(θ).

Unroll D<sup>(l)</sup>, D<sup>(2)</sup>, D<sup>(3)</sup> to get `gradientVec`.


Example:

![w5.5](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W5/w5.5.PNG?raw=true)

2. **Gradient Checking**

Gradient checking will assure that our backpropagation works as intended. 

We can approximate the derivative of our cost function with:

![w5.6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W5/w5.6.PNG?raw=true)

Implement: 

```
gradApprox = (J(theta + EPSILON) - J(theta - EPSILON))/(2*EPSILON)
```

With multiple theta matrices, we can approximate the derivative with respect to θ<sub>j</sub> as follows:

θ = [θ1, θ2, ..., θn]

∂J(θ)/∂θ<sub>1</sub> ≈ (J(θ<sub>1</sub>+ε, θ<sub>2</sub>, ..., θ<sub>n</sub>) - J(θ<sub>1</sub>-ε, θ<sub>2</sub>, ..., θ<sub>n</sub>))/2ε

∂J(θ)/∂θ<sub>2</sub> ≈ (J(θ<sub>1</sub>, θ<sub>2</sub>+ε, ..., θ<sub>n</sub>) - J(θ<sub>1</sub>, θ<sub>2</sub>-ε, ..., θ<sub>n</sub>))/2ε

...

∂J(θ)/∂θ<sub>n</sub> ≈ (J(θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>n</sub>+ε) - J(θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>n</sub>-ε))/2ε

Implementation -- In octave/matlab we can do it as follows:

```matlab
EPSILON = 1e-4; 
% a small value for ε(epsilon) guarantees that the math works out properly
% if the value for ϵ is too small, we can end up with numerical problems

for i = 1:n,
    thetaPlus = theta;
    thetaPlus(i) += EPSILON;
    thetaMinus = theta;
    thetaMinus(i) -= EPSILON;
    gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*EPSILON)
end;
```

Check that `gradApprox` ≈ `DVec` (got from back propagation)

Once you have verified once that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to **compute gradApprox can be very slow**.

3. **Random Initialization**

__Initializing parameters θ to zeros does NOT work for neural network__ b/c parameters corresponding to inputs going to each of two hidden units are identical and when we backpropagate, all nodes will update to the same value repeatedly.

So we use **random initialization** to prevent from symmetry breaking

Initialize each θ<sub>i</sub><sub>j</sub><sup>(l)</sup> to a random value in [-ε, ε]

Eg:

```matlab
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.
Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

Note: the epsilon used above is unrelated to the epsilon from Gradient Checking

Q: Consider this procedure for initializing the parameters of a neural network:

1) Pick a random number r = rand(1,1) * (2 * INIT\_EPSILON) - INIT\_EPSILON;

2) Set θ<sub>i</sub><sub>j</sub><sup>(l)</sup> = r for all i, j, l.

Does this work?

A: No, because this fails to break symmetry.

4. **Training a Neural Network -- putting it together**

**Step 1: Pick a neural network architecture** (the \# of hidden layers and hidden units)

\# of input units: dimension of features x<sup>(i)</sup>
\# of output units: num of classes

Reasonable default: >= 1 hidden layer, same \# of hidden units in each hidden layer (but usually the more the better)

**Step 2: Randomly initialize the weights**

Initialize each θ<sub>i</sub><sub>j</sub><sup>(l)</sup> to a random value in [-ε, ε]

Eg:

```matlab
% If the dimensions of Theta1 is h1x(n+1), Theta2 is h2x(h1+1) and Theta3 is 1x(h2+1).
Theta1 = rand(h1,n+1) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(h2,h1+1) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,h2+1) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

**Step 3: Implement forward propagation to get h<sub>θ</sub>(x<sup>(i)</sup>) for any x<sup>(i)</sup>**

a<sup>(1)</sup> = x

z<sup>(2)</sup> = θ<sup>(1)</sup> * a<sup>(1)</sup>

a<sup>(2)</sup> = g(z<sup>(2)</sup>) [add a<sub>0</sub><sup>(2)</sup>]

z<sup>(3)</sup> = θ<sup>(2)</sup> * a<sup>(2)</sup>

a<sup>(3)</sup> = g(z<sup>(3)</sup>) [add a<sub>0</sub><sup>(3)</sup>]

...

z<sup>(L)</sup> = θ<sup>(L-1)</sup> * a<sup>(L-1)</sup>

a<sup>(L)</sup> = g(z<sup>(L)</sup>)

**Step 4: Compute the cost function J(θ)**

logistic regression cost function for neural networks without regularization:

![w5.7]()

logistic regression cost function for neural networks with regularization:

![w5.8]()

**Step 5: Implement backpropagation to compute partial derivatives ∂J(θ)/∂θ<sub>n</sub>**

𝛿<sub>j</sub><sup>(l)</sup> = "error" of node j in layer l.

using y<sup>(i)</sup>, compute 𝛿<sup>(L)</sup> = a<sup>(L)</sup> - y<sup>(i)</sup>:

𝛿<sub>j</sub><sup>(L)</sup> = a<sub>j</sub><sup>(L)</sup> - y<sub>j</sub> [Note here a<sub>j</sub><sup>(L)</sup> is (h<sub>θ</sub>(x))<sub>j</sub><sup>(L)</sup>]

...

𝛿<sup>(3)</sup> = (θ<sup>(3)</sup>)<sup>T</sup>\*𝛿<sup>(4)</sup>.\*g'(z<sup>(3)</sup>) = (θ<sup>(3)</sup>)<sup>T</sup> \* 𝛿<sup>(4)</sup>.\*(a<sup>(3)</sup>.\*(1-a<sup>(3)</sup>))

𝛿<sup>(2)</sup> = (θ<sup>(2)</sup>)<sup>T</sup>\*𝛿<sup>(3)</sup>.\*g'(z<sup>(2)</sup>) = (θ<sup>(2)</sup>)<sup>T</sup> \* 𝛿<sup>(3)</sup>.\*(a<sup>(2)</sup>.\*(1-a<sup>(2)</sup>))

There's no 𝛿<sup>(1)</sup> term b/c the first layer, which is the input layer, we don't want to change the input features we observed in our training sets, so that doesn't have any error associated with it. 

**Or**

We can use advanced optimization functions such as "fminunc()".

First, we have to "unroll" all the elements and put them into one long vector:

UNFINISHED

**Step 6: Use gradient checking to confirm that your backpropagation works. Then disable gradient checking**

∂J(θ)/∂θ<sub>1</sub> ≈ (J(θ<sub>1</sub>+ε, θ<sub>2</sub>, ..., θ<sub>n</sub>) - J(θ<sub>1</sub>-ε, θ<sub>2</sub>, ..., θ<sub>n</sub>))/2ε

∂J(θ)/∂θ<sub>2</sub> ≈ (J(θ<sub>1</sub>, θ<sub>2</sub>+ε, ..., θ<sub>n</sub>) - J(θ<sub>1</sub>, θ<sub>2</sub>-ε, ..., θ<sub>n</sub>))/2ε

...

∂J(θ)/∂θ<sub>n</sub> ≈ (J(θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>n</sub>+ε) - J(θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>n</sub>-ε))/2ε

Implementation -- In octave/matlab we can do it as follows:

```matlab
EPSILON = 1e-4; 
% a small value for ε(epsilon) guarantees that the math works out properly
% if the value for ϵ is too small, we can end up with numerical problems

for i = 1:n,
    thetaPlus = theta;
    thetaPlus(i) += EPSILON;
    thetaMinus = theta;
    thetaMinus(i) -= EPSILON;
    gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*EPSILON)
end;
```

Check that `gradApprox` ≈ `DVec` (got from back propagation)

**Step 7: Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta**

The following image gives us an intuition of what is happening as we are implementing our neural network:

![implementing gradient descent on cost function](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hGk18LsaEea7TQ6MHcgMPA_8de173808f362583eb39cdd0c89ef43e_Screen-Shot-2016-12-05-at-10.40.35-AM.png?expiry=1559520000000&hmac=-qNlzkeYNRccMEIAvO9xuUxRa0aN5OS0ajMVlJYOxFg)

Ideally, you want h<sub>θ</sub>(x<sup>(i)</sup>) ≈ y<sup>(i)</sup>. This will minimize our cost function. However, keep in mind that J(θ) is not convex and thus we can end up in a local minimum instead of a global minimum.

### III. Application of Neural Networks -- Autonomous Driving

[lecture video](https://d18ky98rnyall9.cloudfront.net/09.8-NeuralNetworksLearning-AutonomousDrivingExample.76891270b22b11e487451d0772c554c0/full/720p/index.mp4?Expires=1559520000&Signature=M7OEmN7ojEhWZ4tQDYdI6MrEuO7gyZ-OCO6YBz2Erw-WhnGrNdtIaoqdej07PkspiSuCHs1nSp8YZ1tvmIRMZ4mvlnHgy9gzJd4e5ICCH73KP~4-F1fWYQ4u8fMq9E3GiG18oNOqurfG0n5WWhmc8DydtCswnSUf-q7RhPlxARw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

[Another self-driving car example with reinforcement training](https://youtu.be/eRwTbRtnT1I)