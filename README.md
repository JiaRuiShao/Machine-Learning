## Week 2 Linear Regression with Multiple Variables

### I. Multivariate Linear Regression

1. **Multiple Features/Variables**

![w2.1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.1.PNG?raw=true)

Q: In the training set above, what is X<sub>1</sub> <sup>(4)</sup>? 

Size (feet)^2 | Number of bedrooms | Number of floors | Age of home (years) | Price ($1000)
------|-------|------|-------|-------
2104	|5	|1	|45	|460
1416	|3	|2	|40	|232
1534	|3	|2	|30	|315
852	    |2	|1	|36	|178
...	    |...	|...	|...	|...

Answer:
The size (in feet<sup>2</sup>) of the 4th home in the training set

Hypothesis:
h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub>x<sub>1</sub> + &theta;<sub>2</sub>x<sub>2</sub> + &theta;<sub>3</sub>x<sub>3</sub> + ...+ &theta;<sub>n</sub>x<sub>n</sub>

For convenience of notation, define x<sub>0</sub><sup>(i)</sup>=1

h<sub>&theta
;</sub>(x) = &theta;<sub>0</sub>x<sub>0
</sub> + &theta;<sub>1</sub>x<sub>1</sub> + &theta;<sub>2</sub>x<sub>2</sub> + &theta;<sub>3</sub>x<sub>3</sub> + ...+ &theta;<sub>n</sub>x<sub>n</sub>

We can vecterize it as:
h<sub>&theta;</sub>(x) = &theta;<sup>T</sup>X

where

```
X = 	[x0]		θ = 		[θ0]
	[x1]				[θ1]
	[x2]				[θ2]
	[x3]				[θ3]
	[.]				  .
	[.]				  .
	[.]				  .
	[xn]				[θn]
```

2. **Gradient Descent for Multiple Variables**

Cost function:

![w2.2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.2.PNG?raw=true)

Gradient Descent:

Repeat {
	&theta;<sub>j</sub> := &theta;<sub>j</sub> - &alpha; * ∂J(&theta;)/∂&theta;<sub>j</sub>
} for every j = 0,1,..., n

![w2.3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.3.PNG?raw=true)

To elaborate on that: 
![w2.4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.4.PNG?raw=true)

3. **Gradient Descent -- Feature Scaling & Mean Normalization**

We can speed up gradient descent by having each of our input values in roughly the same range. This is because __θ will descend quickly on small ranges and slowly on large ranges__, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

−1 ≤ x<sub>(i)</sub> ≤ 1

or

−0.5 ≤ x<sub>(i)</sub> ≤ 0.5

Notice that these aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges.

Two techniques to help with this are feature scaling and mean normalization. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.

**Feature Scaling**

Idea: make sure features are on a similar scale

![w2.5](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.5.PNG?raw=true)

By implementing feature scaling, gradient descent can converge more faster.

Normally, we want to get every feature into aproximately a -1 <= x<sub>i</sub> <= 1 range

**Mean Normalization**

Replace x<sub>i</sub> with x<sub>i</sub> - μ<sub>i</sub> to make features have approximately zero mean (Do not apply to X0 = 1)

To implement both of these techniques, adjust your input values as shown in this formula:

x<sub>i</sub> := (x<sub>i</sub> - μ<sub>i</sub>)/s<sub>i</sub>

where μi is the average of all the values for feature (i) and s<sub>i</sub> is the range of values (max - min), or s<sub>i</sub> is the standard deviation.

For example, if x<sub>i</sub> represents housing prices with a range of 100 to 2000 and a mean value of 1000, then x<sub>i</sub> := (price - 1000)/1900

4. **Gradient Descent -- Learning Rate alpha &alpha;**

Making sure gradient descent is working correctly: J(&theta;)should decrease after every iteration.

**A. Debugging gradient descent**. Make a plot with number of iterations on the x-axis. Now plot the cost function, J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

![w2.6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.6.PNG?raw=true)

It has been proven that if learning rate α is sufficiently small, J(θ) should decrease on every iteration.

- If α is too small, gradient descent can be slow to converge.
- If α is too large,J(θ) may not decrease on every iteration and thus may not converge.

**B. Automatic convergence test**. Declare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as 10−3. However in practice it's difficult to choose this threshold value.

Q: Suppose a friend ran gradient descent three times, with α=0.01, α=0.1, and α=1, and got the following three plots (labeled A, B, and C):

![w2.7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.7.PNG?raw=true)

Which plots corresponds to which values of α?

Answer: A is &alpha; = 0.1; B is &alpha; = 0.01; C is &alpha; = 1 b/c
in graph C, the cost function is increasing, so the learning rate is set too high. Both graphs A and B converge to an optimum of the cost function, but graph B does so very slowly, so its learning rate is set too low. Graph A lies between the two.

To choose &alpha;, try ..., 0.001, 0.01, 0.1, 1,...

5. **Features and Polynomial Regression**

We can improve our features and the form of our hypothesis function in a couple different ways.

We can combine multiple features into one. For example, we can combine x<sub>1</sub> and x<sub>2</sub> into a new feature x<sub>3</sub> by taking x<sub>1</sub> * x<sub>2</sub> 

**Polynomial Regression**

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is
sub>&theta;</sub>(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub>x<sub>1</sub> then we can create additional features based on x<sub>1</sub>, to get thecquadratic function h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub>x<sub>1</sub> + &theta;<sub>2</sub>x<sub>1</sub><sup>2</sup> or the cubic function h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub>x<sub>1</sub> + &theta;<sub>2</sub>x<sub>1</sub><sup>2</sup> + &theta;<sub>3</sub>x<sub>1</sub><sup>3</sup>

In the cubic version, we have created new features x<sub>2</sub> and x<sub>3</sub> where x<sub>2</sub> = x<sub>1</sub><sup>2</sup> and x<sub>3</sub> = x<sub>1</sub><sup>3</sup>

To make it a square root function, we could do: h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub>x<sub>1</sub> + &theta;<sub>2</sub>√x<sub>1</sub>

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

If x<sub>1</sub> has range 1-1000 then x<sub>1</sub><sup>2</sup> becomes 1-1000000 and x<sub>1</sub><sup>3</sup> becomes 1-1000000000 

### II. Computing Parameters Analytically

1. **Normal Equation -- Alternative to Gradient Descent**

Normal Equation: Method to solve for θ analytically.

Cost function: ![w2.2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.2.PNG?raw=true)

Let ∂J(θ)/∂θ<sub>j</sub> = 0 for every j;

Solve θ = 0, 1, 2, ...,n

```
	[---(x(1))T---]
X = 	[---(x(2))T---]
	[-----...-----]
	[---(x(m))T---]
```

Note: Dimensional Analysis is pretty important

dim of X: (m,n+1);

dim of y: (m,1);

dim of θ: (n+1,1);

θ = (X<sup>T</sup> * X)<sup>-1</sup> * X<sup>T</sup> * Y

There is **NO NEED** to do feature scaling with the normal equation.

**Pros and Cons for Gradient Descent and Normal Equation**

**Gradient Descent** | **Normal Equation**
---|---
Needs to choose α | No need to choose α 
Needs many iterations | Don't need to iterate
Works well even when n is pretty large | Needs to compute (X<sup>T</sup> * X)<sup>-1</sup>, which is slow when n is large

2. **Normal Equation Noninvertibility**

When computing θ = (X<sup>T</sup> * X)<sup>-1</sup> * X<sup>T</sup> * Y,

- What if X<sup>T</sup> * X is non-invertible/singular/degenerate?
- Octave: pinv/inv

When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of θ even if X<sup>T</sup> * X is not invertible

**Reasons of X<sup>T</sup> * X being non-invertible**:
- **Redundant features**(linearly dependent)
	+ delete a feature that is linearly dependent with another 
- **To many features** (m<=n)
	+ Delete some features, or use regularization(to be explained later)

### III. Octave/Matlab Tutorial

1. **Basic Functions**

```
>> 5+6

ans =

    11

>> 2^6

ans =

    64

>> 1==2

ans =

  logical

   0

>> 1~=2 % true

ans =

  logical

   1

>> 1 && 0 % AND

ans =

  logical

   0

>> 1 || 0 % OR

ans =

  logical

   1

>> a=3; %semicolon supressing output
>> a

a =

     3
 
>> disp(sprintf('7 decimals: %0.7f', pi))
7 decimals: 3.1415927
>> pi

ans =

    3.1416

>> format long
>> pi

ans =

   3.141592653589793

>> format short
>> pi

ans =

    3.1416

>> A = [1 2; 3 4; 5 6]

A =

     1     2
     3     4
     5     6

>> B = [1; 2; 3]

B =

     1
     2
     3

>> x = 1:2:11

x =

     1     3     5     7     9    11

>> x=1:10:55

x =

     1    11    21    31    41    51

>> C = 2*ones(2,3)

C =

     2     2     2
     2     2     2

>> D = zeros(1.3)
Error using zeros
Size inputs must be integers.
 
>> D = zeros(1,3)

D =

     0     0     0

>> rand(3,3)

ans =

    0.8147    0.9134    0.2785
    0.9058    0.6324    0.5469
    0.1270    0.0975    0.9575

>> 10*rand(5,5)

ans =

    9.6489    8.0028    9.5949    6.7874    1.7119
    1.5761    1.4189    6.5574    7.5774    7.0605
    9.7059    4.2176    0.3571    7.4313    0.3183
    9.5717    9.1574    8.4913    3.9223    2.7692
    4.8538    7.9221    9.3399    6.5548    0.4617

>> E = sqrt(100)*randn(1,100)

E =

  Columns 1 through 7

  -29.4428   14.3838    3.2519   -7.5493   13.7030  -17.1152   -1.0224

  Columns 8 through 14

   -2.4145    3.1921    3.1286   -8.6488   -0.3005   -1.6488    6.2771

  Columns 15 through 21

   10.9327   11.0927   -8.6365    0.7736  -12.1412  -11.1350   -0.0685

  Columns 22 through 28

   15.3263   -7.6967    3.7138   -2.2558   11.1736  -10.8906    0.3256

  Columns 29 through 35

    5.5253   11.0061   15.4421    0.8593  -14.9159   -7.4230  -10.6158

  Columns 36 through 42

   23.5046   -6.1560    7.4808   -1.9242    8.8861   -7.6485  -14.0227

  Columns 43 through 49

  -14.2238    4.8819   -1.7738   -1.9605   14.1931    2.9158    1.9781

  Columns 50 through 56

   15.8770   -8.0447    6.9662    8.3509   -2.4372    2.1567  -11.6584

  Columns 57 through 63

  -11.4795    1.0487    7.2225   25.8549   -6.6689    1.8733   -0.8249

  Columns 64 through 70

  -19.3302   -4.3897  -17.9468    8.4038   -8.8803    1.0009   -5.4453

  Columns 71 through 77

    3.0352   -6.0033    4.8997    7.3936   17.1189   -1.9412  -21.3836

  Columns 78 through 84

   -8.3959   13.5459  -10.7216    9.6095    1.2405   14.3670  -19.6090

  Columns 85 through 91

   -1.9770  -12.0785   29.0801    8.2522   13.7897  -10.5818   -4.6862

  Columns 92 through 98

   -2.7247   10.9842   -2.7787    7.0154  -20.5182   -3.5385   -8.2359

  Columns 99 through 100

  -15.7706    5.0797

>> hist(E)
>> % shows histgram
>> eye(3) % identity matrix

ans =

     1     0     0
     0     1     0
     0     0     1

>> help eye
 eye Identity matrix.
    eye(N) is the N-by-N identity matrix.
 
    eye(M,N) or eye([M,N]) is an M-by-N matrix with 1's on
    the diagonal and zeros elsewhere.
 
    eye(SIZE(A)) is the same size as A.
 
    eye with no arguments is the scalar 1.
 
    eye(..., CLASSNAME) is a matrix with ones of class specified by
    CLASSNAME on the diagonal and zeros elsewhere.
 
    eye(..., 'like', Y) is an identity matrix with the same data type, sparsity,
    and complexity (real or complex) as the numeric variable Y.
 
    Note: The size inputs M and N should be nonnegative integers. 
    Negative integers are treated as 0.
 
    Example:
       x = eye(2,3,'int8');
 
    See also speye, ones, zeros, rand, randn.

    Reference page for eye
    Other functions named eye

>> 
```

2. **Moving Data Around**

```
>> A = [1 2; 3 4; 5 6]

A =

     1     2
     3     4
     5     6

>> size(A)

ans =

     3     2

>> sz = size(A)

sz =

     3     2

>> size(sz)

ans =

     1     2

>> size(A,1) % first dim of A

ans =

     3

>> size(A,2) % col of A

ans =

     2

>> pwd

ans =

    'C:\Program Files (x86)\MATLAB\bin'

>> ls

.                lcdata_utf8.xml  mex.pl           util             
..               m3iregistry      mexext.bat       win32            
arch             matlab.exe       mexsetup.pm      win64            
deploytool.bat   mbuild.bat       mexutils.pm      worker.bat       
lcdata.xml       mcc.bat          mw_mpiexec.bat   
lcdata.xsd       mex.bat          registry         


>> cd 'C:\Users\surface\Desktop'
>> pwd

ans =

    'C:\Users\surface\Desktop'

>> ls

.                               Google Drive.lnk                
..                              INTERNSHIP.docx                 
2019 Classes.txt                MATLAB.lnk                      
CC Talk.lnk                     WhatsApp.lnk                    
Courses Taken and Planned.xlsx  desktop.ini                     
Du Pan.lnk                      ~$PORTANT (ENGLISH).docx        
GRE Schedule.xlsx               

>> % which lists all files on desktop rn
>> load('2019 Classes.txt')
Error using load
Number of columns on line 2 of ASCII file 2019 Classes.txt must be the
same as previous lines.
 
>> who

Your variables are:

A    B    C    D    E    a    ans  sz   x    

>> whos
  Name      Size             Bytes  Class     Attributes

  A         3x2                 48  double              
  B         3x1                 24  double              
  C         2x3                 48  double              
  D         1x3                 24  double              
  E         1x100              800  double              
  a         1x1                  8  double              
  ans       1x24                48  char                
  sz        1x2                 16  double              
  x         1x6                 48  double              

>> save hello.mat A;
>> % save A to hello.mat file
>>  save hello.txt A -ascii  % save as text(ASCII)
>> A

A =

     1     2
     3     4
     5     6

>> A(3,2)

ans =

     6

>> A(2,:) % ":" means every elements along that row/column

ans =

     3     4

>> A(:,2)

ans =

     2
     4
     6

>> A([1 3],:) 

ans =

     1     2
     5     6

>> % gives everything from the first and the third rows
>> % can also used to assign values
>> A(:,2) = [10;11;12]

A =

     1    10
     3    11
     5    12

>> A = [A,[100;101;102]] % append another column vector to right

A =

     1    10   100
     3    11   101
     5    12   102

>> size(A)

ans =

     3     3

>> A(:) % put all elements of A into a single vector

ans =

     1
     3
     5
    10
    11
    12
   100
   101
   102

>> A

A =

     1    10   100
     3    11   101
     5    12   102

>> B

B =

     1
     2
     3

>> F = [A B]

F =

     1    10   100     1
     3    11   101     2
     5    12   102     3

>> G = [A; B] % B goes bottom of A
Error using vertcat
Dimensions of arrays being concatenated are not consistent.
 
>> G = [A;B] % B goes bottom of A
Error using vertcat
Dimensions of arrays being concatenated are not consistent.
 
>> G = ones(1,3)

G =

     1     1     1

>> H = [A; G] % G goes bottom of A

H =

     1    10   100
     3    11   101
     5    12   102
     1     1     1

```

3. **Computing on Data**

```

>> A * B

ans =

   321
   328
   335

>> A .*B
Error: "A" was previously used as a variable, conflicting with its use
here as the name of a function or command.
See "How MATLAB Recognizes Command Syntax" in the MATLAB documentation for
details.
 
>> A .* B

ans =

     1    10   100
     6    22   202
    15    36   306

>> A

A =

     1    10   100
     3    11   101
     5    12   102

>> B

B =

     1
     2
     3

>> A .^ 2

ans =

           1         100       10000
           9         121       10201
          25         144       10404

>> 1 ./ A

ans =

    1.0000    0.1000    0.0100
    0.3333    0.0909    0.0099
    0.2000    0.0833    0.0098

>> abs[-1;-11;-22]
 abs[-1;-11;-22]
    ↑
Error: Invalid expression. When calling a function or indexing a variable,
use parentheses. Otherwise, check for mismatched delimiters.
 
>> abs([-1;-11;-22])

ans =

     1
    11
    22

>> A+1

ans =

     2    11   101
     4    12   102
     6    13   103

>> A'

ans =

     1     3     5
    10    11    12
   100   101   102

>> % gives A transpose
>> (A')' % which should be A

ans =

     1    10   100
     3    11   101
     5    12   102

>> max(A)

ans =

     5    12   102

>> a

a =

     3

>> a = magic(3)

a =

     8     1     6
     3     5     7
     4     9     2

>> % each rows, columns, and diagnols have the same sum() value
>> [r,c] = find(a>=7)

r =

     1
     3
     2


c =

     1
     2
     3

>> help find
 find   Find indices of nonzero elements.
    I = find(X) returns the linear indices corresponding to 
    the nonzero entries of the array X.  X may be a logical expression. 
    Use IND2SUB(SIZE(X),I) to calculate multiple subscripts from 
    the linear indices I.
  
    I = find(X,K) returns at most the first K indices corresponding to 
    the nonzero entries of the array X.  K must be a positive integer, 
    but can be of any numeric type.
 
    I = find(X,K,'first') is the same as I = find(X,K).
 
    I = find(X,K,'last') returns at most the last K indices corresponding 
    to the nonzero entries of the array X.
 
    [I,J] = find(X,...) returns the row and column indices instead of
    linear indices into X. This syntax is especially useful when working
    with sparse matrices.  If X is an N-dimensional array where N > 2, then
    J is a linear index over the N-1 trailing dimensions of X.
 
    [I,J,V] = find(X,...) also returns a vector V containing the values
    that correspond to the row and column indices I and J.
 
    Example:
       A = magic(3)
       find(A > 5)
 
    finds the linear indices of the 4 entries of the matrix A that are
    greater than 5.
 
       [rows,cols,vals] = find(speye(5))
 
    finds the row and column indices and nonzero values of the 5-by-5
    sparse identity matrix.
 
    See also sparse, ind2sub, relop, nonzeros.

    Reference page for find
    Other functions named find

>> sum(a)

ans =

    15    15    15

>> prod(a)

ans =

    96    45    84

>> floor(a)

ans =

     8     1     6
     3     5     7
     4     9     2

>> ceil(a)

ans =

     8     1     6
     3     5     7
     4     9     2
 
>> max(rand(3), rand(3))

ans =

    0.0965    0.9561    0.4509
    0.1320    0.7317    0.5470
    0.9421    0.6477    0.8212

>> max(A,[],1)

ans =

     5    12   102

>> max(a,[],1) % find max by row

ans =

     8     9     7

>> max(a,[],2) % find max by col

ans =

     8
     7
     9

>> sum(a,1) % sum all rows by col

ans =

    15    15    15

>> sum(a,2) % sum all cols by row

ans =

    15
    15
    15

>> I = magic(9)

I =

    47    58    69    80     1    12    23    34    45
    57    68    79     9    11    22    33    44    46
    67    78     8    10    21    32    43    54    56
    77     7    18    20    31    42    53    55    66
     6    17    19    30    41    52    63    65    76
    16    27    29    40    51    62    64    75     5
    26    28    39    50    61    72    74     4    15
    36    38    49    60    71    73     3    14    25
    37    48    59    70    81     2    13    24    35

>> A .* eye(9)
Matrix dimensions must agree.
 
>> I .* eye(9)

ans =

    47     0     0     0     0     0     0     0     0
     0    68     0     0     0     0     0     0     0
     0     0     8     0     0     0     0     0     0
     0     0     0    20     0     0     0     0     0
     0     0     0     0    41     0     0     0     0
     0     0     0     0     0    62     0     0     0
     0     0     0     0     0     0    74     0     0
     0     0     0     0     0     0     0    14     0
     0     0     0     0     0     0     0     0    35

>> sum(sum(I .* eye(9)))

ans =

   369

>> flipud(eye(9))

ans =

     0     0     0     0     0     0     0     0     1
     0     0     0     0     0     0     0     1     0
     0     0     0     0     0     0     1     0     0
     0     0     0     0     0     1     0     0     0
     0     0     0     0     1     0     0     0     0
     0     0     0     1     0     0     0     0     0
     0     0     1     0     0     0     0     0     0
     0     1     0     0     0     0     0     0     0
     1     0     0     0     0     0     0     0     0

>> a

a =

     8     1     6
     3     5     7
     4     9     2

>> pinv(a)

ans =

    0.1472   -0.1444    0.0639
   -0.0611    0.0222    0.1056
   -0.0194    0.1889   -0.1028
```

4. **Plotting Data**

```
>> t = [0:0.01:0.98];
>> y1=cos(2*pi*4*t); 
>> plot(t,y1)
>> hold on;
>> plot(t,y2,'r')
>> legend('cos','sin')
>> title('my plot')
>> cd 'C:\Users\surface\Desktop';
>> close
>> figure(1)=plot(t,y1);
>> figure(2)=plot(t,y2);
>> subplot(1,2,1); % Divide plot a 1x2 grid, access the first element
>> plot(t,y1);
>> subplot(1,2,2);
>> plot(t,y2);
>> axis([0.5 1 -1 1])
>> clf;
>> a = magic(5)

a =

    17    24     1     8    15
    23     5     7    14    16
     4     6    13    20    22
    10    12    19    21     3
    11    18    25     2     9

>> imagesc(a)
>> imagesc(a), colorbar, colormap gray;
>> imagesc(magic(15)), colorbar, colormap gray;
>> 
```

5. **Cost Function**

```
function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing our training examples.
% y is the class labels

m = size(X,1); % number of training examples
predictions = X*theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions-y) .^ 2; % squared errors

J = 1/(2*m) *sum(sqrErrors);

>> X = [1 1; 1 2; 1 3]

X =

     1     1
     1     2
     1     3

>> y = [1; 2; 3]

y =

     1
     2
     3

>> theta = [0;1]

theta =

     0
     1

>> j = costFunctionJ(X, y, theta)
Undefined function or variable 'costFunctionJ'.
 
>> m = size(X,1); % number of training examples
predictions = X*theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions-y) .^ 2; % squared errors

J = 1/(2*m) *sum(sqrErrors);
>> J

J =

     0

>> theta = [1;1]

theta =

     1
     1

>> predictions = X*theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions-y) .^ 2; % squared errors

J = 1/(2*m) *sum(sqrErrors)

J =

    0.5000
```

6. **Vectorization**

**Vectorize Cost Function**

![w2.8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.8.PNG?raw=true)

Unvectorized implementation

```
prediction = 0;
for j = 1:n+1,
	prediction += theta(j) * x(j)
end;

```

Vectorized implementation

```
prediction = theta' * x;
```

Implementation in C++:

Unvectorized implementation

```
double prediction = 0;
for (j = 0; j <= n; j++)
	prediction += theta[j] * x[j];
```

Vectorized implementation

```
double prediction = theta.transpose() * x;
```

**Vectorize Gradient Descent**

![w2.9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W2/w2.9.PNG?raw=true)

7. **Others**

**MATLAB Documentation**

<https://www.mathworks.com/help/matlab/?refresh=true>


Syntax differences between MATLAB and Python:

**matrix multiplication**

MATLAB: A*B

Python: np.dot(A,B)

**element-wise multiplication**

MATLAB: A.*B

Python: A*B

\*\*division is also the same!! Use `./` if it's a element-wise division

Others:

For MATLAB, have to specify the file location in the beginning using `cd` function.