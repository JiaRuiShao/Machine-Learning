# Application Example: Photo OCR

Identifying and recognizing objects, words, and digits in an image is a challenging task. We discuss how a pipeline can be built to tackle this problem and how to analyze and improve the performance of such a system.

## Photo OCR

### **Problem Description and Pipeline**

1. Text detection
2. Character segmentaion
3. Character classification

![1](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/1.png?raw=true)

Photo OCR Pipeline

![2](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/2.png?raw=true)

### **Sliding Windows Classifier**

Text detection is an unusual problem in computer vision. Because depending on the length of the text you're trying to find, these rectangles that you're trying to find can have different aspect. 

So in order to talk about detecting things in images let's start with a simpler example of pedestrian detection and we'll then later go back to. Ideas that were developed in pedestrian detection and apply them to text detection.

![3](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/3.png?raw=true)

Use a sliding windows classifier/detector in order to find pedestrians in the image.

Come back to the text detection example in our photo OCR pipeline, where our goal is to find the text regions in unit:

![4](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/4.png?raw=true)

Train the classifier to classfy the text split point.

![5](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/5.png?raw=true)

After that, do character clssification on rach segment.

### **Artificial Data Synthesis and Artificial Data**

The idea of artificial data synthesis comprises of two variations:

1. creating new data from scratch
2. amplify the training set we have to turn it into a larger training set

Synthetic data: take characters from different fonts and paste them against different backgrounds

![6](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/6.png?raw=true)

Synthesize data by introducing distortions:

![7](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/7.png?raw=true)

Usually does not help to add purely random noise to your data.

Note: 

+ Make sure you have a low bias classifier before expanding the effort. (Plot learning curves) Eg: keep increasing the # of features/# of hidden units in neural networks till you have a low bias classifier

+ "How much work would it be to get 10x as much data as we currently have?"
	- Artifical data synthesis
	- Collect/Label it yourself
	- Crowd source data labeling (eg: Amazon Mechanical Turk)

### **Ceiling Analysis: What Part of the Pipeline to Work on Next**

Having done ceiling analysis, we can understand what is the upside potential of improving each of these components.

The OCR example we used before:

![8](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/8.png?raw=true)

Another ceiling analysis example:

![9](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/9.png?raw=true)

Q: Suppose you perform ceiling analysis on a pipelined machine learning system, and when we plug in the ground-truth labels for one of the components, the performance of the overall system improves very little. This probably means: (check all that apply)

[ ] We should dedicate significant effort to collecting more data for that component.

[X] It is probably not worth dedicating engineering resources to improving that component of the system.

[X] If that component is a classifier training using gradient descent, it is probably not worth running gradient descent for 10x as long to see if it converges to better classifier parameters.

[ ] Choosing more features for that component may help (reducing bias), and reducing the number of features for that component (reducing variance) is unlikely to do so.

## Summary

![10](https://github.com/JiaRuiShao/Machine-Learning/blob/master/images/W11/10.png?raw=true)