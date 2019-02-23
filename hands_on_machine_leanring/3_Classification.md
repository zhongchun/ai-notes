# Classification

[TOC]

In Chapter 1 we mentioned that the most common supervised learning tasks are regression (predicting values) and classification (predicting classes). In Chapter 2 we explored a regression task, predicting housing values, using various algorithms such as Linear Regression, Decision Trees, and Random Forests (which will be explained in further detail in later chapters). Now we will turn our attention to classification systems.

## MNIST

In this chapter, we will be using the MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Cen‐ sus Bureau. Each image is labeled with the digit it represents. This set has been stud‐ ied so much that it is often called the “Hello World” of Machine Learning: whenever people come up with a new classification algorithm, they are curious to see how it will perform on MNIST. Whenever someone learns Machine Learning, sooner or later they tackle MNIST.

![A few digits from the MNIST dataset](https://ws4.sinaimg.cn/large/006tKfTcly1g0gi50w5vnj30u00u0mzu.jpg)

## Training a Binary Classifier

Let’s simplify the problem for now and only try to identify one digit—for example, the number 5. This “5-detector” will be an example of a binary classifier, capable of distinguishing between just two classes, 5 and not-5. Let’s create the target vectors for this classification task:

```python
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits. 
y_test_5 = (y_test == 5)
```

## Performance Measures

Evaluating a classifier is often significantly trickier than evaluating a regressor, so we will spend a large part of this chapter on this topic. There are many performance measures available, so grab another coffee and get ready to learn many new concepts and acronyms!

### Measuring Accuracy Using Cross-Validation

A good way to evaluate a model is to use cross-validation, just as you did in Chapter 2.

This demonstrates why accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).

### Confusion Matrix

A much better way to evaluate the performance of a classifier is to look at the confu‐ sion matrix. The general idea is to count the number of times instances of class A are classified as class B. For example, to know the number of times the classifier confused images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion matrix.

![](https://ws4.sinaimg.cn/large/006tKfTcly1g0gi8pa6szj310i0jcjvr.jpg)

### Precision and Recall

$precision=\frac{TP}{TP+FP}​$

$recall=\frac{TP}{TP+FN}$

$F_1=\frac{2}{\frac{1}{precision}+\frac{1}{recall}}​$

### Precision/Recall Tradeoff

![](https://ws1.sinaimg.cn/large/006tKfTcly1g0gicmdbkbj310c0d8gox.jpg)

![Precision and recall versus the decision threshold](https://ws3.sinaimg.cn/large/006tKfTcly1g0gid0crvdj31o00u0dir.jpg)

![Precision versus recall](https://ws2.sinaimg.cn/large/006tKfTcly1g0gidxl67mj31400u0tas.jpg)

### The ROC Curve

![ROC curve](https://ws4.sinaimg.cn/large/006tKfTcly1g0gif03ifdj31400u0whu.jpg)

## Multiclass Classification

Whereas binary classifiers distinguish between two classes, multiclass classifiers (also called multinomial classifiers) can distinguish between more than two classes.

Some algorithms (such as Random Forest classifiers or naive Bayes classifiers) are capable of handling multiple classes directly. Others (such as Support Vector Machine classifiers or Linear classifiers) are strictly binary classifiers. However, there are vari‐ ous strategies that you can use to perform multiclass classification using multiple binary classifiers.

- OvA: one-versus-all
- OvO: one-versus-one

Some algorithms (such as Support Vector Machine classifiers) scale poorly with the size of the training set, so for these algorithms OvO is preferred since it is faster to train many classifiers on small training sets than training few classifiers on large training sets. For most binary classification algorithms, however, OvA is preferred.

## Error Analysis

Of course, if this were a real project, you would follow the steps in your Machine Learning project checklist (see Appendix B): exploring data preparation options, try‐ ing out multiple models, shortlisting the best ones and fine-tuning their hyperpara‐ meters using GridSearchCV, and automating as much as possible, as you did in the previous chapter. Here, we will assume that you have found a promising model and you want to find ways to improve it. One way to do this is to analyze the types of errors it makes.

![Confusion matrix](https://ws1.sinaimg.cn/large/006tKfTcly1g0gigwracxj30u00u0q3c.jpg)![Confusion matrix errors](https://ws3.sinaimg.cn/large/006tKfTcly1g0gihb8gabj30u00u0t94.jpg)

## Multilabel Classification

Until now each instance has always been assigned to just one class. In some cases you may want your classifier to output multiple classes for each instance. For example, consider a facerecognition classifier: what should it do if it recognizes several people on the same picture? Of course it should attach one label per person it recognizes. Say the classifier has been trained to recognize three faces, Alice, Bob, and Charlie; then when it is shown a picture of Alice and Charlie, it should output [1, 0, 1] (meaning “Alice yes, Bob no, Charlie yes”). Such a classification system that outputs multiple binary labels is called a multilabel classification system.

## Multioutput Classification

The last type of classification task we are going to discuss here is called multioutput- multiclass classification (or simply multioutput classification). It is simply a generaliza‐ tion of multilabel classification where each label can be multiclass (i.e., it can have more than two possible values).

## Reference

1. [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)
2. [Errata for Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/catalog/errata.csp?isbn=0636920052289)
3. [GitHub: Hands-On Machine Learnging With Scikit-Learn and TensorFlow](https://github.com/ageron/handson-ml)
