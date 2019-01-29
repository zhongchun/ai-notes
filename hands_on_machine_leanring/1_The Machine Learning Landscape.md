# The Machine Learning Landscape

[TOC]

Where does Machine Learning start and where does it end? What exactly does it mean for a machine to learn something? 

## What is Machine Learning
Machine Learning is the science (and art) of programming computers so they can learn from data.

Here is a slightly more general definition:
```
    [Machine Learning is the] field of study that gives computers the ability to learn 
    without being explicitly programmed.
        —Arthur Samuel, 1959
```

And a more engineering-oriented one:
```
    A computer program is said to learn from experience E with respect to some task T 
    and some performance measure P, if its performance on T, as measured by P, improves 
    with experience E.
        —Tom Mitchell, 1997
```

## Why use Machine Learning
![Screen Shot 2019-01-29 at 7.55.13 AM](https://ws4.sinaimg.cn/large/006tNc79ly1fzn4hjlv1oj31400ja0vn.jpg)

![Screen Shot 2019-01-29 at 10.20.09 PM](https://ws2.sinaimg.cn/large/006tNc79ly1fznthdqu5ej31400k4gph.jpg)

![Screen Shot 2019-01-29 at 10.20.49 PM](https://ws1.sinaimg.cn/large/006tNc79ly1fznti0qg9dj31400fgwie.jpg)

![Screen Shot 2019-01-29 at 10.21.19 PM](https://ws1.sinaimg.cn/large/006tNc79ly1fzntijgqm0j31400je798.jpg)

To summarize, Machine Learning is great for:

- Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform bet ter.
- Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.

## Types of Machine Learning Systems

There are so many different types of Machine Learning systems that it is useful to classify them in broad categories based on:

- Whether or not they are trained with human supervision (supervised, unsuper‐ vised, semisupervised, and Reinforcement Learning)
- Whether or not they can learn incrementally on the fly (online versus batch learning)
- Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning)

### Supervised/Unsupervised Learning

Machine Learning systems can be classified according to the amount and type of supervision they get during training. There are four major categories: supervised learning, unsupervised learning, semisupervised learning, and Reinforcement learning.

#### Supervised learning

In supervised learning, the training data you feed to the algorithm includes the desired solutions, called labels (Figure 1-5).

![1548632887379](https://ws2.sinaimg.cn/large/006tNc79ly1fzlz1i60q3j326t0u0n3p.jpg)

A typical supervised learning task is classification. The spam filter is a good example of this: it is trained with many example emails along with their class (spam or ham), and it must learn how to classify new emails.

Another typical task is to predict a target numeric value, such as the price of a car, given a set of features (mileage, age, brand, etc.) called predictors. This sort of task is called regression (Figure 1-6). To train the system, you need to give it many examples of cars, including both their predictors and their labels (i.e., their prices).

![Screen Shot 2019-01-29 at 7.39.14 AM](https://ws4.sinaimg.cn/large/006tNc79ly1fzn410gk2tj31400mi79c.jpg)

Here are some of the most important supervised learning algorithms (covered in this book):

- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural networks

#### Unsupervised learning

In unsupervised learning, as you might guess, the training data is unlabeled (Figure 1-7). The system tries to learn without a teacher.

![Screen Shot 2019-01-29 at 7.42.42 AM](https://ws1.sinaimg.cn/large/006tNc79ly1fzn44d9kn8j31400eggq6.jpg)

Here are some of the most important unsupervised learning algorithms (we will cover dimensionality reduction):

- Clustering
  - k-Means
  - Hierarchical Cluster Analysis (HCA)
  - Expectation Maximization
- Visualization and dimensionality reduction
  - Principal Component Analysis (PCA)
  - Kernel PCA
  - Locally-Linear Embedding (LLE)
  - t-distributed Stochastic Neighbor Embedding (t-SNE)

- Association rule learning
  - Apriori
  - Eclat

![Screen Shot 2019-01-29 at 7.47.23 AM](https://ws4.sinaimg.cn/large/006tNc79ly1fzn498dqrtj31400eoq77.jpg)

![Screen Shot 2019-01-29 at 7.48.00 AM](https://ws3.sinaimg.cn/large/006tNc79ly1fzn49w77axj31400qmb29.jpg)

![Screen Shot 2019-01-29 at 7.48.59 AM](https://ws3.sinaimg.cn/large/006tNc79ly1fzn4avh8lrj31400gsdiy.jpg)

#### Semisupervised learning

Some algorithms can deal with partially labeled training data, usually a lot of unla‐ beled data and a little bit of labeled data. This is called semisupervised learning (Figure 1-11).

![Screen Shot 2019-01-29 at 7.50.11 AM](https://ws3.sinaimg.cn/large/006tNc79ly1fzn4c4g6qij31400gwn0n.jpg)

Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms. For example, deep belief networks (DBNs) are based on unsu‐ pervised components called restricted Boltzmann machines (RBMs) stacked on top of one another. RBMs are trained sequentially in an unsupervised manner, and then the whole system is fine-tuned using supervised learning techniques.

#### Reinforcement learning

Reinforcement Learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative rewards, as in Figure 1-12). It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

![Screen Shot 2019-01-29 at 7.52.13 AM](https://ws1.sinaimg.cn/large/006tNc79ly1fzn4e9a9ryj31400n67et.jpg)

### Batch and Online Learning

Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.

#### Batch learning

In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning.

If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then stop the old system and replace it with the new one.

Fortunately, the whole process of training, evaluating, and launching a Machine Learning system can be automated fairly easily (as shown in Figure 1-3), so even a batch learning system can adapt to change. Simply update the data and train a new version of the system from scratch as often as needed.

This solution is simple and often works fine, but training using the full set of data can take many hours, so you would typically train a new system only every 24 hours or even just weekly. If your system needs to adapt to rapidly changing data (e.g., to pre‐ dict stock prices), then you need a more reactive solution.

Also, training on the full set of data requires a lot of computing resources (CPU, memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge, it may even be impossible to use a batch learning algorithm.

Finally, if your system needs to be able to learn autonomously and it has limited resources (e.g., a smartphone application or a rover on Mars), then carrying around large amounts of training data and taking up a lot of resources to train for hours every day is a showstopper.

Fortunately, a better option in all these cases is to use algorithms that are capable of learning incrementally.

#### Online learning

In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives (see Figure 1-13).

![Screen Shot 2019-01-29 at 10.25.56 PM](https://ws3.sinaimg.cn/large/006tNc79ly1fzntnc4k2qj31400iyn0k.jpg)

Online learning is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously.

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (this is called out-of-core learning). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data (see Figure 1-14).

This whole process is usually done offline (i.e., not on the live sys‐ tem), so online learning can be a confusing name. Think of it as incremental learning.

![Screen Shot 2019-01-29 at 10.29.40 PM](https://ws3.sinaimg.cn/large/006tNc79ly1fzntr7wdg6j31400km0xd.jpg)

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. 

A big challenge with online learning is that if bad data is fed to the system, the sys‐ tem’s performance will gradually decline.If we are talking about a live system, your clients will notice. For example, bad data could come from a malfunctioning sensor on a robot, or from someone spamming a search engine to try to rank high in search results. To reduce this risk, you need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm).


### Instance-Based Versus Model-Based Learning

One more way to categorize Machine Learning systems is by how they generalize. Most Machine Learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to generalize to examples it has never seen before. Having a good performance measure on the training data is good, but insufficient; the true goal is to perform well on new instances.

There are two main approaches to generalization: instance-based learning and model-based learning.

#### Instance-based learning

Possibly the most trivial form of learning is simply to learn by heart. If you were to create a spam filter this way, it would just flag all emails that are identical to emails that have already been flagged by users—not the worst solution, but certainly not the best.

Instead of just flagging emails that are identical to known spam emails, your spam filter could be programmed to also flag emails that are very similar to known spam emails. This requires a measure of similarity between two emails. A (very basic) simi‐ larity measure between two emails could be to count the number of words they have in common. The system would flag an email as spam if it has many words in com‐ mon with a known spam email.

This is called instance-based learning: the system learns the examples by heart, then generalizes to new cases using a similarity measure (Figure 1-15).

![Screen Shot 2019-01-29 at 10.32.58 PM](https://ws1.sinaimg.cn/large/006tNc79ly1fzntunu4uxj31400gk0v6.jpg)

#### Model-based learning

Another way to generalize from a set of examples is to build a model of these exam‐ ples, then use that model to make predictions. This is called model-based learning (Figure 1-16).

![Screen Shot 2019-01-29 at 10.33.40 PM](https://ws2.sinaimg.cn/large/006tNc79ly1fzntvcq0wsj31400mmaeb.jpg)

## Main Challenges of Machine Learning

In short, since your main task is to select a learning algorithm and train it on some data, the two things that can go wrong are “bad algorithm” and “bad data”.

### Insufficient Quantity of Training data

For a toddler to learn what an apple is, all it takes is for you to point to an apple and say “apple” (possibly repeating this procedure a few times). Now the child is able to recognize apples in all sorts of colors and shapes. Genius.

Machine Learning is not quite there yet; it takes a lot of data for most Machine Learning algorithms to work properly. Even for very simple problems you typically need thousands of examples, and for complex problems such as image or speech recogni‐ tion you may need millions of examples (unless you can reuse parts of an existing model).

### Nonrepresentative Training Data

In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to. This is true whether you use instance-based learning or model-based learning.

It is crucial to use a training set that is representative of the cases you want to general‐ ize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise (i.e., nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.

### Poor-Quality Data

Obviously, if your training data is full of errors, outliers, and noise (e.g., due to poor- quality measurements), it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well. It is often well worth the effort to spend time cleaning up your training data. The truth is, most data scientists spend a significant part of their time doing just that. For example:

- If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually.
- If some instances are missing a few features (e.g., 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute alto‐ gether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it, and so on.

### Irrelevant Features

As the saying goes: garbage in, garbage out. Your system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones. A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process, called feature engineering, involves:

- Feature selection: selecting the most useful features to train on among existing features.
- Feature extraction: combining existing features to produce a more useful one (as we saw earlier, dimensionality reduction algorithms can help).
- Creating new features by gathering new data.

### Overfitting the Training Data

Overfitting means that the model performs well on the training data, but it does not generalize well.

Figure 1-22 shows an example of a high-degree polynomial life satisfaction model that strongly overfits the training data. Even though it performs much better on the training data than the simple linear model, would you really trust its predictions?

![Screen Shot 2019-01-29 at 10.40.28 PM](https://ws2.sinaimg.cn/large/006tNc79ly1fznu2fwmxnj31400ggtc8.jpg)

Overfitting happens when the model is too complex relative to the amount and noisiness of the training data. The possible solutions are:

- To simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training data or by constraining the model
- To gather more training data
- To reduce the noise in the training data (e.g., fix data errors and remove outliers)

Constraining a model to make it simpler and reduce the risk of overfitting is called regularization.

The amount of regularization to apply during learning can be controlled by a hyper‐ parameter. A hyperparameter is a parameter of a learning algorithm (not of the model). As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training. If you set the regularization hyperparameter to a very large value, you will get an almost flat model (a slope close to zero); the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution. Tuning hyperparameters is an important part of building a Machine Learning system (you will see a detailed example in the next chapter).

### Underfitting the Training Data

As you might guess, underfitting is the opposite of overfitting: it occurs when your model is too simple to learn the underlying structure of the data. For example, a lin‐ ear model of life satisfaction is prone to underfit; reality is just more complex than the model, so its predictions are bound to be inaccurate, even on the training examples.

The main options to fix this problem are:

- Selecting a more powerful model, with more parameters
- Feeding better features to the learning algorithm (feature engineering)
- Reducing the constraints on the model (e.g., reducing the regularization hyper‐ parameter)

### Stepping Back

By now you already know a lot about Machine Learning. However, we went through so many concepts that you may be feeling a little lost, so let’s step back and look at the big picture:

- Machine Learning is about making machines get better at some task by learning from data, instead of having to explicitly code rules.
- There are many different types of ML systems: supervised or not, batch or online, instance-based or model-based, and so on.
- In a ML project you gather data in a training set, and you feed the training set to a learning algorithm. If the algorithm is model-based it tunes some parameters to fit the model to the training set (i.e., to make good predictions on the training set itself), and then hopefully it will be able to make good predictions on new cases as well. If the algorithm is instance-based, it just learns the examples by heart and uses a similarity measure to generalize to new instances.
- The system will not perform well if your training set is too small, or if the data is not representative, noisy, or polluted with irrelevant features (garbage in, garbage out). Lastly, your model needs to be neither too simple (in which case it will underfit) nor too complex (in which case it will overfit).

## Testing and Validating

The only way to know how well a model will generalize to new cases is to actually try it out on new cases. One way to do that is to put your model in production and moni‐ tor how well it performs. This works well, but if your model is horribly bad, your users will complain—not the best idea.

A better option is to split your data into two sets: the training set and the test set. As these names imply, you train your model using the training set, and you test it using the test set. The error rate on new cases is called the generalization error (or out-of- sample error), and by evaluating your model on the test set, you get an estimation of this error. This value tells you how well your model will perform on instances it has never seen before.

If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the train‐ ing data.

So evaluating a model is simple enough: just use a test set. Now suppose you are hesi‐ tating between two models (say a linear model and a polynomial model): how can you decide? One option is to train both and compare how well they generalize using the test set.

Now suppose that the linear model generalizes better, but you want to apply some regularization to avoid overfitting. The question is: how do you choose the value of the regularization hyperparameter? One option is to train 100 different models using 100 different values for this hyperparameter. Suppose you find the best hyperparameter value that produces a model with the lowest generalization error, say just 5% error.

So you launch this model into production, but unfortunately it does not perform as well as expected and produces 15% errors. What just happened?

The problem is that you measured the generalization error multiple times on the test set, and you adapted the model and hyperparameters to produce the best model for that set. This means that the model is unlikely to perform as well on new data.

A common solution to this problem is to have a second holdout set called the validation set. You train multiple models with various hyperparameters using the training set, you select the model and hyperparameters that perform best on the validation set, and when you’re happy with your model you run a single final test against the test set to get an estimate of the generalization error.

To avoid “wasting” too much training data in validation sets, a common technique is to use cross-validation: the training set is split into complementary subsets, and each model is trained against a different combination of these subsets and validated against the remaining parts. Once the model type and hyperparameters have been selected, a final model is trained using these hyperparameters on the full training set, and the generalized error is measured on the test set.