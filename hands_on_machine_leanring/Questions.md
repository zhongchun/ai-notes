# Questions
[TOC]

1. 交叉验证有什么用，目的是什么？

    参见：[交叉验证原理](https://www.cnblogs.com/pinard/p/5992719.html)

2. 交叉验证和 GridSearch 异同？

    网格交叉验证用于找到一组最优的参数组合,使得在这组参数下模型效果最好;而交叉验证主要用于模型的效果验证,它是对于数据集的测试集和验证集的选择,也能够有效的防止模型过拟合.所以说,这两者是不同的概念.

3. ML flow ?

4. Regression and Classification methods?

- Regression methods:
    - LinearRegression
    - DecisionTreeRegressor
    - RandomForestRegressor
    - SVR

- Classification methods:
    - Logistic Regression
    - SVM
    - Decision Tree
    - Random Forest: SGDClassifier

5. Scikit-Learn Design
- Consistency: All objects share a consistent and simple interface
    - Estimators
    - Transformers
    - Predictors
