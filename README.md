# Supervised Learning. Decision Trees and ensembles.

Summary: this project is an introduction to decision trees (CART) and ensemble methods (Random Forest, GBDT).

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Decision Tree](#decision-tree) \
    2.2. [Pros of CART](#pros-of-cart) \
    2.3. [Cons of CART](#cons-of-cart) 
3. [Chapter III. Goal](#chapter-iii-goal) 
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

We have already discussed different algorithms to handle classification/regression problems. 
However, the most powerful ML algorithms for tabular data problems are based on Decision Trees 
(or **classification and regression trees, CART**). CART models are easy-to-understand models with a high level of explainability that mimic 
(or automate) human decision processes. CART don’t provide good accuracy in most cases but **ensembling** techniques allow us to build solid 
and accurate models based on sets of  trained CART models.

## Chapter II. Introduction

Classification and regression trees work by recursively partitioning the input space (feature space) and defining 
a local model in each partition (region). In every node of the tree we try to come up with the next split suggestion: 
what type of split would reduce entropy or deviation in the subsequent nodes? The overall model can be represented by a 
(binary) tree, with one leaf per region.

### Decision Tree

The following picture (animal classification) gives an overview of how the decision tree works: 
we separate input into subspaces recursively; each partition “shifts” us a bit towards the right answer. 
On step 0 (root node) we don’t know the type of an animal, on level 1 non-yellow animals go to the right branch, 
now we can separate animals even more (based on height, weight, etc.) and so on.

![decision_tree_animals](misc/images/decision_tree_animals.png)

*Source [x] (https://www.simplilearn.com/tutorials/machine-learning-tutorial/decision-tree-in-python)*

There is decision process for Elephant as input:
1. Elephant is not yellow so it goes to the right branch.
2. Elephant is very tall, so it again goes to the right branch.

The following picture shows another decision tree example, for classic Iris dataset:

![iris_dataset](misc/images/iris_dataset.png)

*Source: [x] Murphy*

Picture (a) provides an example of a decision tree for the Iris classification problem. 
Picture (b) shows the decision rule induced by tree (a). 
We can see that CART partitioned input space using axis-parallel splits. 
Every region corresponds to constant value prediction: for example, any input instance in the orange region is labeled 
by “setosa”. CART can be viewed as a linear model on highly non-linear indicator features 
(every partition is a combination of comparisons).

Discrete tree structure leads to non differentiable loss function for our problem.
Finding the optimal partitioning of the data is an NP-complete problem. 
The standard practice is to use a **greedy** procedure: we iteratively grow the tree one node at a time, 
choosing the best possible split on the current step.

The training procedure consist of the following steps:
1. On any step, we compute reduction of entropy (for classification problem) or reduction of standard deviation (for regression problem) for every possible split. We want inputs that fall into our new 2 nodes to be more homogeneous in terms of target variable. 
   <br/><br/>Imagine that the task is to classify an animal as elephant or non-elephant based on 3 features: color, weight, height. Possible splits include any color, all weight values, etc. Splitting by color is informative (elephants are gray and foxes aren’t) but is not the best: there are too many different species with gray color; left subtree (non-grey animals) wouldn’t have any elephants but right subtree (gray animals) would contain not only elephants but lots of other species.
     <br/><br/>In this task, **[weight > 1000 kg]** is the best possible split: elephants are the heaviest land animals.
 <br/><br/>That split gives maximal reduction of uncertainty in 2 new nodes:
**[weight >= 1000 kg]** has only elephants, **[weight < 1000 kg]** doesn’t contain elephants at all (or maybe some marginal amount of tiny, light-weighted elephants).
2. After introducing 2 new nodes we recursively start the splitting process in them.
3. We finish splitting after some stopping criteria is met (like predefined maximum depth of the tree, too low uncertainty reduction in the following splits, etc).

### Pros of CART

Decision trees have the following strong sides:
* CART models are a good choice when you need an interpretable model: explaining prediction of tree model means traversing through tree’s splits.
* CART can work with mixed distribution datasets. CARTS are almost insensitive to monotone transformations. Preprocessing techniques such as normalization, scaling, and binning are not usually applied in case of CARTS usage.
* Outliers in features also don’t have a strong impact on CART performance.


### Cons of CART

In real-life problems CART is rarely used as standalone model because of it’s significant drawbacks:
* CART models are prone to overfitting because of the tree's greedy learning process.
* Trees are unstable: small changes to the input data can have large effects on the structure of the tree, due to the hierarchical nature of the tree-growing process, causing errors at the top to affect the rest of the tree [Murphy].

### Ensemble methods

It is easy to overfit with CART models. However, CART are actively used as building blocks for other, more complex and accurate tree-based models. Instability of CART may be mitigated with averaging a lot of different CART models trained on the same dataset.

**Random Forest** model is one example of that idea:
1. We create a bunch (usually 100-1000) CART models using random subsamples of training data. Typically, we subsample some fraction (⅔, for example) of random rows and M random columns from our dataset and use very deep trees. We end with a bunch (**“bag”**) of trained trees that are potentially overfitted and have a low accuracy.
2. Output of the Random Forest model is *averaging predictions of every trained tree*. Every tree is overfitted to the part of the training data but their average provides much more stable and accurate prediction.

Fitting different base models to different randomly sampled versions of the data is called **bagging**. 
Bagging relies on the assumption that re-running the same learning algorithm on different subsets of the data will 
result in sufficiently diverse base models. Random Forest model is one example of using bagging.

Every tree separates input space into boxes. Averaging bundles of trees leads to a more smooth decision boundary. 
The following picture illustrates that principle:

![ensemble_methods](misc/images/ensemble_methods.png)

Picture (a) illustrates the decision boundary for a single tree, (b) corresponds to averaging a bag of 10 trees, 
(c) corresponds to a Random Forest model with 50 trees. We see that (c) doesn’t have border artifacts like on picture (a).

**ExtraTrees** is another type of tree-based model [ET paper]. It is basically Random Forests without subsampling on rows 
(we subsample only features) that uses **extra-randomized trees** instead of CART. 
Extremely randomized tree is a CART model when you pick up one random split value for every feature: 
we search for the best splits looking only at 1 random split for every feature. 
In practice, the ExtraTrees model is less accurate than RandomForest but less prone to overfitting.

Unfortunately, methods based on multiple trees lose their interpretability properties: 
we trade high accuracy for less clear decision process. 
There are various approaches for explaining black-box algorithms:SHAP, LIME, etc.

### Gradient Boosting

Imagine a trained decision tree classifier. We may improve the performance of our model by adding another decision tree 
that tries to fix errors induced by the first tree. 
Then we can continue this process by learning another tree that increases the accuracy of our ensemble. 
The result is called an **additive model**.

![gradient_boosting](misc/images/gradient_boosting.png)

*Source: [Murphy]*

The picture illustrates an additive model with 3 learned CART. Tree #2 tries to reduce an error of tree #1. 
Tree #3 decreases errors imposed by an ensemble of tree #1 and tree #2. How do we learn such an additive model?

From previous lessons we know Stochastic Gradient Descent. Updating parameters based on steps towards loss function 
anti-gradient makes errors of the ML algorithm smaller.

The same principle can be applied in additive modeling:

* for differentiable loss function, compute the anti-gradient of an error at the current stage and train a new decision tree to approximate the anti-gradient.
    <br/><br/>It is equivalent to a supervised problem with:
    <br/><br/>**x** as input and negative gradient by A of **Loss(y, A(x))** as a target, where **A(x)** – prediction of current ensemble for instance **x**.
* Add the output (with a small coefficient) of the new tree to models’ output in the current ensemble.
* We have reduced our loss by adding a new tree to the ensemble. We can repeat previous steps until we meet some stopping criteria (maximum number of iterations, too low improvements, etc.).

GBDT training procedure is clearly explained in [4].

Approximation of anti-gradient requires using regression trees. 
Resulting model is called **Gradient Boosting Decision Trees**. 
In general, any types of trees (or different ML models) can be used as base learners but trees are the most convenient. 
In practice, different implementations of GBDT include different types of base trees.

The final form of the GBDT model reminds RandomForest: it is a linear combination of many trees. 
However, in the GBDT training scheme every tree is built on top of the result of the previous trees; on the other hand, 
in Random Forest individual trees aren’t directly related: we train different CART then average. Usually, 
GBDT exploits some tricks from Random Forest: subsampling training instances and features.

### Modern Gradient Boosting Implementations

Modern GBDT packages provide users with tens of various parameters that control the creation (learning) 
process of every tree in the ensemble. There are different efficient implementations of GBDT: 
3 of the most popular GBDT implementations today are LightGBM, Catboost and XGBoost. 
They all have hundreds of tweakable parameters, different tree structures, learning regimes, etc.

Brief overview of differences:
* All three use different trees or learning strategies: depth-wise (XGBoost), leaf-wise (LightGBM), Oblivious trees (Catboost).
* LightGBM supports ==/!= splits for categorical data.
* Catboost is devoted to work with categorical features: smart target-encoding is implemented.
* LightGBM and XGBoost have DART mode: learning scheme with dropping out subsample of already learned trees.
* LightGBM supports ExtraTrees and Linear Trees (we replace prediction with mean target in leaf with prediction of LinearRegression)

These articles [1], [2], [3], give a good overview of the best GBDT libraries.

GBDT algorithms are capable of learning extremely complex functions of inputs with mixed distributions. 
GBDT algorithms are considered to be the best choice (on average) for tabular data with mixed distributions. 
Additive nature and usage of multiple trees makes GBDT models extremely sophisticated to have a clear interpretation.

### References

[1] https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

[2] https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc

[3] https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm

[4] https://alexanderdyakonov.files.wordpress.com/2017/06/book_boosting_pdf.pdf

[5] https://link.springer.com/content/pdf/10.1007/s10994-006-6226-1.pdf?pdf=button

[6] https://www.simplilearn.com/tutorials/machine-learning-tutorial/decision-tree-in-python

## Chapter III. Goal

The goal of this task is to get a deep understanding of tree-based models for classification 
(CART, Random Forest, GBDT) and modern implementations (LightGBM, XGBoost, Catboost). 

## Chapter IV. Instructions

* This project will only be evaluated by humans. You are free to organize and name
your files as you desire.
* Here and further we use Python 3 as the only correct version of Python.
* For training deep learning algorithms you can try [Google Colab](https://colab.research.google.com). It offers kernels
(Runtime) with GPU for free that is faster than CPU for such tasks.
* The norm is not applied to this project. Nevertheless, you are asked to be clear
and structured in the conception of your source code.
* Store the datasets in the subfolder data.

## Chapter V. Task

1. Download data from [Don’tGetKicked competition](https://www.kaggle.com/c/DontGetKicked).Design train/validation/test split.
    <br/><br/>Use “PurchDate” field for splitting, test must be later in time than validation, the same goes for validation and train: train.PurchDate < valid.PurchDate < test.PurchDate. 
<br/>Use the first 33% of dates for train, last 33% of dates for test, and middle 33% for validation set. *Don’t use the test dataset until the end!*
<br/><br/>Use LabelEncoder or OneHotEncoder from sklearn to preprocess categorical variables. Be careful with data leakage (fit Encoder on train and apply on validation & test). Consider another encoding approach if you meet new categorical values in valid & test (unseen in the training dataset), for example: https://contrib.scikit-learn.org/category_encoders/count.html
2. Create a Python class for Decision Tree Classifier and Decision Tree Regressor (MSE loss).
<br/>It should support *fit*, *predict_proba* and *predict methods*. Also, maximal depth (max_depth) must be a parameter of your class. Use Gini impurity criterion as criterion for choice of splitting.
<br/><br/>Here is the blueprint:
<br/><br/>**model = DecisionTreeClassifier(max_depth=7)**
<br/>**model.fit(Xtrain, ytrain)**
<br/>**model.predict_proba(Xvalid)**<br/><br/>
   * Create a separate class for Node. It should be able to contain data (sample features and targets), compute Gini impurity and have pointers to children (left and right nodes). For Regressor use standard deviation instead of Gini impurity. 
   * Implement a function that finds the best possible split in the current node. 
   * Combine previous steps into your working Decision Tree Classifier.
   * *Implement extra-randomized tree via designing another function for best-split search.
3. With your DecisionTree module you have to receive at least 0.1 Gini score on the validation dataset.
4. Use DecisionTreeClassifier from sklearn and check its performance on validation dataset. Is it better than your module? If it is, why?
5. Implement RandomForestClassifier and check its performance. You have to improve the result of one single tree and receive at least 0.15 Gini score on the validation dataset. Be able to set fixed random seed.
6. Using your DecisionTree design class for GDBT classifier. This class must have *max_depth*, *number_of_trees* and *max_features* attributes. You have to compute the gradient of the binary cross entropy loss function and implement incremental learning: train the next tree using the results of previous trees.
7. Use LightGBM, Catboost and XGBoost to fit on a training set and prediction on validation dataset. Check documentation of the libraries, and fine-tune the algorithms for the task.
Write down key differences between all implementations. Analyze special features of every algorithm (how do “categorical feature” work in Catboost, what is DART mode in XGBoost)?
What GBDT model gives the best result? Can you explain why?
8. Pick up the best model and estimate its performance on the test dataset: check Gini scores on all three datasets for your best model: train Gini, valid Gini, test Gini. Can you see any drop in performance when comparing valid quality vs test quality? Is your model overfitted or not? Explain.
9*. Implement ExtraTreesClassifier and check its performance. You have to improve the result of one single tree and receive at least 0.12 Gini score on the validation dataset.

## Chapter VI. Bonus part

Bonus part marked with * in [Chapter V. Task](#chapter-v-task)


>Please leave feedback on the project in the [feedback form.](https://forms.yandex.ru/cloud/646b46eff47e732ee1311d6f/) 

