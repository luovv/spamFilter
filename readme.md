# Spam Filter use Naive Bayes / Naive Bayes TFIDF / Adaboost Naive Bayes / Logistic Regression / KNN

## Environment:
The project can run under python2.7 with numpy, sklearn and matplotlib


## How to Run:
you can run the program by pass a filename parameter, it will print spam or ham
> E.g. python filter.py spam.txt
the program will read in the trained classifer, which is Naive Bayes classifier with tfidf and print the result


## What’s More?
You can use different method, retrain and validate the classifier by edit the last line of code

### EmailFilter take three parameters.
* 1, The first parameter is what algorithm the classifier uses. You can use one of following method:
	NAIVE_BAYES
	NAIVE_BAYES_TFIDF
	LOGISTIC_REGRESSION
	ADABOOST_NAIVE_BAYES
	BASELINE
	KNN
* 2, The second parameter(boolean) is whether conduct cross validation. The default value is False.
* 3, the third parameter(boolean) is whether train the classifer. The default value is False.

### If you want the program to plot ROC Curve,call drawGraph() function.
drawGraph take 2 parameters:
* 1, graph title(str)
* 2, list of method(list). You can use one of following method:
	NAIVE_BAYES
	NAIVE_BAYES_TFIDF
	LOGISTIC_REGRESSION
	ADABOOST_NAIVE_BAYES
	BASELINE
	KNN

> E.g. EmailFilter().drawGraph('ROC', [NAIVE_BAYES, KNN]) plots the ROC curves of Naive Bayes and KNN classifiers