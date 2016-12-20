import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sys
from sklearn import metrics
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import random
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

NAIVE_BAYES = 0
NAIVE_BAYES_TFIDF = 1
LOGISTIC_REGRESSION = 2
ADABOOST_NAIVE_BAYES = 3
BASELINE = 4
KNN = 5

class EmailClassifier:
    def __init__(self, spam, ham, pSpam, count_vect):
        self.spam = spam
        self.ham = ham
        self.pSpam = pSpam
        self.count_vect = count_vect
        self.DS = None

    def predict(self, email):
        words = self.count_vect.transform([email])
        words = words.toarray()[0]

        p0 = sum(words * self.ham) + np.log(1 - self.pSpam)
        if self.DS is not None:
            p1 = sum(words * self.spam * self.DS) + np.log(self.pSpam)
        else:
            p1 = sum(words * self.spam) + np.log(self.pSpam)

        print 'ham' if p0 > p1 else 'spam'
        return (p0, p1, 0) if p0 > p1 else (p0, p1, 1)

class BaseLine:
    def predict(self, email):
        rand = int(random.random() * 10)
        print 'ham' if rand<8 else 'spam'
        return (0,0,0) if rand < 9 else (1,1,1)


class EmailFilter:
    def __init__(self, method=-1, validation=False, train=False):
        self.method = method
        trainFile = load_files('public')

        if method == NAIVE_BAYES:
            self.count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
            if validation:
                self.crossValidation(trainFile)
            else:
                filename = sys.argv[1]
                if train:
                    self.train(trainFile.data, trainFile.target, 'NBClassifier.txt')
                clf = self.loadClassifier('NBClassifier.txt')
                self.predict(clf, filename)

        elif method == NAIVE_BAYES_TFIDF:
            self.count_vect = TfidfVectorizer(stop_words="english", decode_error='ignore')
            if validation:
                self.crossValidation(trainFile)
            else:
                filename = sys.argv[1]
                if train:
                    self.train(trainFile.data, trainFile.target, 'TFIDFClassifier.txt')
                clf = self.loadClassifier('TFIDFClassifier.txt')
                self.predict(clf, filename)
        elif method == LOGISTIC_REGRESSION:
            self.count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
            if validation:
                self.logisticRegressionCrossValidation(trainFile)
            else:
                filename = sys.argv[1]
                if train:
                    self.logisticRegressionTrain(trainFile.data, trainFile.target, 'LRClassifier.txt')
                clf = self.loadClassifier('LRClassifier.txt')
                self.predict(clf, filename)
        elif method == ADABOOST_NAIVE_BAYES:
            self.count_vect = TfidfVectorizer(stop_words="english", decode_error='ignore')
            # clf = self.getDS(trainFile.data, trainFile.target)
            if validation:
                self.crossValidation(trainFile)
            else:
                filename = sys.argv[1]
                if train:
                    self.adaboostTrain(trainFile.data, trainFile.target, 'AdaboostNBClassifier.txt')
                clf = self.loadClassifier('AdaboostNBClassifier.txt')
                self.predict(clf, filename)
        elif method == BASELINE:
            if validation:
                self.crossValidation(trainFile)
            else:
                print 'Baseline is not a classifier. You can only apply validation!'

        elif method == KNN:
            self.count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
            if validation:
                self.knnCrossValidation(trainFile)
            else:
                filename = sys.argv[1]
                if train:
                    self.knnTrain(trainFile.data, trainFile.target, 'KNNClassifier.txt')
                clf = self.loadClassifier('KNNClassifier.txt')
                self.predict(clf, filename)

    def predict(self, clf, filename):
        if self.method == LOGISTIC_REGRESSION or self.method == KNN:
            email = self.readTestFile(filename)
            email = self.count_vect.transform([email])
            result = clf.predict(email)
            print 'ham' if result[0]==0 else 'spam'
        else:
            email = self.readTestFile(filename)
            clf.predict(email)
    

    def loadClassifier(self,filename):
        f = open(filename,'r')
        data = f.read()
        f.close()
        clf = pickle.loads(data)
        return clf

    def readTestFile(self, filename):
        f = open(filename)
        email = f.read()
        email = re.sub('[^a-zA-Z\s]+', '', email)
        email = re.sub('[\s]{2,}', ' ', email)
        f.close()
        return email

    def train(self, data, l, outputFile=None):
        for i, item in enumerate(data):
            data[i] = re.sub('[^a-zA-Z\s]+', '', item)
            data[i] = re.sub('[\s]{2,}', ' ', data[i])

        X = self.count_vect.fit_transform(data)
        pSpam = sum(l) / float(len(l))

        spamWordsCountList = np.ones(X.shape[1])
        hamWordsCountList = np.ones(X.shape[1])

        spamCount = 0
        hamCount = 0
        for i in range(X.shape[0]):
            if l[i] == 1:
                for j in X[i].indices:
                    if self.method == NAIVE_BAYES_TFIDF or self.method == ADABOOST_NAIVE_BAYES:
                        spamWordsCountList[j] += -np.log(X[i, j])
                    else:
                        spamWordsCountList[j] += X[i, j]
                spamCount += 1
            else:
                for j in X[i].indices:
                    if self.method == NAIVE_BAYES_TFIDF or self.method == ADABOOST_NAIVE_BAYES:
                        hamWordsCountList[j] += -np.log(X[i, j])
                    else:
                        hamWordsCountList[j] += X[i, j]
                hamCount += 1
        spam = spamWordsCountList/spamCount
        ham = hamWordsCountList/hamCount

        spam = np.log(spam)
        ham = np.log(ham)
        clf = EmailClassifier(spam, ham, pSpam, self.count_vect)
        if outputFile:
            f = open(outputFile, 'w')
            f.write(pickle.dumps(clf))
            f.close()

        return clf

    def crossValidation(self, trainFile):
        X =  trainFile.data
        l = trainFile.target

        fold = 10
        result = []
        total_test = []
        total_prediction = []

        for i in range(fold):
            p1 = i*len(X)/fold
            p2 = (i+1)*len(X)/fold
            d_test = X[p1:p2]
            d_train = np.concatenate((X[0:p1], X[p2:len(X)]))
            l_test = l[p1:p2]
            l_train = np.concatenate((l[0:p1], l[p2:len(l)]))
            d_predicted = []
            if self.method == ADABOOST_NAIVE_BAYES:
                clf = self.adaboostTrain(d_train, l_train)
            elif self.method == NAIVE_BAYES or self.method == NAIVE_BAYES_TFIDF:
                clf = self.train(d_train, l_train)
            elif self.method == BASELINE:
                clf = BaseLine()

            for i in d_test:
                email = re.sub('[^a-zA-Z\s]+', '', i)
                email = re.sub('[\s]{2,}', ' ', email)
                p0, p1, prediction = clf.predict(email)
                d_predicted.append(prediction)
            result.append(np.mean(d_predicted == l_test))
            total_prediction+=d_predicted
            total_test+=list(l_test)
            # print metrics.confusion_matrix(l_test, d_predicted)
            # print metrics.classification_report(l_test, d_predicted)

        fpr, tpr, thresholds = roc_curve(total_test, total_prediction)

        print metrics.confusion_matrix(total_test, total_prediction)
        print metrics.classification_report(total_test, total_prediction)
        print 'mean squared error: %s' % mean_squared_error(total_test, total_prediction)
        print 'mean accuracy: %s' % np.mean(result)

        print 'calibrated mean squared error: %s' % mean_squared_error(total_test, self.calibration(total_prediction, total_test))
        return fpr, tpr

    def logisticRegressionCrossValidation(self, trainFile):
        X = trainFile.data
        l = trainFile.target

        fold = 10
        result = []
        total_test = []
        total_prediction = []
        for i in range(fold):
            p1 = i * len(X) / fold
            p2 = (i + 1) * len(X) / fold
            d_test = X[p1:p2]
            d_train = np.concatenate((X[0:p1], X[p2:len(X)]))
            l_test = l[p1:p2]
            l_train = np.concatenate((l[0:p1], l[p2:len(l)]))

            clf = self.logisticRegressionTrain(d_train, l_train)
            for i, item in enumerate(d_test):
                d_test[i] = re.sub('[^a-zA-Z\s]+', '', item)
                d_test[i] = re.sub('[\s]{2,}', ' ', d_test[i])
            d_test = self.count_vect.transform(d_test)
            d_predicted = clf.predict(d_test)

            result.append(np.mean(d_predicted == l_test))
            total_prediction += list(d_predicted)
            total_test += list(l_test)
            print metrics.confusion_matrix(l_test, d_predicted)
            print metrics.classification_report(l_test, d_predicted)

        fpr, tpr, thresholds = roc_curve(total_test, total_prediction)

        print metrics.confusion_matrix(total_test, total_prediction)
        print metrics.classification_report(total_test, total_prediction)
        print 'mean squared error: %s' % mean_squared_error(total_test, total_prediction)
        print 'mean accuracy: %s' % np.mean(result)
        return fpr, tpr

    def knnCrossValidation(self, trainFile):
        X = trainFile.data
        l = trainFile.target

        fold = 10
        result = []
        total_test = []
        total_prediction = []
        for i in range(fold):
            p1 = i * len(X) / fold
            p2 = (i + 1) * len(X) / fold
            d_test = X[p1:p2]
            d_train = np.concatenate((X[0:p1], X[p2:len(X)]))
            l_test = l[p1:p2]
            l_train = np.concatenate((l[0:p1], l[p2:len(l)]))

            clf = self.knnTrain(d_train, l_train)
            for i, item in enumerate(d_test):
                d_test[i] = re.sub('[^a-zA-Z\s]+', '', item)
                d_test[i] = re.sub('[\s]{2,}', ' ', d_test[i])
            d_test = self.count_vect.transform(d_test)
            d_predicted = clf.predict(d_test)

            result.append(np.mean(d_predicted == l_test))
            total_prediction += list(d_predicted)
            total_test += list(l_test)
            print metrics.confusion_matrix(l_test, d_predicted)
            print metrics.classification_report(l_test, d_predicted)

        fpr, tpr, thresholds = roc_curve(total_test, total_prediction)

        print metrics.confusion_matrix(total_test, total_prediction)
        print metrics.classification_report(total_test, total_prediction)
        print 'mean squared error: %s' % mean_squared_error(total_test, total_prediction)
        print 'mean accuracy: %s' % np.mean(result)
        return fpr, tpr

    def adaboostTrain(self, data, l, outputFile=None):
        for i, item in enumerate(data):
            data[i] = re.sub('[^a-zA-Z\s]+', '', item)
            data[i] = re.sub('[\s]{2,}', ' ', data[i])
        X = self.count_vect.fit_transform(data)
        testData = []
        testLabes = []
        size = X.shape[0]/2
        for i in range(size):
            pos = int(random.uniform(0, X.shape[0]))
            testData.append(data[pos])
            testLabes.append(l[pos])
            np.delete(data, pos)
            np.delete(l, pos)

        clf = self.train(data, l)
        clf.DS = np.ones(X.shape[1])
        DS = np.ones(X.shape[1])

        emailCount = self.count_vect.transform(testData)
        minErrorRate = np.inf
        for i in range(30):
            oldDS = DS
            error = 0
            for j in range(size):
                ph, ps, predicted = clf.predict(testData[j])
                if not predicted == testLabes[j]:
                    error += 1
                    alpha = (ps-ph)/2
                    col = emailCount[j].indices
                    if alpha > 0:
                        DS[col] = np.abs((DS[col] - np.exp(-alpha)) / DS[col])
                    else:
                        DS[col] = (DS[col] + np.exp(alpha)) / DS[col]

            errorRate = error/float(size)
            if errorRate < minErrorRate:
                minErrorRate = errorRate
                clf.DS = oldDS
            # print ' %d: error %d, errorRate %f' % (i, error, errorRate)
            if errorRate == 0:
                break

        if outputFile:
            f = open(outputFile, 'w')
            f.write(pickle.dumps(clf))
            f.close()
        return clf

    def logisticRegressionTrain(self, data, l, outputFile=None):
        for i, item in enumerate(data):
            data[i] = re.sub('[^a-zA-Z\s]+', '', item)
            data[i] = re.sub('[\s]{2,}', ' ', data[i])
        X = self.count_vect.fit_transform(data)
        clf = LogisticRegression().fit(X, l)
        if outputFile:
            f = open(outputFile, 'w')
            f.write(pickle.dumps(clf))
            f.close()
        return clf

    def knnTrain(self, data, l, outputFile=None):
        for i, item in enumerate(data):
            data[i] = re.sub('[^a-zA-Z\s]+', '', item)
            data[i] = re.sub('[\s]{2,}', ' ', data[i])
        X = self.count_vect.fit_transform(data)
        clf = KNeighborsClassifier(n_neighbors=3).fit(X, l)
        if outputFile:
            f = open(outputFile, 'w')
            f.write(pickle.dumps(clf))
            f.close()
        return clf

    def drawGraph(self, title, methods):
        import matplotlib.pyplot as plt
        trainFile = load_files('public')
        if NAIVE_BAYES in methods:
            self.method = NAIVE_BAYES
            self.count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
            fpr, tpr = self.crossValidation(trainFile)
            mean_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='indigo', label='Naive Bayes(mean accuracy: %0.2f)' % mean_auc, lw=2)

        if NAIVE_BAYES_TFIDF in methods:
            self.method = NAIVE_BAYES_TFIDF
            self.count_vect = TfidfVectorizer(stop_words="english", decode_error='ignore')
            fpr, tpr = self.crossValidation(trainFile)
            mean_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='blue', label='Naive Bayes TFIDF(mean accuracy: %0.2f)' % mean_auc, lw=2)

        if BASELINE in methods:
            self.method = BASELINE
            fpr, tpr = self.crossValidation(trainFile)
            mean_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='seagreen', label='Baseline(mean accuracy: %0.2f)' % mean_auc, lw=2)

        if ADABOOST_NAIVE_BAYES in methods:
            self.method = ADABOOST_NAIVE_BAYES
            self.count_vect = TfidfVectorizer(stop_words="english", decode_error='ignore')
            fpr, tpr = self.crossValidation(trainFile)
            mean_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', label='Adaboost Naive Bayes(mean accuracy: %0.2f)' % mean_auc, lw=2)

        if LOGISTIC_REGRESSION in methods:
            self.method = LOGISTIC_REGRESSION
            self.count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
            fpr, tpr = self.logisticRegressionCrossValidation(trainFile)
            mean_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='navy', label='Logistic_Regression(mean accuracy: %0.2f)' % mean_auc, lw=2)

        if KNN in methods:
            self.method = KNN
            self.count_vect = CountVectorizer(stop_words="english", decode_error='ignore')
            fpr, tpr = self.knnCrossValidation(trainFile)
            mean_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='teal', label='Logistic_Regression(mean accuracy: %0.2f)' % mean_auc, lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def calibration(self, V, L):
        out = np.array(V)
        L = np.array(L)
        target = L == 1
        prior1 = np.float(np.sum(target))
        prior0 = len(target) - prior1
        A = 0
        B = np.log((prior0 + 1) / (prior1 + 1))
        hiTarget = (prior1 + 1) / (prior1 + 2)
        loTarget = 1 / (prior0 + 2)
        # print 't+: ', hiTarget
        # print 't-: ', loTarget
        labda = 1e-3
        olderr = 1e300
        pp = np.ones(out.shape) * (prior1 + 1) / (prior0 + prior1 + 2)
        T = np.zeros(target.shape)
        for it in range(1, 100):
            a = 0
            b = 0
            c = 0
            d = 0
            e = 0
            for i in range(len(out)):
                if target[i]:
                    t = hiTarget
                    T[i] = t
                else:
                    t = loTarget
                    T[i] = t
                d1 = pp[i] - t
                d2 = pp[i] * (1 - pp[i])
                a += out[i] * out[i] * d2
                b += d2
                c += out[i] * d2
                d += out[i] * d1
                e += d1
            if (abs(d) < 1e-9 and abs(e) < 1e-9):
                break
            oldA = A
            oldB = B
            err = 0
            count = 0
            while 1:
                det = (a + labda) * (b + labda) - c * c
                if det == 0:
                    labda *= 10
                    continue
                A = oldA + ((b + labda) * d - c * e) / det
                B = oldB + ((a + labda) * e - c * d) / det
                err = 0
                for i in range(len(out)):
                    p = self.sigmoid(out[i], A, B)
                    pp[i] = p
                    t = T[i]
                    if p == 0:
                        err -= t * (-200) + (1 - t) * np.log(1 - p)
                    elif p == 1:
                        err -= t * np.log(p) + (1 - t) * (-200)
                    else:
                        err -= t * np.log(p) + (1 - t) * np.log(1 - p)
                if err < olderr * (1 + 1e-7):
                    labda *= 0.1
                    break
                labda *= 10
                if labda > 1e6:
                    break
                diff = err - olderr
                scale = 0.5 * (err + olderr + 1)
                if diff > -1e-3 * scale and diff < 1e-7 * scale:
                    count += 1
                else:
                    count = 0
                olderr = err
                if count == 3:
                    break
        # print 'A: %s'%A
        # print 'B: %s' % B
        # print np.array(V)[0:50]
        # print np.array(self.sigmoid(np.array(V), A, B))[0:50]
        return np.array(self.sigmoid(np.array(V), A, B))
        # print A, B

    def sigmoid(self, v, A, B):
            return 1 / (1 + np.exp(v * A + B))
        # return A, B


# EmailFilter(NAIVE_BAYES, validation=True)
# EmailFilter(NAIVE_BAYES_TFIDF, validation=True)
# EmailFilter(LOGISTIC_REGRESSION, validation=True)
# EmailFilter(ADABOOST_NAIVE_BAYES, validation=True)
# EmailFilter(BASELINE, validation=True)
# EmailFilter(KNN, validation=True)
# EmailFilter().drawGraph('ROC', [NAIVE_BAYES, KNN, BASELINE])


# EmailFilter(NAIVE_BAYES, train=False)
EmailFilter(NAIVE_BAYES_TFIDF, train=False)
# EmailFilter(ADABOOST_NAIVE_BAYES, train=False)