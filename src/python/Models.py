from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import time
from sklearn import svm
from sklearn.lda import LDA
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.mixture.gmm import GMM
from scipy.sparse import issparse

def report_performance(gold, predicted):
    """
    Report performance (detailed - precision/recall/f1 and accuracy)
    :param gold: true values
    :param predicted: predicted values
    :return: string of performance
    """
    detailed = classification_report(gold, predicted)
    accuracy = "Accuracy: %f" % accuracy_score(gold, predicted)
    return detailed + "\n" + accuracy


def report_model(make_model, train_X, train_Y, test_X, test_Y, model_name = None):
    """
    Generic training and running of model, prints both training and test performance
    Returns test performance
    :param make_model: 0-argument lambda to create model. Model must have fit and predict methods
    :param train_X:
    :param train_Y:
    :param test_X:
    :param test_Y:
    :param model_name: optional name of model
    :return: string of test performance summary
    """
    model_name = "" if model_name == None else model_name
    print "Starting %s training: %s" % (model_name, time.asctime())
    classifier = make_model()
    classifier = classifier.fit(train_X, train_Y)
    print "End %s training: %s" % (model_name, time.asctime())
    print "----> Training Performance"
    train_classifications = classifier.predict(train_X)
    print report_performance(train_Y, train_classifications)
    print "----> Test Performance"
    test_classifications = classifier.predict(test_X)
    test_perf = report_performance(test_Y, test_classifications)
    print test_perf
    return test_perf


def report_BernoulliNB(train_X, train_Y, test_X, test_Y, **args):
    model_name = "Bernoulli Naive Bayes"
    make_model = lambda: BernoulliNB(**args) if args else BernoulliNB(alpha = 0.1)
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)

def report_GaussianNB(train_X, train_Y, test_X, test_Y):
    model_name = "Gaussian Naive Bayes"
    make_model = lambda: GaussianNB()
    # Gaussian NB wants matrices to be dense
    if issparse(train_X):
        train_X = train_X.todense()
    if issparse(test_X):
        test_X = test_X.todense()
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)

def report_MaxEnt(train_X, train_Y, test_X, test_Y, **args):
    model_name = "MaxEnt"
    make_model = lambda: LogisticRegression(**args) if args else LogisticRegression()
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)

# def report_GaussianNB(train_X, train_Y, test_X, test_Y):
#     # Naive Bayes model
#     print "start naive training: ", time.asctime()
#     naiveClassifier = BernoulliNB(alpha = 0.1)
#     naiveClassifier = naiveClassifier.fit(train_X, train_Y)
#     print "end naive training: ", time.asctime()
#     nbClassifications1 = naiveClassifier.predict(test_X)
#     print classification_report(test_Y, nbClassifications1)
#     print "Accuracy: %f" % accuracy_score(test_Y, nbClassifications1)

def report_SVM_linearK(train_X, train_Y, test_X, test_Y):
    model_name = "SVM with Linear Kernel"
    make_model = lambda: LinearSVC()
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)


def report_SVM_rbfK(train_X, train_Y, test_X, test_Y):
    model_name = "SVM with RBF Kernel"
    make_model = lambda: svm.SVC(C=1, kernel='rbf')
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)

def report_SVM_sigK(train_X, train_Y, test_X, test_Y):
    model_name = "SVM with Sigmoid Kernel"
    make_model = lambda: svm.SVC(C=1, kernel='sigmoid')
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)


def report_SVM_polyK(train_X, train_Y, test_X, test_Y):
    model_name = "SVM with Polynomial Kernel"
    make_model = lambda: svm.SVC(C=1, kernel='poly')
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)


def report_FisherLDA(train_X, train_Y, test_X, test_Y):
    model_name = "Fisher Linear Classifier"
    make_model = lambda: LDA()
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)


def report_GMM(train_X, train_Y, test_X, test_Y):
    model_name = "Multivariate Gaussian Classifier"
    make_model = lambda: GMM()
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)

def report_LogisticReg(train_X, train_Y, test_X, test_Y):
    model_name = "Logistic Regression Classifier"
    make_model = lambda: LogisticRegression()
    return report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)