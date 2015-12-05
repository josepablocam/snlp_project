from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import time
from sklearn import svm
from sklearn.lda import LDA
import Features

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
    report_model(make_model, train_X, train_Y, test_X, test_Y, model_name)

def report_GaussianNB(train_X, train_Y, test_X, test_Y):
    model_name = "Gaussian Naive Bayes"
    make_model = lambda: GaussianNB()
    # Gaussian NB wants matrices to be dense
    report_model(make_model, train_X.todense(), train_Y, test_X.todense(), test_Y, model_name)

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

    svmClassifier = LinearSVC()
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(test_Y, svmClassifications)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_SVM_rbfK(train_X, train_Y, test_X, test_Y):

    svmClassifier = svm.SVC(C=1, kernel='rbf')
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(test_Y, svmClassifications)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_SVM_sigK(train_X, train_Y, test_X, test_Y):

    svmClassifier = svm.SVC(C=1, kernel='sigmoid')
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(test_Y, svmClassifications)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_SVM_polyK(train_X, train_Y, test_X, test_Y):

    svmClassifier = svm.SVC(C=1, kernel='poly')
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(test_Y, svmClassifications)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_FisherLDA(train_X, train_Y, test_X, test_Y):
    fisherClassifier = LDA()
    print "start lda training: ", time.asctime()
    fisherClassifier.fit(train_X, train_Y)
    print "end lda  training: ", time.asctime()
    fisherClassifications = fisherClassifier.predict(test_X)
    print classification_report(test_Y, fisherClassifications)
    print "Accuracy: %f" % accuracy_score(test_Y, fisherClassifications)

def report_GMM(train_X, train_Y, test_X, test_Y):
    print "Haven't implemented"

def report_LogisticReg(train_X, train_Y, test_X, test_Y):
    print "Haven't implemented"