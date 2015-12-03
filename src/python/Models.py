from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
import time
from sklearn import svm
from sklearn.lda import LDA

def report_GaussianNB(train_X, train_Y, test_X, test_Y):
    # Naive Bayes model
    print "start naive training: ", time.asctime()
    naiveClassifier = BernoulliNB(alpha = 0.1)
    naiveClassifier = naiveClassifier.fit(train_X, train_Y)
    print "end naive training: ", time.asctime()
    nbClassifications1 = naiveClassifier.predict(test_X)
    print classification_report(nbClassifications1, test_Y)
    print "Accuracy: %f" % accuracy_score(test_Y, nbClassifications1)

def report_SVM_linearK(train_X, train_Y, test_X, test_Y):

    svmClassifier = LinearSVC()
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(svmClassifications, test_Y)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_SVM_rbfK(train_X, train_Y, test_X, test_Y):

    svmClassifier = svm.SVC(C=1, kernel='rbf')
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(svmClassifications, test_Y)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_SVM_sigK(train_X, train_Y, test_X, test_Y):

    svmClassifier = svm.SVC(C=1, kernel='sigmoid')
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(svmClassifications, test_Y)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_SVM_polyK(train_X, train_Y, test_X, test_Y):

    svmClassifier = svm.SVC(C=1, kernel='poly')
    print "start svm training: ", time.asctime()
    svmClassifier.fit(train_X, train_Y)
    print "end svm  training: ", time.asctime()
    svmClassifications = svmClassifier.predict(test_X)
    print classification_report(svmClassifications, test_Y)
    print "Accuracy: %f" % accuracy_score(test_Y, svmClassifications)

def report_FisherLDA(train_X, train_Y, test_X, test_Y):
    fisherClassifier = LDA()
    print "start lda training: ", time.asctime()
    fisherClassifier.fit(train_X, train_Y)
    print "end lda  training: ", time.asctime()
    fisherClassifications = fisherClassifier.predict(test_X)
    print classification_report(fisherClassifications, test_Y)
    print "Accuracy: %f" % accuracy_score(test_Y, fisherClassifications)

def report_GMM(train_X, train_Y, test_X, test_Y):
    print "Haven't implemented"

def report_LogisticReg(train_X, train_Y, test_X, test_Y):
    print "Haven't implemented"