from sklearn import svm

__author__ = 'xiuyanni'
import csv
import nltk
import getopt
import sys
import time
from nltk import FreqDist
import nltk.classify
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

bagofwords_features = []

from ReadData import *
import Globals


# def getwordsFromLabeledData(labeledData):


def getBagofwordsWordFeatures(labeledData):
    """

    :rtype : list
    """
    all_words = []
    for (words, sentiment) in labeledData:
        all_words.extend(words)
    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    return word_features


# return a dict, key is the word, value is True or False
def bagofwordsFeatureExtractor(document):

    document_words = set(document)
    features = {}
    for word in bagofwords_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def ngramFeatureExtractor(document):
    assert isinstance(document, list)

    #TODO: implement n gram feature extractor
    features = Counter()
    for i in range(len(document) - 1):
        features[(document[i], document[i+1])] += 1
    return features
# all words appear in the dict has a value True
#def word_feats(words):
#    return dict([(word, True) for word in words])


def calcualteAccuracy(clr, featureName, testRecords):
    correctCount = 0.0
    for testRecord in testRecords:
        if featureName == "bagofwords":
            predictLabel = clr.classify(bagofwordsFeatureExtractor(testRecord[0]))
        elif featureName == 'ngram':
            predictLabel = clr.classify(ngramFeatureExtractor(testRecord[0]))
        else:
            predictLabel = clr.classify(bagofwordsFeatureExtractor(testRecord[0])) # default feature is bag of words

        if predictLabel == testRecord[1]:
            correctCount += 1
    return correctCount / len(testRecords)

# not in use, use the crossvalidation in sklearn
def split_train_test(labeledBlogData, CUTOFF):

    blogDataWithEmotion = [b for b in labeledBlogData if b[1] == '1']

    blogDataWithoutEmotion = [b for b in labeledBlogData if b[1] == '0']
    emotionCutoff = int(len(blogDataWithEmotion) * CUTOFF)
    nonEmotionCutoff = int(len(blogDataWithoutEmotion) * CUTOFF)

    blogtrainingSet = blogDataWithEmotion[:emotionCutoff] + blogDataWithoutEmotion[:nonEmotionCutoff]
    blogtestingSet = blogDataWithEmotion[emotionCutoff:] + blogDataWithoutEmotion[nonEmotionCutoff:]
    return blogtrainingSet, blogtestingSet

def getX(data):
    return [obs[0] for obs in data]

def getY(data):
    return [obs[1] for obs in data]

def get_gaussianNB_model(tr_x, tr_y):
    clf = GaussianNB()
    clf.fit(tr_x, tr_y)
    return clf

def predict_GaussianNB(X_train, y_train, X_test, y_test):
    model = get_gaussianNB_model(X_train, y_train)
    predicted = model.predict(X_test)

    return accuracy_score(y_test, predicted)


def main(argv):

    import nltk

    twitterfile = ''
    blogtagfile = ''
    blogsentencefile = ''
    classifierName = ''
    featureName = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["classifierName=", "featureName="])
    except getopt.GetoptError:
        print 'nltkClassifiers.py -classifierName <naive, svm, logistic> -featureName <feature extractor>'
        sys.exit(2)
    for opt, arg in opts:
        if opt.lower() == '-h':
            print 'nltkClassifiers.py -classifierName <naive, svm, logistic> -featureName <feature extractor>'
            sys.exit()
        elif opt.lower() in ("--classifiername"):
            classifierName = arg
        elif opt.lower() in ("--featurename"):
            featureName = arg

    print 'Classifier: ', classifierName
    print 'Feature: ', featureName

    binarylabeledBlogData = prepareBlogData(Globals.BLOG_DATA['annotations'], Globals.BLOG_DATA['sentences'])
 #   twitterTestingData = prepareTwitterData(Globals.TWITTER_TEST)

  #  allSentences = getX(emotionlabeledBlogData)
  #  allLabels = getY(emotionlabeledBlogData)
    (trainingData, validationData) = train_test_split(binarylabeledBlogData, test_size = 0.2, random_state = 10)
 #   blogtrainingSet, blogtestingSet = split_train_test(emotionlabeledBlogData, 0.75)
    global bagofwords_features

    bagofwords_features = getBagofwordsWordFeatures(trainingData)
    # training_set is a list of tuple,
    # the first element of the tuple is a dict,
    # the second element of the tuple is the label
    if featureName == "bagofwords":
        training_set = nltk.classify.apply_features(bagofwordsFeatureExtractor, trainingData)
    elif featureName == "ngram":
        training_set = nltk.classify.apply_features(ngramFeatureExtractor, trainingData)
    else:
        training_set = nltk.classify.apply_features(bagofwordsFeatureExtractor, trainingData) # default feature is bag of words

    print "start training: ", classifierName,"with feature: ",featureName,  time.asctime()
    print "this is the traning set:", len(training_set)

    if classifierName.lower() == "naive":
        myClassifier = nltk.NaiveBayesClassifier.train(training_set)

        # naive bayes
    else:
        if classifierName.lower() == "linearsvm":
            myClassifier = nltk.classify.SklearnClassifier(LinearSVC())
            myClassifier.train(training_set)
        elif classifierName.lower() == "rbfsvm":
            myClassifier = nltk.classify.SklearnClassifier(svm.SVC(C=1, kernel='rbf'))
            myClassifier.train(training_set)
        elif classifierName.lower() == "logistic":
          #  print "great , I am training logistic"
            myClassifier = nltk.classify.SklearnClassifier(LogisticRegression())
            myClassifier.train(training_set)
        else:
            myClassifier = nltk.NaiveBayesClassifier.train(training_set) # default classifier, naive bayes

    print "end training ", classifierName, time.asctime()

    accuracy = calcualteAccuracy(myClassifier, featureName, validationData)
    print "The accuracy of ", classifierName, "on feature ", featureName, "is ", accuracy

if __name__ == "__main__":
   main(sys.argv[1:])