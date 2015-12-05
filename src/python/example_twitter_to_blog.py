# Example showing how to use various modules to create (terrible) model


from collections import defaultdict
# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from ReadData import *
from Features import *
# Models and Features
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
# global paths
import Globals 
import time
 
 
# Roughly following 
# http://scikit-learn.org/stable/auto_examples/text/ \
# document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py
# for purposes of sklearn usage
def experiment():
    # Use twitter + wiki data as training data, we test on blog
    # we need everything to be utf8
    print "Collecting training and test data"
    twitterEmotion = to_utf8(prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False))
    wikiNoEmotion = prepareWikiData(Globals.WIKI_TRAIN, splitwords = False)
    allTrainingData = twitterEmotion + wikiNoEmotion
    (trainingData, validationData) = train_test_split(allTrainingData, test_size = 0.2, random_state = 10)
    trainingData = allTrainingData
    labeledBlogData = to_utf8(prepareBlogData(Globals.BLOG_DATA['annotations'], Globals.BLOG_DATA['sentences'], splitwords = False))
    
    # Featurize
    print "Featurizing"    
    # create training data with sparse matrix
    training_set_X = bagOfWordsSkLearn(getX(trainingData))
    training_set_Y = getY(trainingData)

    test_set_X = bagOfWordsSkLearn(getX(labeledBlogData))
    test_set_Y = getY(labeledBlogData)
    
    # Naive Bayes model
    print "start naive training: ", time.asctime()
    naiveClassifier = BernoulliNB(alpha = 0.1)
    naiveClassifier = naiveClassifier.fit(training_set_X, training_set_Y)
    print "end naive training: ", time.asctime()
    nbClassifications1 = naiveClassifier.predict(test_set_X)
    print "Naive Bayes (Trained on 80% Twitter + Wiki), eval'ed on all blog"
    print classification_report(test_set_Y, nbClassifications1)
    print "Accuracy: %f" % accuracy_score(test_set_Y, nbClassifications1) 

    # SVM model
    svmClassifier = LinearSVC()
    print "start svm training: ", time.asctime()
    svmClassifier.fit(training_set_X, training_set_Y)
    print "end svm  training: ", time.asctime()
    print "SVM (Trained on 80% Twitter + Wiki), eval'ed on all blog"
    svmClassifications = svmClassifier.predict(test_set_X)
    print classification_report(test_set_Y, svmClassifications)
    print "Accuracy: %f" % accuracy_score(test_set_Y, svmClassifications) 
    
if __name__ == "__main__":
    experiment()

    
