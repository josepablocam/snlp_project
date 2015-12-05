from sklearn.cross_validation import train_test_split
from ReadData import *
from Features import *
import Globals
from Models import *
from clean_wiki import *

def experiment(train_set, test_set):
    # Use twitter + wiki data as training data, we test on blog
    # we need everything to be utf8
    print "Collecting training and test data"
    assert isinstance(train_set, str)
    if isinstance(test_set, float):
        print 'train on: ', 1-test_set , 'of',  train_set, ', test on: ', test_set, 'of ',  train_set
        if train_set == 'twitterwiki':
            twitterEmotion = to_utf8(prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False))
            wikiNoEmotion = prepareWikiData(Globals.WIKI_TRAIN, splitwords = False)
            cleanedWiki = clean_wiki(wikiNoEmotion)
            allTrainingData = twitterEmotion + cleanedWiki
        elif train_set == 'twitter':
            allTrainingData = to_utf8(prepareTwitterDataWithPNLabel(Globals.TWITTER_TRAIN, splitwords = False))
        elif train_set == 'bi_blog':
            allTrainingData = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords = False))
        elif train_set == 'multi_blog':
            allTrainingData = to_utf8(prepareBlogDataWithEmotionLabel(Globals.BLOG_DATA, splitwords = False))
        else:
            print 'Error, no such data! Twitter data with Pos and Neg label will be used default dataset...'
            allTrainingData = to_utf8(prepareTwitterDataWithPNLabel(Globals.TWITTER_TRAIN, splitwords = False))

        (trainingData, testingData) = train_test_split(allTrainingData, test_size = test_set, random_state = 10)

    else:
        print 'train on: ', train_set, ', test on: ', test_set
        if train_set == 'twitterwiki':
            twitterEmotion = to_utf8(prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False))
            wikiNoEmotion = prepareWikiData(Globals.WIKI_TRAIN, splitwords = False)
            cleanedWiki = clean_wiki(wikiNoEmotion)
            trainingData = twitterEmotion + cleanedWiki

        elif train_set == "blog":
            trainingData = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords = False))
        elif train_set == 'twitter':
            trainingData = to_utf8(prepareTwitterData(Globals.TWITTER_TEST, splitwords = False))

        else:
            trainingData = to_utf8(prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False))

        if test_set == 'twitterwiki':
            twitterEmotion = to_utf8(prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False))
            wikiNoEmotion = prepareWikiData(Globals.WIKI_TRAIN, splitwords = False)
            cleanedWiki = clean_wiki(wikiNoEmotion)
            testingData = twitterEmotion + cleanedWiki
        elif test_set == 'blog':
            testingData = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords = False))
        elif test_set == 'twitter':
            testingData = to_utf8(prepareTwitterData(Globals.TWITTER_TEST, splitwords = False))
        else:
            print "Error: no such data! Blog data will be used as default testing set..."
            testingData = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords = False))

      #  (trainingData, validationData) = train_test_split(allTrainingData, test_size = 0.2, random_state = 10)

    # Featurize
    print "Featurizing"
    # create training data with sparse matrix
    training_set_X = bagOfWordsSkLearn(getX(trainingData))
    training_set_Y = getY(trainingData)

    test_set_X = bagOfWordsSkLearn(getX(testingData))
    test_set_Y = getY(testingData)

 #   print "The report for Naive Bayes Gaussian Classifier is: ", report_GaussianNB(training_set_X, training_set_Y, test_set_X,  test_set_Y)
    print "The report for SVM with linear kernel is: \n", report_SVM_linearK(training_set_X, training_set_Y, test_set_X,  test_set_Y)
 #   print "The report for SVM with RBF kernel is: \n", report_SVM_rbfK(training_set_X, training_set_Y, test_set_X,  test_set_Y)
 #   print "The report for SVM with sigmoid kernel is: \n", report_SVM_sigK(training_set_X, training_set_Y, test_set_X,  test_set_Y)
 #   print "The report for SVM with polynomial kernel is: \n", report_SVM_polyK(training_set_X, training_set_Y, test_set_X,  test_set_Y)
 #   print "The report for Fisher Linear Classifier is: \n", report_FisherLDA(training_set_X, training_set_Y, test_set_X,  test_set_Y)

 #   print "The report for  Multivariate Gaussian Classifier is: \n", report_GMM(training_set_X, training_set_Y, test_set_X,  test_set_Y)
 #   print "The report for LogisticRegression Classifier is: \n ", report_LogisticReg(training_set_X, training_set_Y, test_set_X,  test_set_Y)



if __name__ == "__main__":

    # train and test using only twitterwiki data, binary labels
    experiment("twitterwiki", 0.2)
    # train and test using only twitter(TWITTER_TRAIN) data, binary labels
    experiment("twitter", 0.2)
    # train and test using only blog data, binary labels
    experiment("bi_blog", 0.2)
    # train and test using only blog data, original emotion labels
    experiment("multi_blog", 0.2)
    #  train on twitterwiki data, and test on blog data,  binary labels
    experiment("twitterwiki", "blog")
    #  train on twitter data (TWITTER_TEST), and test on blog data,  binary labels
    experiment("twitter", "blog")
    #  train on blog data, and test on twitter data (TWITTER_TEST),  binary labels
    experiment("blog", "twitter")
    #  train on blog data, and test on twitterwiki data,  binary labels
    experiment("blog", "twitterwiki")

