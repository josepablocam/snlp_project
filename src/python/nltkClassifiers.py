__author__ = 'xiuyanni'
import csv
import nltk
from nltk import FreqDist
def readTwitterData(dataFile):
    f = open(dataFile)
    csv_f = csv.reader(f)
    sentimentLabels = [] # list of str
    twitterSentences = [] # list of str
    for row in csv_f:
        sentimentLabels.append(row[0])
        twitterSentences.append(row[-1])

    for i in range(len(twitterSentences)): # make the sentence into a list of strings
        twitterSentences[i] = twitterSentences[i].split()

  #  print "type(sentimentLabels[0]), type(twitterSentences[0])", type(sentimentLabels[0]), type(twitterSentences[0]), twitterSentences[0]
    return sentimentLabels, twitterSentences

def readBlogData(labeledDataFile, sentencesFile):
    with open(labeledDataFile, "r") as f:
        dataList = [line.split() for line in f]
    f.close()
    with open(sentencesFile, "r") as f2:
        sentenceList = [line.split() for line in f2]
    f2.close()
    emotionLabels = []
    emotionIndicators = []
    blogSentences = []
    for dataRecord in dataList:
        emotionLabels.append(dataRecord[1])
        emotionIndicators.append(dataRecord[3:len(dataRecord)])
    for sentence in sentenceList:
        blogSentences.append(sentence[1:len(sentence)])

  #  print "type(emotionLabels[0]), type(blogSentences[0])", type(emotionLabels[0]), type(blogSentences[0]), blogSentences[0]
    return emotionLabels, blogSentences
def prepareBlogData(blogLabelsPath, blogSentencePath):
    originalBlogData = []
    blogDataForClassifier = [] # change all words to lowercase
    blogDataLabels, blogDataSentences = readBlogData(blogLabelsPath, blogSentencePath)
    for i in range(len(blogDataLabels)):
        originalBlogData.append((blogDataSentences[i], blogDataLabels[i]))
    for (words, emotion) in originalBlogData:
        if (emotion == 'ne'):
            currentLabel = '0'
        else:
            currentLabel = '1'
        words_filtered = [e.lower() for e in words if len(e) >= 3]
      #  blogDataForClassifier.append((' '.join(words_filtered), currentLabel))
        blogDataForClassifier.append((words_filtered, currentLabel))
    return blogDataForClassifier


def prepareTwitterData(twitterDataFile):
    originalData = []
    dataForClassifier = [] # change all words to lowercase
    twitterLabels, twitterSentences = readTwitterData(twitterDataFile)
    for i in range(len(twitterLabels)):
        originalData.append((twitterSentences[i], twitterLabels[i]))
    for (words, emotion) in originalData:
        if (emotion == '2'):
            currentLabel = '0'
        else:
            currentLabel = '1'
        words_filtered = [e.lower() for e in words if len(e) >= 3]
    #    dataForClassifier.append((' '.join(words_filtered), currentLabel))
        dataForClassifier.append((words_filtered, currentLabel))

    return dataForClassifier



def getwordsFromLabeledData(labeledData):
    all_words = []
    for (words, sentiment) in labeledData:
      all_words.extend(words)
    return all_words

def getWordFeatures(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def word_feats(words):
    return dict([(word, True) for word in words])

labeledBlogData = prepareBlogData("../Emotion-Data/AnnotatedData/AnnotSet1.txt", "../Emotion-Data/AnnotatedData/basefile.txt")
twitterTestingData = prepareTwitterData("../trainingandtestdata/testdata.manual.2009.06.14.csv")

word_features = getWordFeatures(getwordsFromLabeledData(labeledBlogData))

blogDataWithEmotion = [b for b in labeledBlogData if b[1] == '1']
blogDataWithoutEmotion = [b for b in labeledBlogData if b[1] == '0']
emotionCutoff = len(blogDataWithEmotion) * 3/4
nonEmotionCutoff = len(blogDataWithoutEmotion) * 3/4

blogtrainingSet = blogDataWithEmotion[:emotionCutoff] + blogDataWithoutEmotion[:nonEmotionCutoff]
blogtestingSet = blogDataWithEmotion[emotionCutoff:] + blogDataWithoutEmotion[nonEmotionCutoff:]

# return a dict, key is the word, value is True or False
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def calcualteAccuracy(clr, testRecords):
    correctCount = 0.0
    for testRecord in testRecords:
        predictLabel = clr.classify(extract_features(testRecord[0]))
        if predictLabel == testRecord[1]:
            correctCount += 1
    return correctCount / len(testRecords)

# training_set is a list of tuple,
# the first element of the tuple is a dict,
# the second element of the tuple is the label
training_set = nltk.classify.apply_features(extract_features, blogtrainingSet)
import time
print "start naive training: ", time.asctime()
naiveClassifier = nltk.NaiveBayesClassifier.train(training_set)
print "end naive training: ", time.asctime()
naiveAccuracy1 = calcualteAccuracy(naiveClassifier, blogtestingSet)
naiveAccuracy2 = calcualteAccuracy(naiveClassifier, twitterTestingData)
print "Naive classifier the accuracy for 3/4 train on blog, and  1/4 test on blog is: " , naiveAccuracy1
print "Naive classifier the accuracy of training on blog, and test on twitter test is:", naiveAccuracy2
import nltk.classify
from sklearn.svm import LinearSVC

svmClassifier = nltk.classify.SklearnClassifier(LinearSVC())
print "start svm training: ", time.asctime()
svmClassifier.train(training_set)
print "end svm  training: ", time.asctime()

svmAccuracy1 = calcualteAccuracy(svmClassifier, blogtestingSet)
svmAccuracy2 = calcualteAccuracy(svmClassifier, twitterTestingData)

print "SVM classifier the accuracy for 3/4 train on blog, and  1/4 test on blog is: " , svmAccuracy1
print "SVM classifier the accuracy of training on blog, and test on twitter test is:", svmAccuracy2
