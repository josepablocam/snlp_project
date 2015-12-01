# Set of utilities to read in the data for experiments

import csv
import codecs

def readTwitterData(dataFile, splitwords = True):
    f = open(dataFile)
    csv_f = csv.reader(f)
    sentimentLabels = [] # list of str
    twitterSentences = [] # list of str
    for row in csv_f:
        sentimentLabels.append(row[0])
        twitterSentences.append(row[-1])
    # make the sentence into a list of strings
    if splitwords:
        for i in range(len(twitterSentences)):
            twitterSentences[i] = twitterSentences[i].split()
    return sentimentLabels, twitterSentences

def readBlogData(labeledDataFile, sentencesFile, splitwords = True):
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
    return emotionLabels, blogSentences
    
    
def prepareBlogData(blogLabelsPath, blogSentencePath, splitwords = True):
    """
    Get blog data, if splitwords sentence = list of strings, else string
    """
    originalBlogData = []
    blogDataForClassifier = [] # change all words to lowercase
    blogDataLabels, blogDataSentences = readBlogData(blogLabelsPath, blogSentencePath, splitwords)
    for i in range(len(blogDataLabels)):
        originalBlogData.append((blogDataSentences[i], blogDataLabels[i]))
    for (words, emotion) in originalBlogData:
        if (emotion == 'ne'):
            currentLabel = '0'
        else:
            currentLabel = '1'
        words_filtered = [e.lower() for e in words]
        if splitwords:
            blogDataForClassifier.append((words_filtered, currentLabel))
        else:
            blogDataForClassifier.append((' '.join(words_filtered), currentLabel))
    return blogDataForClassifier


def prepareTwitterData(twitterDataFile, splitwords = True):
    """
    Read twitter data, if splitwords sentence = list of strings, else string
    """
    originalData = []
    dataForClassifier = [] # change all words to lowercase
    twitterLabels, twitterSentences = readTwitterData(twitterDataFile, splitwords)
    for i in range(len(twitterLabels)):
        originalData.append((twitterSentences[i], twitterLabels[i]))
    for (words, emotion) in originalData:
        if (emotion == '2'):
            currentLabel = '0'
        else:
            currentLabel = '1'
        dataForClassifier.append((words.lower(), currentLabel))
    return dataForClassifier


def prepareWikiData(wikiDataFile, splitwords = True):
    """
    Read wiki data, if splitwords sentence = list of strings, else string
    """
    file = codecs.open(wikiDataFile, encoding = 'utf-8', mode = "r")
    labeledData = []
    for line in file:
        if splitwords:
            labeledData.append((line.lower().rstrip().split(), '0'))
        else:
            labeledData.append((line.lower().rstrip(), '0'))    
    file.close()    
    return labeledData
    

def to_utf8(data):
    """
    Decode as utf8, throws out elements that don't conform
    """
    clean_data = []
    decode = lambda x: (x[0].decode('utf-8'), x[1]) if isinstance(x, tuple) else x.decode('utf-8')
    for obs in data:
        try:
            clean_obs = decode(obs)
            clean_data.append(clean_obs)
        except UnicodeDecodeError:
            pass
    return clean_data          