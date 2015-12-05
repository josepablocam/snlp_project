# Set of utilities to read in the data for experiments

import csv
import codecs
import random

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

def readBlogData(blogDataPath, splitwords = True):
    with open(blogDataPath, "r") as f:
        dataList = [line.split() for line in f]
    f.close()
    return [ (elem[2:], elem[0]) if splitwords else (" ".join(elem[2:]), elem[0]) for elem in dataList ]

# read blog data using the original 7 emotion labels
def prepareBlogDataWithEmotionLabel(blogDataPath, splitwords = True):
    """
    Get blog data, if splitwords sentence = list of strings, else string, label is original emotion label
    """
    data = readBlogData(blogDataPath, splitwords)
    to_lower = (lambda x: [w.lower() for w in x]) if splitwords else (lambda x: x.lower())
    return [ (to_lower(txt), label) for txt, label in data ]
    
def prepareBlogData(blogDataPath, splitwords = True):
    """
    Get blog data, if splitwords sentence = list of strings, else string, labele is 1 if emotional 0 if not emotional
    """
    observations = prepareBlogDataWithEmotionLabel(blogDataPath, splitwords)
    return [ (txt, '0' if label == 'ne' else '1') for txt, label in observations ]


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



# read the twitter data using its original positive and negative labels
def prepareTwitterDataWithPNLabel(twitterDataFile, splitwords = True):
    """
    Read twitter data, if splitwords sentence = list of strings, else string
    """
    originalData = []
    dataForClassifier = [] # change all words to lowercase
    twitterLabels, twitterSentences = readTwitterData(twitterDataFile, splitwords)
    for i in range(len(twitterLabels)):
        originalData.append((twitterSentences[i], twitterLabels[i]))
    for (words, emotion) in originalData:
        dataForClassifier.append((words.lower(), emotion))
    return dataForClassifier