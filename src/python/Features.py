# Create any useful feature sets 
# Format should be nltk or sklearn friendly
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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


# We can introduce separate vectorizer objects 
# as hashing vectorizer is stateless
def bagOfWordsSkLearn(document, **args):
     vectorizer = HashingVectorizer(**args)  
     return vectorizer.transform(document)

def wordCountsSkLearn(document, **args):
    vectorizer = CountVectorizer(**args)
    return vectorizer.transform(document)


# Utilities    
def getX(data):
    return [obs[0] for obs in data]

def getY(data):
    return [obs[1] for obs in data]
     
        