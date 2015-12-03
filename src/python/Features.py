# Create any useful feature sets 
# Format should be nltk or sklearn friendly
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

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
def bagOfWordsSkLearn(documents, **args):
    """
    1/0 matrix for presence of a given word. Not stateful, as word mapping
    is done with a hashing function, only returns matrix.
    """
     vectorizer = HashingVectorizer(**args)  
     return vectorizer.transform(documents)

def wordCountsSkLearn(documents, vectorizer = None, **args):
    """
    Count word instances. Stateful: returns tuple of
    (vectorizer, matrix). Vectorizer holds mapping of words to indices
    so must be used consistently with new data
    """
    # train and fit, returns both vectorizer and result
    if vectorizer == None:
        vectorizer = CountVectorizer(**args)
        return (vectorizer, vectorizer.fit_transform(documents))
    else:
        # apply existing one (which holds vocab etc)
        return (vectorizer, vectorizer.transform(documents))    

def append_features(matrices):
    """ given a list of matrices append column-wise """
    if len(matrices) == 1:
        return matrices[0]
    merged = hstack((matrices[0], matrices[1]))    
    return append_features([ merged ] + matrices[2:])


# Utilities    
def getX(data):
    return [obs[0] for obs in data]

def getY(data):
    return [obs[1] for obs in data]
     

if __name__ == "__main__":
    print "Trivial example of feature creation and use"
    import Models
    # sample sentences
    trainsents = [ ("this is a test", '0'), ("hope this works", '1'), ("ughhh",'0') ]
    testsents = [("this is working", '0'), ("i hope", '1')]
    # first set of features, word counts
    cter, trainCts = wordCountsSkLearn(getX(trainsents))
    # you should use the same vectorizer here, as the vocabulary mapping is needed
    cter, testCts = wordCountsSkLearn(getX(testsents), cter)

    # second set of features, arbitrary, encode as dictionary
    dict_feats = lambda x: {'starts_t':x[0] == 't', 'ends_s': x[-1] == 's'}
    trainDictFeats = [ dict_feats(elem) for elem in getX(trainsents) ]
    testDictFeats = [ dict_feats(elem) for elem in getX(testsents) ]
    # convert to scipy format easily using
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
    dicttofeats = DictVectorizer()
    trainDictFeatsMatrix = dicttofeats.fit_transform(trainDictFeats)
    # just transform (we already incorprated feature mapping by fit_transform)
    testDictFeatsMatrix = dicttofeats.transform(testDictFeats)

    # append both our features into matrices
    trainX = append_features([trainCts, trainDictFeatsMatrix])
    testX = append_features([testCts, testDictFeatsMatrix])

    # now we can finally feed into some model
    Models.report_GaussianNB(trainX, getY(trainsents), testX, getY(testsents))


    
        