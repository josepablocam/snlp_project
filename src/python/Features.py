# Create any useful feature sets 
# Format should be nltk or sklearn friendly
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import numpy
import nltk
import re


# Utilities
def getX(data):
    return [obs[0] for obs in data]

def getY(data):
    return [obs[1] for obs in data]

def make_experiment_matrices(train_data, test_data, featurizer):
    """
    Returns dictionary of experiment matrices and label vectors
    :param train_data: training data (tuples of observation and label)
    :param test_data: test data (tuples of observation and label)
    :param featurizer: function that takes train_data and test_data and returns 2 matrices (tuple) of training
    and test X matrices (we do this in one step since some featurizers are stateful, e.g. counting words)
    :return: dictionary {'train_X': training matrix X, 'train_Y': vector of X labels, etc}
    """
    train_X, test_X = featurizer(getX(train_data), getY(test_data))
    train_Y = getY(train_data)
    test_Y = getY(test_data)
    return {'train_X' : train_X, 'train_Y' : train_Y, 'test_X' : test_X, 'test_Y' : test_Y }

def append_features(matrices):
    """ given a list of matrices append column-wise """
    if len(matrices) == 1:
        return matrices[0]
    merged = scipy.sparse.hstack((matrices[0], matrices[1]))
    return append_features([ merged ] + matrices[2:])


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
# def bagofwordsFeatureExtractor(document):
#     # TODO: if we want to use this need to define bagofwords_features
#     document_words = set(document)
#     features = {}
#     for word in bagofwords_features:
#         features['contains(%s)' % word] = (word in document_words)
#     return features

# def ngramFeatureExtractor(document):
#     assert isinstance(document, list)
#
#     #TODO: implement n gram feature extractor
#     features = Counter()
#     for i in range(len(document) - 1):
#         features[(document[i], document[i+1])] += 1
#     return features
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


def _ends_with(word, ending):
    pattern = ".*%s$" % ending
    return re.match(pattern, word) != None

def count_word_endings(documents, list_of_endings, vectorizer = None):
    """
    Matrix with count of words with particularly endings
    :param documents: list of strings representing documents or a matrix version of the documents
        if matrix, we don't recount stuff, assumes vectorizer was the one used to create matrix of counts
    :param list_of_endings: list of relevant endings
    :param vectorizer: CountVectorizer or none
    :return: vectorizer, matrix (stateful)
    """
    # To make this fast what we do is filter the columns of the resulting
    # word counts, take advantage of sklearn goodness
    if vectorizer == None:
        vectorizer, cts = wordCountsSkLearn(documents)
    elif isinstance(documents, (scipy.sparse.spmatrix, numpy.matrix)):
        cts = scipy.sparse.csr_matrix(documents)
    else:
        _, cts = wordCountsSkLearn(documents, vectorizer)
    # find relevant columns for each ending
    relevant_cols = dict([(ending, []) for ending in list_of_endings])
    for ending in list_of_endings:
        relevant_cols[ending] += [ col for word, col in vectorizer.vocabulary_.iteritems() if _ends_with(word, ending) ]
    # now we have words that have appropriate endings, retrieve parts of the matrix that are relevant for each
    # and sum row-wise (multiple columns can satisfy one ending)
    matrices = []
    for ending in list_of_endings:
        submatrix = cts[:, relevant_cols[ending]]
        # sum is annoying, doesn't return same datatype, so make it sparse again
        ending_matrix = scipy.sparse.csr_matrix(submatrix.sum(axis = 1))
        matrices.append(ending_matrix)
    # now just concatenate these column-wise
    relevant_cts = append_features(matrices)
    return vectorizer, relevant_cts

def tfIdfSkLearn(documents, vectorizer = None, **args):
    """
    TF-IDF matrix. Stateful: returns tuple of
    (vectorizer, matrix). Vectorizer holds mapping of words to indices
    so must be used consistently with new data
    :param documents:
    :param vectorizer:
    :param args:
    :return:
    """
    # train and fit, returns both vectorizer and result
    if vectorizer == None:
        vectorizer = TfidfVectorizer(**args)
        return (vectorizer, vectorizer.fit_transform(documents))
    else:
        # apply existing one (which holds vocab etc)
        return (vectorizer, vectorizer.transform(documents))

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


    
        