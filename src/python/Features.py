# Create any useful feature sets 
# Format should be nltk or sklearn friendly
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import numpy
import nltk
from nltk.corpus import sentiwordnet as swn
import re
from collections import Counter

# separates pos in a string
POS_DELIM = "_"

# Utilities
def getX(data):
    return [obs[0] for obs in data]

def getY(data):
    return [obs[1] for obs in data]

def make_experiment_matrices(train_data, test_data, featurizer, getX = getX, getY = getY):
    """
    Returns dictionary of experiment matrices and label vectors
    :param train_data: training data (tuples of observation and label)
    :param test_data: test data (tuples of observation and label)
    :param featurizer: function that takes train_data and test_data and returns 2 matrices (tuple) of training
    and test X matrices (we do this in one step since some featurizers are stateful, e.g. counting words)
    :param getX: function to extract X (default: Features.getX)
    :param getY: function to extract Y (default: Features.getY)
    :return: dictionary {'train_X': training matrix X, 'train_Y': vector of X labels, etc}
    """
    train_X, test_X = featurizer(getX(train_data), getX(test_data))
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
    print "Counting words"
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
    print "Calculating tf-idf"
    # train and fit, returns both vectorizer and result
    if vectorizer == None:
        vectorizer = TfidfVectorizer(**args)
        return (vectorizer, vectorizer.fit_transform(documents))
    else:
        # apply existing one (which holds vocab etc)
        return (vectorizer, vectorizer.transform(documents))

# n-gram features
def nGramFeatures(documents, gramNumber = 2, vectorizer=None, **args):
    # every sentence is a dictionary
    tokenized_sents = [nltk.word_tokenize(document) for document in documents]
   # print tokenized_sents
    biFeaturedSentences = [nltk.ngrams(sentence, gramNumber) for sentence in tokenized_sents]
   # print biFeaturedSentences
    DictFeats = [dict(Counter(sentence)) for sentence in biFeaturedSentences]
  #  print DictFeats
    dicttofeats = DictVectorizer(**args)
    if vectorizer == None:
        # second set of features, arbitrary, encode as dictionary
        return (dicttofeats, dicttofeats.fit_transform(DictFeats))
    else:
       # print DictFeats, "error happens"
        return (vectorizer, vectorizer.transform(DictFeats))

def avgSynsetScores(word, cache = None):
    """
    Calculate triple of (negative, objective, positive) for a given word. Each value is the
    average of those scores for the words in the associated synset. Averaging is equally weighted
    across elements
    :param word:
    :param cache: optional dictionary to cache lookups (suggested for large sets)
    :return: triple of scores
    """
    # averages all words return with the lookup...this can certainly yield
    # false positives
    if cache != None and word in cache:
        return cache[word]

    pos, neg, obj, ct = 0.0, 0.0, 0.0, 0
    for meaning in swn.senti_synsets(word):
        pos += meaning.pos_score()
        neg += meaning.neg_score()
        obj += meaning.obj_score()
        ct += 1
    if ct > 0:
        valence = (neg / ct, obj / ct, pos / ct)
        if numpy.isnan(valence).any():
            raise ValueError("NaN Valence for %s" % word)
    else:
        valence = (0.0, 0.0, 0.0)

    if cache != None:
        cache[word] = valence
    return valence


def valenceByFrequency(documents, vectorizer = None, cache_valence = None, **args):
    """
    Returns a matrix of ct(documents) x 3, column 0 is the negative score, 1 is objective score, 2 is positive score.
    A score is calculated as the element-wise sum of the avgSynsetScore for each word in a comment
    and divided by the total count of words in that comment.
    :param documents:
    :param vectorizer:
    :param cache_valence: optionally cache word valence information for faster use later
    :param args:
    :return:
    """
    print "Calculating average valence features"
    vectorizer, cts = wordCountsSkLearn(documents, vectorizer, **args)
    # now for each word in the vocabulary, retrieve a triple
    # of the average negative, objective, positive scores
    # sort it so that the rows match the column order
    valence_data = sorted([(col, avgSynsetScores(word, cache_valence)) for word, col in vectorizer.vocabulary_.iteritems()])
    valence_data = [valence for _, valence in valence_data]
    valence_matrix = scipy.sparse.csr_matrix(valence_data)
    # matrix multiplication will give us the sum of negative, objective, positive
    # values for each observation
    sum_valences = cts * valence_matrix
    # count of words in each observation
    total_cts = cts.sum(axis = 1)
    # replace zero counts with 1 to avoid NaN. We can have zero count
    # if a tweet is determined to be all stopwords
    total_cts[total_cts == 0.0] = 1.0
    # note that this normalizes by total count, not just counts with valence information
    # just dividing for those with valence information can result in division by zero if no
    # words had valence info, whereas we likely want that to be (0, 0, 0)
    return vectorizer, sum_valences / total_cts


def pos_in_txt(txt, pos):
    """
    Return true if text has POS taggin matching what was passed in
    e.g. pos_in_txt(..., "jj") true for adjective or pos_in_text(..., "vb.*") for all verbs
    pos tag sources
    http://cs.nyu.edu/grishman/jet/guide/PennPOS.html
    :param txt:
    :param pos:
    :return:
    """
    pos_regex = ".*_%s( |$)" % pos
    return re.match(pos_regex, txt) != None


def relevantPOSVocabulary(vocabulary, key_pos):
    """
    Given a vocabulary coming from a vectorizer, returns values that meet a list of POS tags
    :param vocabulary: part of speech tagged tokens (from a vectorizer)
    :param key_pos: parts of speech to look for (or regex that identify that pos)
    :return: set of relevant vocabulary
    """
    # return a new vocabulary
    relevant_ngrams = []
    for pos in key_pos:
        relevant_ngrams += [ ngram for ngram in vocabulary if pos_in_txt(ngram, pos) ]
    return set(relevant_ngrams)


def keyPOSNGrams(tagged_documents, key_pos, vectorizer = None, tf_idf = False, **args):
    """
    Vectorizer and  matrix with ngram counts for ngrams that include a relevant part of speech
    Vectorizer's vocabulary mapping has been modified to solely contain relevant
    ngrams as per POS
    :param tagged_documents:
    :param key_pos:
    :param vectorizer:
    :param args:
    :return:

    e.g.
    tagged = [line.rstrip() for line in open(Globals.BLOG_POS, "r")]
    vectorizer, cts = Features.keyPOSNGrams(tagged, ["jj.*", "vb.*"], ngram_range = (1, 2))
    """
    # count directly as is (so that columns will have word_pos counts)
    if tf_idf and vectorizer and type(vectorizer) != TfidfVectorizer:
        raise ValueError("tf_idf = True but vectorizer passed in is not TfidfVectorizer")

    if vectorizer == None:
        # this is a first pass, collect the relevant vocabulary
        print "Collecting relevant vocabulary (first pass)"
        first_pass_vectorizer, _ = wordCountsSkLearn(tagged_documents, **args)
        relevant_vocab = relevantPOSVocabulary(first_pass_vectorizer.vocabulary_, key_pos)
        args['vocabulary'] = relevant_vocab

    # now let's only consider columns that have relevant POS
    if tf_idf:
        vectorizer, cts = tfIdfSkLearn(tagged_documents, vectorizer, **args)
    else:
        vectorizer, cts = wordCountsSkLearn(tagged_documents, vectorizer, **args)
    return vectorizer, cts


def punctuation(documents, puncts = None, vectorizer = None, **kwargs):
    # default punctuation to look for
    print "Calculating punctuation features"
    puncts = puncts if puncts else ['?', '!', '...']
    results = []
    for document in documents:
        doc_results = [ ("has_%s" % punct, punct in document) for punct in puncts ]
        results.append(dict(doc_results))
    if vectorizer == None:
        vectorizer = DictVectorizer()
        return vectorizer, vectorizer.fit_transform(results)
    else:
        return vectorizer, vectorizer.transform(results)


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


    
        