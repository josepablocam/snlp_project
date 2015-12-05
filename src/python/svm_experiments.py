__author__ = 'xiuyanni'
# Experiments involving svm classifiers
import Globals
from ReadData import *
import Features
import clean_wiki
import Models
import re
# sk utilities
from sklearn.cross_validation import train_test_split


# Preparing data
# Possible training data
twitter_train = to_utf8(prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False))
twitter_test = to_utf8(prepareTwitterData(Globals.TWITTER_TEST, splitwords = False))
twitter_PN = to_utf8(prepareTwitterDataWithPNLabel(Globals.TWITTER_TRAIN, splitwords = False))
wiki = prepareWikiData(Globals.WIKI_TRAIN, splitwords= False)
# clean wiki data
wiki = clean_wiki.clean_wiki(wiki)
# blog data
blog = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords=False))
blog_emotion = to_utf8(prepareBlogDataWithEmotionLabel(Globals.BLOG_DATA, splitwords=False))

#### Linear SVM experiments #######
def experiment_linear_svm(train, test, featurizer):
    data = Features.make_experiment_matrices(train, test, featurizer)
    return Models.report_SVM_linearK(data['train_X'], data['train_Y'], data['test_X'], data['test_Y'])

#### RBF SVM experiments #######
def experiment_rbf_svm(train, test, featurizer):
    data = Features.make_experiment_matrices(train, test, featurizer)
    return Models.report_SVM_rbfK(data['train_X'], data['train_Y'], data['test_X'], data['test_Y'])

#### Sigmoid SVM experiments #######
def experiment_sigmoid_svm(train, test, featurizer):
    data = Features.make_experiment_matrices(train, test, featurizer)
    return Models.report_SVM_sigK(data['train_X'], data['train_Y'], data['test_X'], data['test_Y'])

#### Polynomial SVM experiments #######
def experiment_poly_svm(train, test, featurizer):
    data = Features.make_experiment_matrices(train, test, featurizer)
    return Models.report_SVM_polyK(data['train_X'], data['train_Y'], data['test_X'], data['test_Y'])

##################### BernoulliNB + Binary word hasher ############################
def feat1(train, test):
    # Bag of words
    train_matrix = Features.bagOfWordsSkLearn(train)
    test_matrix = Features.bagOfWordsSkLearn(test)
    return train_matrix, test_matrix

# Create various data splits
tw = twitter_train + wiki
tw_80, tw_20 = train_test_split(twitter_train + wiki, test_size = 0.2, random_state = 10)
t_80, t_20 = train_test_split(twitter_PN, test_size = 0.2, random_state = 10)
blog_80, blog_20 = train_test_split(blog, test_size = 0.2, random_state = 20)
blog_80_emotion, blog_20_emotion = train_test_split(blog_emotion, test_size = 0.2, random_state = 20)

print "==============Experiment 1: Linear SVM with bag of word features ============"
# Twitter + Wiki (tw) -> Twitter + Wiki (tw)
# Training on 80% tw
# Testing on 20% tw
print "TW(80) -> TW(20)"
experiment1_tw = experiment_linear_svm(tw_80, tw_20, feat1)
# Training on 80% tw
# Testing on 20% tw
print "T(80) -> T(20)"
experiment1_t = experiment_linear_svm(t_80, t_20, feat1)
# Training on 100% tw, testing on 100% blog
print "TW(100) -> B(100)"
experiment1_twb = experiment_linear_svm(tw, blog, feat1)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment1_b = experiment_linear_svm(blog_80, blog_20, feat1)
# Training on 80% blog_emotion, testing on 20% blog_emotion
print "B_emotion(80) -> B_emotion(20)"
experiment1_be = experiment_linear_svm(blog_80_emotion, blog_20_emotion, feat1)

##################### Linear SVM  + word endings ############################
# Way too slow for large data, so can only use this with blog data
# http://www.ucl.ac.uk/internet-grammar/adjectiv/endings.htm
adjective_endings = [ "able", "ible", "al", "ful", "ic", "ive", "less", "ous" ]
def feat2(train, test):
    ct_adj, train_matrix = Features.count_word_endings(train, adjective_endings)
    _, test_matrix = Features.count_word_endings(test, adjective_endings, ct_adj)
    return train_matrix, test_matrix

print "==============Experiment 2: Linear SVM with word ending features============"
print "TW(100) -> B(100)"
experiment2_twb = experiment_linear_svm(tw, blog, feat2)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment2_b = experiment_linear_svm(blog_80, blog_20, feat2)
print "B_emotion(80) -> B_emotion(20)"
experiment2_be = experiment_linear_svm(blog_80_emotion, blog_20_emotion, feat2)


##################### Linear SVM  + word counts ############################
def feat3(train, test):
    state_info, train_matrix = Features.tfIdfSkLearn(train)
    _, test_matrix = Features.wordCountsSkLearn(test, vectorizer = state_info)
    return train_matrix, test_matrix

print "==============Experiment 3: Linear SVM with word count features ============"
print "TW(100) -> B(100)"
experiment3_twb = experiment_linear_svm(tw, blog, feat3)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment3_b = experiment_linear_svm(blog_80, blog_20, feat3)
print "B_emotion(80) -> B_emotion(20)"
experiment3_be = experiment_linear_svm(blog_80_emotion, blog_20_emotion, feat3)


##################### Linear SVM  + tf-idf ############################
def feat4(train, test):
    state_info, train_matrix = Features.tfIdfSkLearn(train)
    _, test_matrix = Features.tfIdfSkLearn(test, vectorizer = state_info)
    return train_matrix, test_matrix

print "==============Experiment 4: Linear SVM with tf-idf features ============"
print "TW(100) -> B(100)"
experiment4_twb = experiment_linear_svm(tw, blog, feat4)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment4_b = experiment_linear_svm(blog_80, blog_20, feat4)
print "B_emotion(80) -> B_emotion(20)"
experiment4_be = experiment_linear_svm(blog_80_emotion, blog_20_emotion, feat4)


##################### Linear SVM + n-gram features ############################
def feat5(train, test):
    state_info, train_matrix = Features.nGramFeatures(train)
    _, test_matrix = Features.nGramFeatures(test, vectorizer = state_info)
    return train_matrix, test_matrix

print "==============Experiment 5: Linear SVM with n-gram features ============"
print "TW(100) -> B(100)"
experiment5_twb = experiment_linear_svm(tw, blog, feat5)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment5_b = experiment_linear_svm(blog_80, blog_20, feat5)
print "B_emotion(80) -> B_emotion(20)"
experiment5_be = experiment_linear_svm(blog_80_emotion, blog_20_emotion, feat5)
# Training on 80% tw
# Testing on 20% tw
print "TW(80) -> TW(20)"
experiment5_tw = experiment_linear_svm(tw_80, tw_20, feat5)
# Training on 80% tw
# Testing on 20% tw
print "T(80) -> T(20)"
experiment5_t = experiment_linear_svm(t_80, t_20, feat5)