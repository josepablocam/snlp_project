# Jose Cambronero
# Experiments involving naive bayes classifiers
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
wiki = prepareWikiData(Globals.WIKI_TRAIN, splitwords= False)
# clean wiki data
wiki = clean_wiki.clean_wiki(wiki)
# blog data
blog = to_utf8(prepareBlogData(Globals.BLOG_DATA['annotations'], Globals.BLOG_DATA['sentences'], splitwords=False))

# Create various data splits
tw = twitter_train + wiki
tw_80, tw_20 = train_test_split(twitter_train + wiki, test_size = 0.2, random_state = 10)
blog_80, blog_20 = train_test_split(blog, test_size = 0.2, random_state = 20)


# Experiment wrappers
def experiment(model_report, train, test, featurizer):
    data = Features.make_experiment_matrices(train, test, featurizer)
    return model_report(data['train_X'], data['train_Y'], data['test_X'], data['test_Y'])

#### BernoulliNB experiments #######
def experiment_bernoulli_nb(train, test, featurizer):
    return experiment(Models.report_BernoulliNB, train, test, featurizer)

#### GaussianNB experiments #######
def experiment_gaussian_nb(train, test, featurizer):
    return experiment(Models.report_GaussianNB, train, test, featurizer)


##################### BernoulliNB + Binary word hasher ############################
def feat1(train, test):
    # Bag of words
    train_matrix = Features.bagOfWordsSkLearn(train)
    test_matrix = Features.bagOfWordsSkLearn(test)
    return train_matrix, test_matrix


print "==============Experiment 1: BernoulliNB with bag of word features ============"
# Twitter + Wiki (tw) -> Twitter + Wiki (tw)
# Training on 80% tw
# Testing on 20% tw
print "TW(80) -> TW(20)"
experiment1_tw = experiment_bernoulli_nb(tw_80, tw_20, feat1)
# Training on 100% tw, testing on 100% blog
print "TW(100) -> B(100)"
experiment1_twb = experiment_bernoulli_nb(tw, blog, feat1)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment1_b = experiment_bernoulli_nb(blog_80, blog_20, feat1)


##################### GaussianNB + word endings ############################
# Way too slow for large data, so can only use this with blog data
# http://www.ucl.ac.uk/internet-grammar/adjectiv/endings.htm
adjective_endings = [ "able", "ible", "al", "ful", "ic", "ive", "less", "ous" ]
def feat2(train, test):
    ct_adj, train_matrix = Features.count_word_endings(train, adjective_endings)
    _, test_matrix = Features.count_word_endings(test, adjective_endings, ct_adj)
    return train_matrix, test_matrix

print "==============Experiment 2: GaussianNB with word ending features============"
print "TW(100) -> B(100)"
experiment2_twb = experiment_gaussian_nb(tw, blog, feat2)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment2_b = experiment_gaussian_nb(blog_80, blog_20, feat2)


##################### GaussianNB + Tf IDF counts ############################
def feat3(train, test):
    state_info, train_matrix = Features.tfIdfSkLearn(train)
    _, test_matrix = Features.wordCountsSkLearn(test, vectorizer = state_info)
    return train_matrix, test_matrix

print "==============Experiment 3: GaussianNB with word count features ============"
print "TW(100) -> B(100)"
# TODO: fix hanging! Gaussian NB wants dense, but dense word counts for tw are crazy large
# experiment3_twb = experiment_gaussian_nb(tw, blog, feat3)
# Training on 80% blog, testing on 20% blog
print "B(80) -> B(20)"
experiment3_b = experiment_gaussian_nb(blog_80, blog_20, feat3)
