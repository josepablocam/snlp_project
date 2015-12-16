# Experiments involving maximum entropy classifiers
import Globals
from ReadData import *
import Features
import clean_wiki
import Models
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
import numpy


def get_elems_at(src, indices):
    return to_utf8([src[i] for i in indices])

def get_pos_data(path):
    return [line.lower().rstrip() for line in open(path, "r")]

# Preparing data
# Raw data (i.e. withou unicode clean up etc)
twitter_train_raw = prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False)
twitter_test_raw = prepareTwitterData(Globals.TWITTER_TEST, splitwords = False)
wiki_raw = prepareWikiData(Globals.WIKI_TRAIN, splitwords= False)
blog_raw = prepareBlogData(Globals.BLOG_DATA, splitwords=False)

# Raw POS data
twitter_train_pos_raw = get_pos_data(Globals.TWITTER_TRAIN_POS)
twitter_test_pos_raw = get_pos_data(Globals.TWITTER_TEST_POS)
wiki_pos_raw = get_pos_data(Globals.WIKI_POS)
blog_pos_raw = get_pos_data(Globals.BLOG_POS)

# see which indices are clean for each
twitter_train_indices = to_utf8(twitter_train_raw, return_indices=True)
twitter_test_indices = to_utf8(twitter_test_raw, return_indices=True)
wiki_indices = clean_wiki.clean_wiki(wiki_raw, return_indices = True)
blog_indices = to_utf8(blog_raw, return_indices= True)


# Clean up utf8 issues and nonsense observations (wiki)
twitter_train = get_elems_at(twitter_train_raw, twitter_train_indices)
twitter_test = get_elems_at(twitter_test_raw, twitter_test_indices)
wiki = get_elems_at(wiki_raw, wiki_indices)
blog = get_elems_at(blog_raw, blog_indices)

# POS data
twitter_train_pos = get_elems_at(twitter_train_pos_raw, twitter_train_indices)
twitter_test_pos = get_elems_at(twitter_test_pos_raw, twitter_test_indices)
wiki_pos = get_elems_at(wiki_pos_raw, wiki_indices)
blog_pos = get_elems_at(blog_pos_raw, blog_indices)
tw_pos = twitter_train_pos + wiki_pos

# Create various data splits
tw = twitter_train + wiki
tw_80, tw_20 = train_test_split(twitter_train + wiki, test_size = 0.2, random_state = 10)
blog_80, blog_20 = train_test_split(blog, test_size = 0.2, random_state = 20)
blog_80_pos, blog_20_pos = train_test_split(blog_pos, test_size=0.2, random_state=20)


# Experiment wrappers
def experiment(model_report, train, test, featurizer):
    data = Features.make_experiment_matrices(train, test, featurizer)
    return model_report(data['train_X'], data['train_Y'], data['test_X'], data['test_Y'])

#### MaxEnt experiments #######
def experiment_svm_polyK(train, test, featurizer):
    return experiment(Models.report_SVM_polyK, train, test, featurizer)


print "SVM with Poly kernel"

# Feature-Set 1 - averaged sentiment valence
cache_valence = dict()
def feat1(train, test):
    vectorizer, train_matrix = Features.valenceByFrequency(train, vectorizer = None, cache_valence = cache_valence, stop_words = 'english')
    _, test_matrix = Features.valenceByFrequency(test, vectorizer = vectorizer, cache_valence = cache_valence, stop_words = 'english')
    return train_matrix, test_matrix

print "=>Experiment 1: valence blog(80%) -> blog(20%)"
experiment1_b = experiment_svm_polyK(blog_80, blog_20, feat1)
print "=>Experiment 1: valence twitter+wiki -> blog"
experiment1_twb = experiment_svm_polyK(tw, blog, feat1)
print "=>Experiment 1: valence twitter+wiki -> twitter(test)"
experiment1_tw = experiment_svm_polyK(tw, twitter_test, feat1)


# Feature set 2 - tf-idf
def feat2(train, test):
    state_info, train_matrix = Features.tfIdfSkLearn(train)
    _, test_matrix = Features.wordCountsSkLearn(test, vectorizer = state_info, stop_words = 'english')
    return train_matrix, test_matrix

print "=>Experiment 2: tf-idf blog(80%) -> blog(20%)"
experiment2_b = experiment_svm_polyK(blog_80, blog_20, feat2)
print "=>Experiment 2: tf-idf twitter+wiki -> blog"
experiment2_twb = experiment_svm_polyK(tw, blog, feat2)
print "=>Experiment 2: tf-idf twitter+wiki -> twitter(test)"
experiment2_tw = experiment_svm_polyK(tw, twitter_test, feat2)

# avg sentiment and tf-idf
def feat3(train, test):
    # valence info
    train_valence, test_valence = feat1(train, test)
    # tf idf info
    train_cts, test_cts = feat2(train, test)
    # combined info
    train_matrix = Features.append_features([train_valence, train_cts])
    test_matrix = Features.append_features([test_valence, test_cts])
    return train_matrix, test_matrix

print "=>Experiment 3: valence + tf-idf blog(80%) -> blog(20%)"
experiment3_b = experiment_svm_polyK(blog_80, blog_20, feat3)
print "=>Experiment 3: valence + tf-idf twitter+wiki -> blog"
experiment3_twb = experiment_svm_polyK(tw, blog, feat3)
print "=>Experiment 3: valence + tf-idf twitter+wiki -> twitter(test)"
experiment3_tw = experiment_svm_polyK(tw, twitter_test, feat3)

# feature set 3 and punctuation
def feat4(train, test):
    # feature set 3
    train_f3, test_f3 = feat3(train, test)
    # punctuation
    puncter, train_punct = Features.punctuation(train)
    _, test_punct = Features.punctuation(test, vectorizer = puncter)
    train_matrix = Features.append_features([train_f3, train_punct])
    test_matrix = Features.append_features([test_f3, test_punct])
    return train_matrix, test_matrix

print "=>Experiment 4: valence + tf-idf + punctuation blog(80%) -> blog(20%)"
experiment4_b = experiment_svm_polyK(blog_80, blog_20, feat4)
print "=>Experiment 4: valence + tf-idf + punctuation twitter+wiki -> blog"
experiment4_twb = experiment_svm_polyK(tw, blog, feat4)
print "=>Experiment 4: valence + tf-idf + punctuation twitter+wiki -> twitter(test)"
experiment4_tw = experiment_svm_polyK(tw, twitter_test, feat4)

# Just word valence and punctuation
def feat5(train, test):
    train_valence, test_valence = feat1(train, test)
    puncter, train_punct = Features.punctuation(train)
    _, test_punct = Features.punctuation(test, vectorizer = puncter)
    train_matrix = Features.append_features([train_valence, train_punct])
    test_matrix = Features.append_features([test_valence, test_punct])
    return train_matrix, test_matrix

print "=>Experiment 5: valence + punctuation blog(80%) -> blog(20%)"
experiment5_b = experiment_svm_polyK(blog_80, blog_20, feat5)
print "=>Experiment 5: valence + punctuation twitter+wiki -> blog"
experiment5_twb = experiment_svm_polyK(tw, blog, feat5)
print "=>Experiment 5: valence + punctuation twitter+wiki -> twitter(test)"
experiment5_tw = experiment_svm_polyK(tw, twitter_test, feat5)


## valence, punctuation and relevant POS counts
# Being sloppy for now and have a function per experiment set... we can clean this up later
def feat6_generic(train, test, train_pos, test_pos):
    train_f5, test_f5 = feat5(train, test)
    cter, train_cts = Features.keyPOSNGrams(train_pos, ["jj.*", "vb.*"], tf_idf = True)
    _, test_cts = Features.keyPOSNGrams(test_pos, ["jj.*", "vb.*"], vectorizer = cter, tf_idf= True)
    train_matrix = Features.append_features([train_f5, train_cts])
    test_matrix = Features.append_features([test_f5, test_cts])
    return train_matrix, test_matrix

# a function per experiment set...ugly sorry :(
def feat6_b():
    return lambda train, test: feat6_generic(train, test, blog_80_pos, blog_20_pos)
def feat6_tw_b():
    return lambda train, test: feat6_generic(train, test, tw_pos, blog_pos)
def feat6_tw():
    return lambda train, test: feat6_generic(train, test, tw_pos, twitter_test_pos)


print "Experiment 6: valence + punctuation + key POS word counts blog(80%) -> blog(20%)"
experiment6_b = experiment_svm_polyK(blog_80, blog_20, feat6_b())
print "Experiment 6: valence + punctuation + key POS word counts twitter+wiki -> blog"
experiment6_twb = experiment_svm_polyK(tw, blog, feat6_tw_b())
print "Experiment 6: valence + punctuation + key POS word counts twitter+wiki -> twitter(test)"
experiment6_tw = experiment_svm_polyK(tw, twitter_test, feat6_tw())


# Cross validation for blog -> blog experiment with best accuracy (to compare to original paper)
folds = KFold(n = len(blog), n_folds= 10, random_state = 1)
test_accuracies = []
for train_indices, test_indices in folds:
    train_data = get_elems_at(blog, train_indices)
    test_data = get_elems_at(blog, test_indices)
    data = Features.make_experiment_matrices(train_data, test_data, feat4)
    model = LogisticRegression()
    model.fit(data['train_X'], data['train_Y'])
    predictions = model.predict(data['test_X'])
    accuracy = accuracy_score(data['test_Y'], predictions)
    test_accuracies.append(accuracy)


print "10-CV accuracy blog on blog:%.2f[+/-%.2f]" % (numpy.mean(test_accuracies), numpy.std(test_accuracies))


