# Experiments involving maximum entropy classifiers
import Globals
from ReadData import *
import Features
import clean_wiki
import Models
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
import sys

# writing to results file
out_path = sys.argv[1]
out_file = open(out_path, "w")

def write_cv(f, details, acc_and_err):
    f.write(details + "\n")
    f.write("accuracy:%f\nstd-error:%f\n\n" % acc_and_err)

def write_detailed(f, details, report):
    f.write(details + "\n")
    f.write(report + "\n\n")

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
def experiment_maxent(train, test, featurizer):
    return experiment(Models.report_MaxEnt, train, test, featurizer)

# create maxent model
maxent_model = lambda : Models.LogisticRegression()

# Feature-Set 1 - averaged sentiment valence
cache_valence = dict()
def feat1(train, test):
    vectorizer, train_matrix = Features.valenceByFrequency(train, vectorizer = None, cache_valence = cache_valence, stop_words = 'english')
    _, test_matrix = Features.valenceByFrequency(test, vectorizer = vectorizer, cache_valence = cache_valence, stop_words = 'english')
    return train_matrix, test_matrix

print "=>Experiment 1: valence blog(80%) -> blog(80%) CV-10"
experiment1_b = Models.model_cv(maxent_model, blog_80, feat1, n_folds = 10)
print "=>Experiment 1: valence twitter+wiki -> blog(80%)"
experiment1_twb = experiment_maxent(tw, blog_80, feat1)
print "=>Experiment 1: valence twitter+wiki -> twitter+wiki CV-5)"
experiment1_tw = Models.model_cv(maxent_model, tw, feat1, n_folds = 5)

write_cv(out_file, "experiment 1 b", experiment1_b)
write_detailed(out_file, "experiment 1 twb", experiment1_twb)
write_cv(out_file, "experiment 1 tw", experiment1_tw)


# Feature set 2 - tf-idf
def feat2(train, test):
    state_info, train_matrix = Features.tfIdfSkLearn(train)
    _, test_matrix = Features.wordCountsSkLearn(test, vectorizer = state_info, stop_words = 'english')
    return train_matrix, test_matrix

print "=>Experiment 2: tf-idf blog(80%) -> blog(80%) CV-10"
experiment2_b = Models.model_cv(maxent_model, blog_80, feat2, n_folds = 10)
print "=>Experiment 2: tf-idf twitter+wiki -> blog(80%)"
experiment2_twb = experiment_maxent(tw, blog_80, feat2)
print "=>Experiment 2: tf-idf twitter+wiki -> twitter+wiki CV-5)"
experiment2_tw = Models.model_cv(maxent_model, tw, feat2, n_folds = 5)

write_cv(out_file, "experiment 2 b", experiment2_b)
write_detailed(out_file, "experiment 2 twb", experiment2_twb)
write_cv(out_file, "experiment 2 tw", experiment2_tw)

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

print "=>Experiment 3: valence + tf-idf blog(80%) -> blog(80%) CV-10"
experiment3_b = Models.model_cv(maxent_model, blog_80, feat3, n_folds = 10)
print "=>Experiment 3: valence + tf-idf twitter+wiki -> blog (80%)"
experiment3_twb = experiment_maxent(tw, blog_80, feat3)
print "=>Experiment 3: valence + tf-idf twitter+wiki -> twitter+wiki CV-5"
experiment3_tw = Models.model_cv(maxent_model, tw, feat3, n_folds = 5)

write_cv(out_file, "experiment 3 b", experiment3_b)
write_detailed(out_file, "experiment 3 twb", experiment3_twb)
write_cv(out_file, "experiment 3 tw", experiment3_tw)

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

print "=>Experiment 4: valence + tf-idf + punctuation blog(80%) -> blog(80%) CV-10"
experiment4_b = Models.model_cv(maxent_model, blog_80, feat4, n_folds = 10)
print "=>Experiment 4: valence + tf-idf + punctuation twitter+wiki -> blog(80%)"
experiment4_twb = experiment_maxent(tw, blog_80, feat4)
print "=>Experiment 4: valence + tf-idf + punctuation twitter+wiki -> twitter+wiki CV-5"
experiment4_tw = Models.model_cv(maxent_model, tw, feat4, n_folds = 5)

write_cv(out_file, "experiment 4 b", experiment4_b)
write_detailed(out_file, "experiment 4 twb", experiment4_twb)
write_cv(out_file, "experiment 4 tw", experiment4_tw)

# Just word valence and punctuation
def feat5(train, test):
    train_valence, test_valence = feat1(train, test)
    puncter, train_punct = Features.punctuation(train)
    _, test_punct = Features.punctuation(test, vectorizer = puncter)
    train_matrix = Features.append_features([train_valence, train_punct])
    test_matrix = Features.append_features([test_valence, test_punct])
    return train_matrix, test_matrix

print "=>Experiment 5: valence + punctuation blog(80%) -> blog(80%) CV-10"
experiment5_b = Models.model_cv(maxent_model, blog_80, feat5, n_folds = 10)
print "=>Experiment 5: valence + punctuation twitter+wiki -> blog(80%)"
experiment5_twb = experiment_maxent(tw, blog_80, feat5)
print "=>Experiment 5: valence + punctuation twitter+wiki -> twitter+wiki CV-5"
experiment5_tw = Models.model_cv(maxent_model, tw, feat5, n_folds = 5)

write_cv(out_file, "experiment 5 b", experiment5_b)
write_detailed(out_file, "experiment 5 twb", experiment5_twb)
write_cv(out_file, "experiment 5 tw", experiment5_tw)

# Valence + punctuation + tf-idf for words with a relevant POS
def feat6(train, test):
    normal_train, train_pos = map(list, zip(*train))
    normal_test, test_pos = map(list, zip(*test))
    train_f5, test_f5 = feat5(normal_train, normal_test)
    cter, train_cts = Features.keyPOSNGrams(train_pos, ["jj.*", "vb.*"], tf_idf = True)
    _, test_cts = Features.keyPOSNGrams(test_pos, ["jj.*", "vb.*"], vectorizer = cter, tf_idf= True)
    train_matrix = Features.append_features([train_f5, train_cts])
    test_matrix = Features.append_features([test_f5, test_cts])
    return train_matrix, test_matrix

def combine_with_pos(labeled, with_pos):
    return [((txt, pos_txt), label) for (txt,label),pos_txt in zip(labeled, with_pos)]


print "Experiment 6: valence + punctuation + key POS word counts blog(80%) -> blog(80%) CV-10"
experiment6_b = Models.model_cv(maxent_model, combine_with_pos(blog_80, blog_80_pos), feat6, n_folds = 10)
print "Experiment 6: valence + punctuation + key POS word counts twitter+wiki -> blog"
experiment6_twb = experiment_maxent(combine_with_pos(tw, tw_pos), combine_with_pos(blog_80, blog_80_pos), feat6)
print "Experiment 6: valence + punctuation + key POS word counts twitter+wiki -> twitter(test)"
experiment6_tw = Models.model_cv(maxent_model, combine_with_pos(tw, tw_pos), feat6, n_folds = 5)

write_cv(out_file, "experiment 6 b", experiment6_b)
write_detailed(out_file, "experiment 6 twb", experiment6_twb)
write_cv(out_file, "experiment 6 tw", experiment6_tw)


# Cross validation for blog -> blog experiment with best accuracy (to compare to original paper)
paper_comp = Models.model_cv(lambda : LogisticRegression(), blog, feat4, random_state = 1)
print "[Paper Comparison] CV-10 accuracy blog on blog:%.2f[+/-%.2f]" % paper_comp

write_cv(out_file, "paper comparison b", paper_comp)

out_file.close()