# Experiments involving maximum entropy classifiers
import Globals
from ReadData import *
import Features
import clean_wiki
import Models
from sklearn.cross_validation import train_test_split
import sys

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
def experiment_linearK(train, test, featurizer):
    return experiment(Models.report_SVM_linearK, train, test, featurizer)

# Feature-Set 1 - averaged sentiment valence
cache_valence = dict()
def feat1(train, test):
    vectorizer, train_matrix = Features.valenceByFrequency(train, vectorizer = None, cache_valence = cache_valence, stop_words = 'english')
    _, test_matrix = Features.valenceByFrequency(test, vectorizer = vectorizer, cache_valence = cache_valence, stop_words = 'english')
    return train_matrix, test_matrix

# Feature set 2 - tf-idf
def feat2(train, test):
    state_info, train_matrix = Features.tfIdfSkLearn(train)
    _, test_matrix = Features.wordCountsSkLearn(test, vectorizer = state_info, stop_words = 'english')
    return train_matrix, test_matrix

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

# Just word valence and punctuation
def feat5(train, test):
    train_valence, test_valence = feat1(train, test)
    puncter, train_punct = Features.punctuation(train)
    _, test_punct = Features.punctuation(test, vectorizer = puncter)
    train_matrix = Features.append_features([train_valence, train_punct])
    test_matrix = Features.append_features([test_valence, test_punct])
    return train_matrix, test_matrix

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

# Valence + punctuation + tf-idf for uni/bi-grams with a relevant POS in either unit
def feat7(train, test):
    normal_train, train_pos = map(list, zip(*train))
    normal_test, test_pos = map(list, zip(*test))
    train_f5, test_f5 = feat5(normal_train, normal_test)
    cter, train_cts = Features.keyPOSNGrams(train_pos, ["jj.*", "vb.*"], tf_idf = True, ngram_range = (1, 2))
    _, test_cts = Features.keyPOSNGrams(test_pos, ["jj.*", "vb.*"], vectorizer = cter, tf_idf= True, ngram_range = (1, 2))
    train_matrix = Features.append_features([train_f5, train_cts])
    test_matrix = Features.append_features([test_f5, test_cts])
    return train_matrix, test_matrix

def combine_with_pos(labeled, with_pos):
    return [((txt, pos_txt), label) for (txt,label),pos_txt in zip(labeled, with_pos)]

# Writing experiment results
def write_cv(f, details, acc_and_err):
    f.write(details + "\n")
    f.write("accuracy:%f\nstd-error:%f\n\n" % acc_and_err)

def write_detailed(f, details, report):
    f.write(details + "\n")
    f.write(report + "\n\n")

def log_results(out_file, msg, tw_cv, blog_cv, featurizer):
    if blog_cv:
        print msg + " blog(80%) -> blog(80%) CV-10"
        results_b = Models.model_cv(Models.LinearSVC, blog_cv, featurizer, n_folds = 10)
        write_cv(out_file, msg + " b", results_b)
    if tw_cv and blog_cv:
        print msg + " twitter+wiki -> blog(80%)"
        results_twb = experiment_linearK(tw_cv, blog_cv, featurizer)
        write_detailed(out_file, msg + " twb", results_twb)
    if tw_cv:
        print msg + " twitter+wiki -> twitter+wiki CV-5"
        results_tw = Models.model_cv(Models.LinearSVC, tw_cv, featurizer, n_folds = 5)
        write_cv(out_file, msg + " tw", results_tw)

# Actions: performing different experiments
def first_pass(out_file):
    log_results(out_file, "==>Experiment 1: valence", tw, blog_80, feat1)
    log_results(out_file, "==>Experiment 2: tf-idf", tw, blog_80, feat2)
    log_results(out_file, "==>Experiment 3: valence + tf-idf", tw, blog_80, feat3)
    log_results(out_file, "==>Experiment 4: valence + tf-idf + punctuation", tw, blog_80, feat4)
    log_results(out_file, "==>Experiment 5: valence + punctuation", tw, blog_80, feat5)
    tw_combined = combine_with_pos(tw, tw_pos)
    blog_combined = combine_with_pos(blog_80, blog_80_pos)
    log_results(out_file, "==>Experiment 6: valence + punctuation + key POS word counts", tw_combined, blog_combined, feat6)
    log_results(out_file, "==>Experiment 7: valence + punctuation + key POS uni/bigram counts", tw_combined, blog_combined, feat7)
     # Cross validation for blog -> blog experiment with best accuracy (to compare to original paper)
    log_results(out_file, "===> Paper comparison: CV-10 accuracy blog on blog", None, blog_80, feat4)

def error_analyze(make_model, train_data, test_data, featurizer):
    matrices = Features.make_experiment_matrices(train_data, test_data, featurizer)
    model = make_model()
    model.fit(matrices['train_X'], matrices['train_Y'])
    bins = [v / 100.0 for v in range(50, 110, 5)]
    ext_preds = Models.extended_predict(model, matrices['test_X'], matrices['test_Y'])
    return Models.error_analysis(ext_preds, bins = bins)

def error_analysis(featurizers = None):
    if not featurizers:
        featurizers = [feat4, feat5, feat4]
    b_train, b_test = train_test_split(blog_80, test_size = 0.1, random_state = 1)
    blog_errors = error_analyze(Models.LogisticRegression, b_train, b_test, featurizers[0])
    twb_errors = error_analyze(Models.LogisticRegression, tw, blog_80, featurizers[1])
    tw_train, tw_test = train_test_split(tw, test_size = 0.1, random_state = 1)
    tw_errors = error_analyze(Models.LogisticRegression, tw_train, tw_test, featurizers[2])
    # TODO: figure out best way to return these results
    # so far used interactively
    print blog_errors['confusion_matrix']
    print twb_errors['confusion_matrix']
    print tw_errors['confusion_matrix']



def tuning_l2_penalty(out_file, featurizers = None):
    # featurizers for blog/blog, twitter+wiki/blog and twitter+wiki/twitter+wiki respectively
    if not featurizers:
        featurizers = [feat4, feat5, feat4]
    # used to weigh L-2 penalty
    c_vals = [ v / 100.0 for v in range(50, 110, 10)]
    # data splits used
    b_train, b_test = train_test_split(blog_80, test_size = 0.1, random_state = 1)
    tw_train, tw_test = train_test_split(tw, test_size = 0.1, random_state = 1)
    # count sizes only once
    n_btest = float(len(b_test))
    n_b80 = float(len(blog_80))
    n_twtest = float(len(tw_test))

    for c_val in c_vals:
        print "Running l-2 tunning for C:%.2f" % c_val
        # Using split validation, as otherwise too slow
        make_model = lambda: Models.LogisticRegression(C = c_val)
        blog_errors = error_analyze(make_model, b_train, b_test, featurizers[0])
        twb_errors = error_analyze(make_model, tw, blog_80, featurizers[1])
        tw_errors = error_analyze(make_model, tw_train, tw_test, featurizers[2])

        blog_acc = 1 - len(blog_errors["error_indices"]) / n_btest
        twb_acc = 1 - len(twb_errors['error_indices']) / n_b80
        tw_acc = 1 - len(tw_errors['error_indices']) / n_twtest
        # write to file provided
        out_file.write("C=%f\n" % c_val)
        out_file.write("b=%f, twb=%f, tw=%f\n\n" % (blog_acc, twb_acc, tw_acc))


def test_perf(out_file):
    # finally...test performance
    results_b = experiment_linearK(blog_80, blog_20, feat4)
    results_twb = experiment_linearK(tw, blog_20, feat5)
    results_tw = experiment_linearK(tw, twitter_test, feat4)
    write_detailed(out_file, "Blog->Blog Test Perf", results_b)
    write_detailed(out_file, "Twitter+Wiki->Blog Test Perf", results_twb)
    write_detailed(out_file, "Twitter+Wiki->Twitter Test Perf", results_tw)


def main(action, out_path):
    out_file = open(out_path, "w")
    if action == "first_pass":
        first_pass(out_file)
    elif action == "tuning":
        tuning_l2_penalty(out_file)
    elif action == "error_analysis":
        # currently used interactively :(
        error_analysis()
    elif action == "test_perf":
        test_perf(out_file)
    else:
        raise ValueError("Unspecified action: %s" % action)

    out_file.close()
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("provide action and results path")
    action = sys.argv[1]
    out_path = sys.argv[2]
    main(action, out_path)
