# Simple exploratory analysis

import Globals
from ReadData import *
import Features
import clean_wiki
import numpy
import os.path
from collections import Counter

def histogram_cts(ct_matrix, bucket_size = 5, **kwargs):
    word_freq = ct_matrix.sum(axis = 0)
    max_freq = word_freq.max()
    buckets = range(0, max_freq + bucket_size, bucket_size)
    return numpy.histogram(word_freq, bins = buckets, **kwargs)

def histo_to_tuples(data, name):
    names = [name] * len(data[0])
    cts = data[0]
    buckets = data[1][1:] if len(cts) != len(data[1]) else data[1]
    return zip(names, buckets, cts)



# Preparing data
# Possible training data
twitter_train = to_utf8(prepareTwitterData(Globals.TWITTER_TRAIN, splitwords = False))
twitter_test = to_utf8(prepareTwitterData(Globals.TWITTER_TEST, splitwords = False))
wiki = prepareWikiData(Globals.WIKI_TRAIN, splitwords= False)
# clean wiki data
wiki = clean_wiki.clean_wiki(wiki)
tw = twitter_train + wiki
# blog data
blog = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords=False))

# Counts of unigrams -> data is sparse in all three sources
def count_unigrams(outpath):
    tw_cter, twitter_cts = Features.wordCountsSkLearn(Features.getX(tw), stop_words = 'english')
    blog_cter, blog_cts = Features.wordCountsSkLearn(Features.getX(blog), stop_words = 'english')

    # Total number of non-stop-word unigrams
    unigrams = set(tw_cter.vocabulary_.keys() + blog_cter.vocabulary_.keys())
    print "Data has %d distinct unigrams" % len(unigrams)

    # Distribution of unigram cts
    twitter_unigram_histo = histogram_cts(twitter_cts)
    blog_unigram_histo = histogram_cts(blog_cts)

    unigram_histo = histo_to_tuples(twitter_unigram_histo, 'twitter+wiki') + \
                    histo_to_tuples(blog_unigram_histo, 'blog')

    # Write out to csv
    with open(outpath, 'w') as unigram_histo_file:
        for elem in unigram_histo:
            unigram_histo_file.write("%s,%d,%f\n" % elem)
    return 0

def count_labels(outpath):
    tw_cts = Counter(Features.getY(tw))
    blog_cts = Counter(Features.getY(blog))
    cts = zip(["twitter+wiki", "blog"], [tw_cts, blog_cts])
    # Write out to csv
    with open(outpath, 'w') as labels_histo_file:
        for src, counter in cts:
            for k, v in counter.iteritems():
                labels_histo_file.write("%s,%s,%d\n" % (src, k, v))
    return 0


def main(actions, outdir):
    if 'ct_unigrams' in actions:
        count_unigrams(os.path.join(outdir, "count_unigrams.csv"))
    if 'ct_labels' in actions:
        count_labels(os.path.join(outdir, "count_labels.csv"))
    # add more stuff here



