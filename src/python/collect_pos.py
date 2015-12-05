# Requires having Stanford Core NLP
# And Gate
import Globals
import ReadData
import subprocess
import tempfile

# http://nlp.stanford.edu/software/pos-tagger-faq.shtml
stanford_cmd = ['java', '-cp', 'stanford-postagger.jar', 'edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', 'models/english-left3words-distsim.tagger', '-outputFormat', 'slashTags',
                '-sentenceDelimiter', 'newline', '-textFile']
# favor faster tagger
# https://gate.ac.uk/wiki/twitter-postagger.html
gate_cmd = ['java', '-jar',  'twitie_tag.jar', 'models/gate-EN-twitter-fast.model']

def tag(cmd, cwd, infilename, outfilehandle):
    full_cmd = cmd + [ infilename ]
    p = subprocess.Popen(full_cmd, cwd = cwd, stdout = outfilehandle)
    return p.wait()

def write_temp(data):
    # write data a line per sentence, to temporary file
    temp = tempfile.NamedTemporaryFile()
    for obs in data:
        temp.write(obs + "\n")
    temp.flush()
    return temp


# Wikipedia data is already in a sentence per line format
with open(Globals.WIKI_POS, "w") as pos_file:
    tag(stanford_cmd, Globals.STANFORD_PATH, Globals.WIKI_TRAIN, pos_file)


# Twitter, we will read in data, and write to temporary file
# Training data
twitter_train_raw = ReadData.readTwitterData(Globals.TWITTER_TRAIN, splitwords=False)
twitter_temp_file = write_temp(twitter_train_raw[1])
with open(Globals.TWITTER_TRAIN_POS, 'w') as pos_file:
    tag(gate_cmd, Globals.GATE_PATH, twitter_temp_file.name, pos_file)
twitter_temp_file.close()

# Test data
twitter_test_raw = ReadData.readTwitterData(Globals.TWITTER_TEST, splitwords= False)
twitter_temp_file = write_temp(twitter_test_raw[1])
with open(Globals.TWITTER_TEST_POS, 'w') as pos_file:
    tag(gate_cmd, Globals.GATE_PATH, twitter_temp_file.name, pos_file)
twitter_temp_file.close()

# Blog data
blog = ReadData.readBlogData(Globals.BLOG_DATA, splitwords = False)
blog = [ txt for txt,label in blog ]
blog_temp_file = write_temp(map(lambda x: " ".join(x), blog[1]))
with open(Globals.BLOG_POS, 'w') as pos_file:
    tag(gate_cmd, Globals.GATE_PATH, blog_temp_file.name, pos_file)
blog_temp_file.close()





