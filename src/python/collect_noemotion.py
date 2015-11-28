# Small script to collect random sentences from wikipedia
# We assume wikipedia is as close to non-emotional as possible
# since it is "factual"

import wikipedia as wiki
import re
import argparse
import sys
import nltk

# CONSTANTS
# How many times do we allow the wiki api to timeout
TIMEOUT_CT_LIMIT = 100
# marker for sections/subsections in content
HEADER_START = "="
# minimum number of spaces in a "sentence" to remove things like names, section headers etc
SPACE_CT = 4

# Sentence detector model from nltk (punctuation + capitals + etc)
ssplit = nltk.data.load('tokenizers/punkt/english.pickle')

def filter_noise(elemlist):
    """
    Remove anything with less than 3 spaces (removes random noise and citations)
    """
    return [ elem for elem in elemlist if elem.count(" ") > SPACE_CT ]

def not_header(txt):
    return len(txt) > 0 and txt[0] != HEADER_START
    
def split_content(txt):
    """
    Take content text and split into sentences
    """
    sections = re.split(r'\n+', txt)
    if len(sections) >= 1:
        sections = [ section for section in sections if not_header(section) ]
        # remove super short sentences, which are probably just headers
        sections = filter_noise(sections)
        flat_sentences = [ sentence for section in sections for sentence in ssplit.tokenize(section.strip()) ]
        return flat_sentences
    else:
        return []    
    
    
def get_random_page(nsent): 
    """
    Get sentences from random page (up to nsent sentences)
    """
    page = wiki.page(wiki.random())
    sentences = split_content(page.content)
    sentences = filter_noise(sentences)
    ct = len(sentences)
    takenct = min(nsent, ct)
    takensent = sentences[:takenct]
    return (takensent, takenct)


def get_random(ntotal, outpath):
    """
    Collect ntotal of sentences from random wiki articles and write to outpath
    """
    outfile = file(outpath, "a")
    ct = 0
    ct_timeouts = 0
    while ct < ntotal:
        print "Collected " + str(ct)
        if ct_timeouts > TIMEOUT_CT_LIMIT:
            print "Timeouts in excess of " + str(TIMEOUT_CT_LIMIT)
            outfile.close()
            sys.exit(1)
        try:
            (sentences, addct) = get_random_page(ntotal - ct)
            for sentence in sentences:
                utf8sentence = sentence.encode('UTF-8')
                outfile.write(utf8sentence + "\n")
            ct += addct
        except wiki.exceptions.HTTPTimeoutError as e:
            ct_timeouts += 1
            print "Timeout error, enabling rate limit"
            wiki.set_rate_limiting(True)
        except wiki.exceptions.WikipediaException:
            # ignore others I suppose...
            pass 
    outfile.close()
            
                    
def main(ntotal, outpath):
    print "Collecting "  + str(ntotal) + " sentences"
    print "Output File: " + str(outpath)
    get_random(ntotal, outpath)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Collect wikipedia sentences")
    parser.add_argument('ntotal', metavar = 'n', type = int, help = "Number of sentences to collect")
    parser.add_argument('outpath', metavar = 'p', type = str, help = "Output file")
    args = parser.parse_args()
    if args.ntotal <= 0:
        print "ntotal must be >= 0"
        sys.exit(1)
    main(args.ntotal, args.outpath)
    
               