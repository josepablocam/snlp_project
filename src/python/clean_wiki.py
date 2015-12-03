# A script to systematically clean wikipedia "no-emotion" data
# feel free to add any more conditions, as long as you appropriately comment
import re

# lambdas to apply, with descriptive name, so we can keep track of what's what
wiki_filters = {}
# only start with letters
wiki_filters['alpha_start'] = lambda x: x[0].isalpha()
wiki_filters['no_quotes'] = lambda x: not '"' in x
# better than removing all dashes...gets rid of a lot of table stuff
wiki_filters['no_dash_space'] = lambda x: not ' - ' in x
wiki_filters['no_eq_ineq'] = lambda x: not '=' in x and not '<' in x and not '>' in x
# parenthesized numbers tend to be annoying wikipedia stuff
wiki_filters['no_paren_nums'] = lambda x: not re.match(r".*\([0-9]+\)", x)
# though about removing stuff without a verb, but to be fair, tweets might not have verbs either


def clean_wiki(data, filters = wiki_filters, filter_names = None):
    """
    Filter out elements in wiki data based on set of filters
    :param list: list of wikipedia sentences (each sentence being a string)
    :param filter: dictionary of possible filters to apply (filter should return True to keep observation)
    :param filter_names: a list of filter names to apply, if None (default) applies all
    :return: cleaned list
    """
    # wrap application to handle tuple and non-tuple
    satisfies = lambda f, e: f(e[0]) if isinstance(data[0], tuple) else lambda f, e: f(e)
    defined_filters = set(filters.keys())
    applied = defined_filters.intersection(filter_names) if filter_names != None else defined_filters
    if not applied:
        raise ValueError("Must apply at least one filter. try filter_names = None")
    if filter_names != None and len(applied) != len(filter_names):
        missing = applied.difference(filter_names)
        raise ValueError("Unspecified filter names: %s" % ", ".join(missing))
    # ok down to business now
    clean_data = data
    for filter_name in applied:
        print "Applying filter: %s" % filter_name
        filter = filters[filter_name]
        clean_data = [ obs for obs in clean_data if satisfies(filter, obs) ]
    return clean_data


# an example
if __name__ == "__main__":
    import ReadData
    import Globals
    wikidata = ReadData.prepareWikiData(Globals.WIKI_TRAIN, splitwords= False)
    cleandata = clean_wiki(wikidata)
    print "Reduced data by %f" % (len(cleandata) / float(len(wikidata)))

