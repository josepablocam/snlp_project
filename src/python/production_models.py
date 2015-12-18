#A Python script to handle running data against production systems
import sys
sys.path.insert(0, '../../lib/alchemyapi_python')

from alchemyapi import AlchemyAPI
import json

import Globals
from ReadData import *

alchemyapi = AlchemyAPI()

#demo_text = ['Sometimes I really hate doing work.', 'Sometimes I really love doing work.', 'Sometimes there is work.']
#print(demo_text);

def sentimentToInt(sentimentType):
	if sentimentType == 'negative':
		return 1
	elif sentimentType == 'positive':
		return 1
	elif sentimentType == 'neutral':
		return 0

#twitter_test = to_utf8(prepareTwitterData(Globals.TWITTER_TEST, splitwords = False))
def blogDataRequests(blog_slice):
	blog = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords=False))
	blog_1 = blog[0:1000]
	blog_2 = blog[1000:2000]
	blog_3 = blog[2000:3000]
	blog_4 = blog[3000:4000]
	blog_5 = blog[4000:5000]

	if blog_slice == 1:
		blog_query = blog_1;
	elif blog_slice == 2:
		blog_query = blog_2;
	elif blog_slice == 3:
		blog_query = blog_3;
	elif blog_slice == 4:
		blog_query = blog_4;
	elif blog_slice == 5:
		blog_query = blog_5;

	count = 0;
	correct = 0;

	for item in blog_query:
		count = count + 1;
		print item[0];
		response = alchemyapi.sentiment('text', item[0])
		if response['status'] == 'OK' :
			print 'Guess: ' + str(sentimentToInt(response['docSentiment']['type']))
			print 'Gold: ' + item[1]
			guess = sentimentToInt(response['docSentiment']['type'])
			gold = item[1]
			match = (str(guess) == str(gold));
			print 'Match: ' + str(match)
			if match:
				correct = correct + 1
		else:
		    print('Error in targeted sentiment analysis call: ',
		          response['statusInfo'])
	print float(correct)/float(count);

def twitterDataRequests():

	twitter_test = to_utf8(prepareTwitterData(Globals.TWITTER_TEST, splitwords = False))

	for item in twitter_test:
		count = count + 1
		print item[0]
		response = alchemyapi.sentiment('text', item[0])
		if response['status'] == 'OK' :
			print 'Guess: ' + str(sentimentToInt(response['docSentiment']['type']))
			print 'Gold: ' + item[1]
			guess = sentimentToInt(response['docSentiment']['type'])
			gold = item[1]
			match = (str(guess) == str(gold));
			print 'Match: ' + str(match)
			if match:
				correct = correct + 1
		else:
		    print('Error in targeted sentiment analysis call: ',
		          response['statusInfo'])

blogDataRequests(5);

# response = alchemyapi.sentiment('text', twitter_test[0][0])
# if response['status'] == 'OK':
#     print('## Response Object ##')
#     print(json.dumps(response, indent=4))

#     print('')
#     print('## Targeted Sentiment ##')
#     print('type: ', response['docSentiment']['type'])

#     if 'score' in response['docSentiment']:
#         print('score: ', response['docSentiment']['score'])

#     if response['docSentiment']['type'] == ''
# else:
#     print('Error in targeted sentiment analysis call: ',
#           response['statusInfo'])

#for text in twitter_test:
	#count = count+1
#	print(text[1]);
# for text in demo_text:

# 	print('Processing text: ', text)
# 	print('')

# 	response = alchemyapi.sentiment('text', text)

# 	if response['status'] == 'OK':
# 	    print('## Response Object ##')
# 	    print(json.dumps(response, indent=4))

# 	    print('')
# 	    print('## Targeted Sentiment ##')
# 	    print('type: ', response['docSentiment']['type'])

# 	    if 'score' in response['docSentiment']:
# 	        print('score: ', response['docSentiment']['score'])
# 	else:
# 	    print('Error in targeted sentiment analysis call: ',
# 	          response['statusInfo'])