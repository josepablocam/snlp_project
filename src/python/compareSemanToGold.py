#A Python script to handle running data against production systems
import sys
import Globals
from ReadData import *

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
def blogStats():
	blog = to_utf8(prepareBlogData(Globals.BLOG_DATA, splitwords=False))
	f = open("../../data/semantria_results/blog_sentiments.txt");

	semantria_sentiment = []

	for line in f:
		semantria_sentiment.append(sentimentToInt(line.rstrip()))
	f.close();

	count = 0;
	correct = 0;
	FP = 0;
	FN = 0;
	TP = 0;
	for item, line in zip(blog, semantria_sentiment):
		count = count + 1
		guess = line
		gold = item[1]
		match = (str(guess) == str(gold));
		if match:
			correct = correct + 1;
		if (str(guess) == "1") & (str(gold) == "1"):
			TP = TP + 1;
		if (str(guess) == "0") & (str(gold) == "1"):
			FN = FN + 1;
		if (str(guess) == "1") & (str(gold) == "0"):
			FP = FP + 1;
	print count
	print correct
	print TP
	print FP
	print FN

def twitterStats():

	f = open("../../data/semantria_results/twitter_sentiments.txt");
	twitter_test = to_utf8(prepareTwitterData(Globals.TWITTER_TEST, splitwords = False))

	semantria_sentiment = []

	for line in f:
		semantria_sentiment.append(sentimentToInt(line.rstrip()))
	f.close();

	count = 0;
	correct = 0;
	FP = 0;
	FN = 0;
	TP = 0;
	for item, line in zip(twitter_test, semantria_sentiment):
		count = count + 1
		guess = line
		gold = item[1]
		match = (str(guess) == str(gold));
		if match:
			correct = correct + 1;
		if (str(guess) == "1") & (str(gold) == "1"):
			TP = TP + 1;
		if (str(guess) == "0") & (str(gold) == "1"):
			FN = FN + 1;
		if (str(guess) == "1") & (str(gold) == "0"):
			FP = FP + 1;
	print count
	print correct
	print TP
	print FP
	print FN

#twitterStats();
blogStats();
