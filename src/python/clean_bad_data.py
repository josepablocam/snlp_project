f = open("../../data/alchemy_results/out_3.txt");
line = f.readline();

correct = 0;
total = 0;

while line:
	total = total + 1;
	print line.rstrip('\n');
	line = f.readline();
	print line.rstrip('\n');
	guess = line[7:]
	line = f.readline();
	print line.rstrip('\n');
	gold = line[6:]
	line = f.readline();
	line = f.readline();
	correct = correct + (guess == gold)
	print "Match: " + str(guess == gold)

f.close();
print correct
print total
print float(correct) / float(total)