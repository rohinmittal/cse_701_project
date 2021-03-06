import glob, nltk, pickle

def prepareData(maxCount=None):
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	allData = ""

	seqs = []
	labels = [] 

	count = 0
	for f in glob.glob("../data/train/neg/*.txt"):
		f = open(f).read().lower()
		words = tokenizer.tokenize(f)
		allData += " ".join(words)
		seqs.append(words)
		labels.append(0)
		count += 1
		print (count)
	
	for f in glob.glob("../data/train/pos/*.txt"):
		f = open(f).read().lower()
		words = tokenizer.tokenize(f)
		allData += " ".join(words)
		seqs.append(words)
		labels.append(1)
		count += 1
		print (count)
	
	data = allData.split(" ")
	dist = nltk.FreqDist(data)

	print("Dumping Vocabulary...")
	f = open('../data/yelp.vocab', "w")
	for each in dist.most_common():
		f.write(each[0] + "\n")
	f.close()

	vocabSize = len(dist.most_common())
	new_seqs = []
	for each in seqs:
		new_seqs.append([(abs(hash(x)) % (vocabSize - 1) + 1) for x in each])
			
	return new_seqs, labels
	
if __name__ == "__main__":
	print ("Preparing data...")
	data, labels = prepareData()
	args = (data, labels)

	# dump data
	print('Dumping data...')
	f = open("../data/yelp.pickle", "wb")
	pickle.dump(args, f)
	f.close()
