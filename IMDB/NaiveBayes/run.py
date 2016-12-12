import nltk, random, glob
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')
stopWords = stopwords.words('english')

def word_feats(words):
	return dict([(word, True) for word in words])

def preProcess(review):
	#remove punctuation and translate to lower case
	words = tokenizer.tokenize(review.lower())

	#remove stop words
	content = [w for w in words if w not in stopWords]

	#stem the words
	stemmer = nltk.stem.porter.PorterStemmer()
	stems = [stemmer.stem(w) for w in content]
	return stems

def loadData(src, splitAt):
	featureSet = []
	count = 0
	for f in glob.glob(src + "neg/*.txt"):
		review = open(f).read()
		words = preProcess(review)
		featureSet.append((word_feats(words), 0))
		count += 1
		print(count)

	for f in glob.glob(src + "pos/*.txt"):
		review = open(f).read()
		words = preProcess(review)
		featureSet.append((word_feats(words), 1))
		count += 1
		print(count)

	indices = [i for i in range(len(featureSet))]
	random.shuffle(indices)

	split_at = int(len(featureSet)*splitAt)
	training_idx, testing_idx = indices[:split_at], indices[split_at:]

	train_set = [featureSet[i] for i in training_idx]
	test_set = [featureSet[i] for i in testing_idx]

	return train_set, test_set

def trainClassifier(train_set):
	print('Training classifier...')
	classifier = NaiveBayesClassifier.train(train_set)
	return classifier

def testClassifier(classifier, test_set):
	print('Testing classifier...')
	return nltk.classify.accuracy(classifier, test_set)

if __name__ == "__main__":
	X_train, X_test = loadData("../data/train/", 0.50)
	classifier = trainClassifier(X_train)
	acc = testClassifier(classifier, X_test)
	print ('Accuracy : ' + str(acc*100) + '%')
