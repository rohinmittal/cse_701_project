# LSTM model with dropout layer between Embedding and lstm layer & lstm layer and dense output layer
# also add dropout to the input of embedding layer

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
import pickle, random

def loadData(minFeatures_, maxFeatures_, *args):
        data, labels = args

        indices = [ind for ind in range(len(data))]
        random.shuffle(indices)

        split_at = int(len(data)*0.50)
        training_idx, testing_idx = indices[:split_at], indices[split_at:]

        X_train, y_train = [data[i] for i in training_idx], [labels[i] for i in training_idx]
        X_test, y_test = [data[i] for i in testing_idx], [labels[i] for i in testing_idx]

        # Ignore if: within top minFeatures_ highest occuring words
        # Ignore if: rank is more than maxFeatures_
        X_train = [[0 if rank > maxFeatures_ or rank <= minFeatures_ else rank for rank in sample] for sample in X_train]
        X_test = [[0 if rank > maxFeatures_ or rank <= minFeatures_ else rank for rank in sample] for sample in X_test]

        return X_train, y_train, X_test, y_test

if __name__ == "__main__":

	# load data from pickle file
	print("Loading data...")
	f = open('../data/imdb.pickle', 'rb')
	data = pickle.load(f)
	f.close()

	# set the arguments
	args = (data[0], data[1])
	min_features = 0
	max_features = 80000
	embedding_vector_length = 128 #dimension of each word in embedding layer
	memory_units = 128 #number of memory units / neurons in LSTM layer 
	batch_size = 128
	maxlen = 500 #length of each review after preprocessing. Shorter are appended with 0
	epochs = 10
	dropout_ = 0.2

	X_train, y_train, X_test, y_test = loadData(min_features, max_features, *args)
	print ("Training for " + str(len(X_train)) + " and validating for " + str(len(X_test)))

	# pad the reviews with 0 if shorter than maxlen else truncate
	X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

	# add layers to the model
	print('Build Model...')
	model = Sequential()

	# embedding layer
	model.add(Embedding(max_features, embedding_vector_length, dropout=dropout_))

	# add dropout layer
	model.add(Dropout(dropout_))

	# LSTM layer
	model.add(LSTM(memory_units))  

	# add dropout layer
	model.add(Dropout(dropout_))

	# dense output layer with a single neuron and sigmoid activation function
	# binary classification so single neuron required
	model.add(Dense(1, activation='sigmoid'))

	# use log-loss as loss function, ADAM optimizer is efficient
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# print model summary
	print(model.summary())

	print('Train Model...')
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, y_test))

	# finally evaluate the model
	print('Evaluate Model...')
	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

	print('Test score:', score)     
	print('Acc score:', acc)
