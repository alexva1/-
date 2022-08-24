import tensorflow as tf
from numpy import array
from keras.preprocessing.sequence import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils.data_utils import pad_sequences
from sklearn.model_selection import KFold
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from keras.layers import LSTM
from keras.layers import CuDNNLSTM

train_samples = []
train_labels = []

samples_filename = "Data/train-data.dat"
samples_file = open(samples_filename,'r')

labels_filename = "Data/train-label.dat"
labels_file = open(labels_filename,'r')

max_length = 0

for text in samples_file:
    word = text.split()
    counter = 1
    doc = []
    sentence_num = int(word[0].strip('<>'))
    for i in range(sentence_num):
        words_num = int(word[counter].strip('<>'))
        counter += 1
        for j in range(words_num):
            doc.append(int(word[counter]))
            counter += 1
    train_samples.append(doc)
    if( max_length < len(doc)):
        max_length = len(doc)

for text in labels_file: 
    line = text.split()
    line = [int(x) for x in line]
    train_labels.append(line)

samples_filename = "Data/test-data.dat"
samples_file = open(samples_filename,'r')

labels_filename = "Data/test-label.dat"
labels_file = open(labels_filename,'r')


for text in samples_file:
    word = text.split()
    counter = 1
    doc = []
    sentence_num = int(word[0].strip('<>'))
    for i in range(sentence_num):
        words_num = int(word[counter].strip('<>'))
        counter += 1
        for j in range(words_num):
            doc.append(int(word[counter]))
            counter += 1
    train_samples.append(doc)
    if( max_length < len(doc)):
        max_length = len(doc)

for text in labels_file: 
    line = text.split()
    line = [int(x) for x in line]
    train_labels.append(line)


padded_train_samples = pad_sequences(train_samples,maxlen=max_length,padding='post')

padded_train_samples = np.array(padded_train_samples)
train_labels = np.array(train_labels)


vocab_size = 8520


kfold = KFold(n_splits=5, random_state=1, shuffle=True)

models = []

for i in range(5): 
    #model = Sequential()
    #model.add(tf.keras.layers.Embedding(vocab_size,64,input_length=max_length))
    #model.add(Flatten())
    #model.add(Dense(units=20,activation='relu'))
    #model.add(Dense(units=40, activation='relu'))
    #model.add(Dense(20, activation='sigmoid'))

    #model = Sequential()
    #model.add(tf.keras.layers.Embedding(vocab_size,64,input_length=max_length))
    #model.add(LSTM(units = 40 ))
    #model.add(Dense(20, activation='sigmoid'))
    
    model = Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size,64,input_length=max_length))
    model.add(LSTM(units = 20 , return_sequences=True))
    model.add(LSTM(units = 40 ))
    model.add(Dense(20, activation='sigmoid'))

    opt = SGD(learning_rate=0.001,momentum=0.6)
    model.compile(
        optimizer=opt, 
        loss='binary_crossentropy', 
        metrics=['MeanSquaredError','accuracy','binary_accuracy']
    )
    models.append(model)



i = 0 
for train, test in kfold.split(padded_train_samples,train_labels):
    hist = models[i].fit(
        x=padded_train_samples[train], 
        y=train_labels[train], 
        validation_data = (padded_train_samples[test],train_labels[test]),
        batch_size=10, 
        epochs=40, 
        verbose=1,
    )
   
    scores = models[i].evaluate(
        x=padded_train_samples[test],
        y=train_labels[test],
        batch_size = 1, 
        verbose=2
    )
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('figures/accuracy'+str(i+1)+'.png')
    
    plt.clf()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('figures/loss'+str(i+1)+'.png')

    plt.clf()
    i += 1
