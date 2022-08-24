from gc import callbacks
import string
import numpy as np
from fileinput import filename
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import *
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.regularizers import l2
from matplotlib import pyplot as plt


train_samples = []
train_labels = []

samples_filename = "Data/train-data.dat"
samples_file = open(samples_filename,'r')

labels_filename = "Data/train-label.dat"
labels_file = open(labels_filename,'r')

for text in samples_file: 
    word = text.split()
    counter = 1
    sentence_num = int(word[0].strip('<>'))
    vec = np.zeros(8520,dtype=int)
    for i in range(sentence_num):
        words_num = int(word[counter].strip('<>'))
        counter += 1
        for j in range(words_num):
            vec[int(word[counter])] += 1
            counter += 1
    train_samples.append(vec)
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
    sentence_num = int(word[0].strip('<>'))
    vec = np.zeros(8520,dtype=int)
    for i in range(sentence_num):
        words_num = int(word[counter].strip('<>'))
        counter += 1
        for j in range(words_num):
            vec[int(word[counter])] += 1
            counter += 1
    train_samples.append(vec)
for text in labels_file: 
    line = text.split()
    line = [int(x) for x in line]
    train_labels.append(line)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

norm = MinMaxScaler().fit(train_samples)
train_samples = norm.transform(train_samples)

kfold = KFold(n_splits=5, random_state=1, shuffle=True)

models = []

for i in range(5):
    #model = Sequential([
        #Dense(units=20, input_shape=(8520,), activation='relu'),
        #Dense(units=20, activation='sigmoid')
    #])
    #model = Sequential([
        #Dense(units=4270, input_shape=(8520,), activation='relu'),
        #Dense(units=20, activation='sigmoid')
    #])
    #model = Sequential([
        #Dense(units=8540, input_shape=(8520,), activation='relu'),
        #Dense(units=20, activation='sigmoid')
    #])
    #model = Sequential([
        #Dense(units=20, input_shape=(8520,), activation='relu'),
        #Dense(units=40, activation='relu'),
        #Dense(units=20, activation='sigmoid')
    #])
    #model = Sequential([
        #Dense(units=20, input_shape=(8520,), activation='relu'),
        #Dense(units=20, activation='relu'),
        #Dense(units=20, activation='sigmoid')
    #])
    #model = Sequential([
        #Dense(units=20, input_shape=(8520,), activation='relu'),
        #Dense(units=10, activation='relu'),
        #Dense(units=20, activation='sigmoid')
    #])
    #model = Sequential([
        #Dense(units=20, input_shape=(8520,), activation='relu'),
        #Dense(units=10, activation='relu', kernel_regularizer=l2(0.1)),
        #Dense(units=20, activation='sigmoid')
    #])
    #model = Sequential([
        #Dense(units=20, input_shape=(8520,), activation='relu'),
        #Dense(units=10, activation='relu', kernel_regularizer=l2(0.5)),
        #Dense(units=20, activation='sigmoid')
    #])
    model = Sequential([
        Dense(units=20, input_shape=(8520,), activation='relu'),
        Dense(units=40, activation='relu' , kernel_regularizer=l2(0.9)),
        Dense(units=20, activation='sigmoid')
    ])
    #opt= SGD(learning_rate=0.001,momentum=0.2)
    opt = SGD(learning_rate=0.001,momentum=0.6)
    #opt = SGD(learning_rate=0.05,momentum=0.6)
    #opt = SGD(learning_rate=0.1,momentum=0.6)
    model.compile(
        optimizer=opt, 
        loss='binary_crossentropy', 
        metrics=['MeanSquaredError','accuracy']
    )
    models.append(model)


i = 0 
for train, test in kfold.split(train_samples,train_labels):
    hist = models[i].fit(
        x=train_samples[train], 
        y=train_labels[train], 
        validation_data = (train_samples[test],train_labels[test]),
        batch_size=10, 
        epochs=40, 
        verbose=1,
    )
   
    scores = models[i].evaluate(
        x=train_samples[test],
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
