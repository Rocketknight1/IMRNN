#!/usr/bin/python3

import numpy as np
import pdb
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def LoadNPZ(file):
    npz_input = np.load(file)
    i = 0
    training_data = []
    while True:
        try:
            training_data.append(npz_input['arr_{}'.format(i)])
            i+=1
        except:
            return training_data #This means we're done
   
def GetSamplesPerEpoch(training_data, batch_size):
    total = 0
    for array in training_data:
        total+=(len(array) // batch_size) * batch_size
    return total

def ChooseCharacter(prediction):
    
    
def BatchGenerator(training_data,batch_size):
    #The job of the batch generator is a little tricky
    #We want it to return batches that come from the same training array, since 
    #all training cases in the same array have the same length, and we can't mix lengths
    #in a single batch.
    #But we want to sample from each array in proportion to the number of remaining samples in it.
    #We want to sample -without- replacement, and so the probability that an array
    #will be chosen is not constant.
    #Algorithm is as follows:
    #1: Shuffle each array
    #2: Generate selection tokens in a list, where each array index is represented
    #proportionally to its length. This is a little inelegant, but it's not huge in memory.
    #3: Shuffle the selection list and repeatedly pop from it. Each time you pop an index, take batch_size
    #elements from the corresponding array, starting at the pointer, and yield them. 
    #Increment the pointer on that array by batch_size.
    while True:
        selections = []
        pointers = [0 for array in training_data]
        for i in range(len(training_data)):
            np.random.shuffle(training_data[i])
            selections.extend([i for x in range(training_data[i].shape[0]//batch_size)])
        
        random.shuffle(selections) # so we take in a random order
        while len(selections) > 0:
            array_index = selections.pop()
            startval = pointers[array_index]
            endval = startval + batch_size
            batch = training_data[array_index][startval:endval,:,:]
            features = batch[:,:-1,:]
            labels = batch[:,-1,:]
            yield (features,labels)
            pointers[array_index] = endval
        #When you fall off the end of the inner while loop the outer loop restarts 
        #and everything is set back up again
        print('Generator resetting! This should sync with the end of an epoch!')
        

#Reminder, dimensions go (samples,timesteps,characters)
BATCH_SIZE = 32
MAX_LEN = 12
training_data = LoadNPZ('training.npz')

generator = BatchGenerator(training_data,BATCH_SIZE)
samples_per_epoch = GetSamplesPerEpoch(training_data,BATCH_SIZE)
num_chars = training_data[0].shape[2]

model = Sequential()
model.add(LSTM(128, return_sequences=True,input_dim=num_chars))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(num_chars))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
while True:
    model.fit_generator(generator,samples_per_epoch,show_accuracy=True,nb_epoch=1)
    #Now to test a prediction from it
    while True:
        generated = 'a'
        input = np.zeros(1,min(len(generated),MAX_LEN),num_chars)
        for i in range(max(0,len(generated)-MAX_LEN),len(generated)):
            input[0,i,char_to_index[generated[i]]]=1
        outputchar = model.predict(input,batch_size=1,verbose=0)
        
            
    
    