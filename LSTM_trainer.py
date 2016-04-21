#!/usr/bin/python3

import numpy as np
import pdb
import random
import time
import sys

from movie_parsing import MakeTitleChopList,MakeFirstCharDistribution

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
   
def GetSamplesPerEpoch(training_data, batch_size):
    usable_total = 0
    total = 0
    for array in training_data:
        usable_total+=(len(array) // batch_size) * batch_size
        total+=len(array)
    print("{} samples total, {} per epoch.".format(total,usable_total))
    return total

def ChooseCharacter(prediction):
    #Softmax layers output probabilities
    #np.random.multinomial selects one based on these probabilities
    #setting that cell to 1 and the others to 0
    #np.argmax returns the index of that cell
    #and so the index of the character we want
    
    #We also renormalize, even though softmax output should already be normalized.
    #Softmax is in float32, however, and rounding errors can cause multinomial
    #sampling to throw an error.
    prediction = prediction/sum(prediction) 
    
    return np.argmax(np.random.multinomial(1,prediction))

def EncodeCharIndicesToLabelMatrix(char_indices,end_index):
    label_matrix = np.zeros((len(char_indices),end_index+1),dtype=np.bool)
    for i in range(len(char_indices)):
        label_matrix[i,char_indices[i]] = 1
    return label_matrix

def EncodeCharVecsToTrainingArray(charvecs,end_index):
    num_timesteps = len(charvecs[0])
    output = np.zeros((len(charvecs),num_timesteps,end_index+1),dtype=np.bool)
    for i in range(len(charvecs)):
        for timestep in range(num_timesteps):
            output[i,timestep,charvecs[i]]=1
    return output

def BatchGenerator(training_data,batch_size,char_to_index,end_index):
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
        pointers = [0 for length_group in training_data]
        for i in range(len(training_data)):
            random.shuffle(training_data[i])
            selections.extend([i for x in range(len(training_data[i])//batch_size)])
        
        random.shuffle(selections) # so we take in a random order
        while len(selections) > 0:
            array_index = selections.pop()
            startval = pointers[array_index]
            endval = startval + batch_size
            batch = training_data[array_index][startval:endval]
            features = EncodeCharVecsToTrainingArray([charvec[:-1] for charvec in batch],end_index)
            labels = EncodeCharIndicesToLabelMatrix([charvec[-1] for charvec in batch],end_index)
            yield (features,labels)
            pointers[array_index] = endval
        #When you fall off the end of the inner while loop the outer loop restarts 
        #and everything is set back up again
        print('\n\nGenerator resetting! This will take a couple of seconds...\n')
        
#Reminder, dimensions go (samples,timesteps,characters)
BATCH_SIZE = 32
MAX_LEN = 12
TITLE_FILE = 'languages_crop'
TAGLINES_FILE = 'taglines_crop'
#oh god here comes the inelegant part
ALLOWED_CHARS = set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','.',';',':','!','?','-','&',',',' ','"',"'"])
#no-one saw that okay

char_to_index = {ch:i for i,ch in enumerate(ALLOWED_CHARS)}
index_to_char = {i:ch for i,ch in enumerate(ALLOWED_CHARS)}	
end_index = max(index_to_char.keys())+1

training_titles, titles = MakeTitleChopList(MAX_LEN,TITLE_FILE,ALLOWED_CHARS,char_to_index)

first_char_probs = MakeFirstCharDistribution(titles,index_to_char)
generator = BatchGenerator(training_titles,BATCH_SIZE,char_to_index,end_index)
samples_per_epoch = GetSamplesPerEpoch(training_titles,BATCH_SIZE)
num_chars = end_index + 1

model = Sequential()
model.add(LSTM(128, return_sequences=True,input_dim=num_chars))
#model.add(LSTM(512, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(num_chars))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam')
while True:
    model.fit_generator(generator,samples_per_epoch=65536,nb_epoch=1)
    generated = index_to_char[ChooseCharacter(first_char_probs)]
    #Now to test a prediction from it
    while True:
        input = np.zeros((1,min(len(generated),MAX_LEN),num_chars),dtype=np.bool)
        input_index_offset = max(0,len(generated)-MAX_LEN)
        for i in range(max(0,len(generated)-MAX_LEN),len(generated)):
            input[0,i-input_index_offset,char_to_index[generated[i]]]=1
        prediction = model.predict(input,batch_size=1,verbose=0)
        nextindex = ChooseCharacter(prediction[0])
        if nextindex == end_index:
            break
        nextchar = index_to_char[nextindex]
        generated = generated + nextchar
    
    print('Title: {}'.format(generated.title()))
            
    
    