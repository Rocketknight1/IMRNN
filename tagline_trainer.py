#!/usr/bin/python3

import numpy as np
import pdb
import random
import time
import sys

from movie_parsing import MakeTaglineTraining,MakeFirstCharDistribution

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import PReLU, ELU, SReLU
from training_functions import GetSamplesPerEpoch,ChooseCharacter,OnlineTaglineGenerator,EncodeSingleCharVec
        
#Reminder, dimensions go (samples,timesteps,characters)
MAX_LEN = 12
TITLE_FILE = 'languages_crop'
TAGLINES_FILE = 'taglines_crop'
#oh god here comes the inelegant part
ALLOWED_CHARS = set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','.',';',':','!','?','-','&',',',' ','"',"'"])
#no-one saw that okay

char_to_index = {ch:i for i,ch in enumerate(ALLOWED_CHARS)}
index_to_char = {i:ch for i,ch in enumerate(ALLOWED_CHARS)}	
end_index = max(index_to_char.keys())+1

training_taglines, titles = MakeTaglineTraining(MAX_LEN,TITLE_FILE,TAGLINES_FILE,ALLOWED_CHARS,char_to_index,end_index)

first_char_probs = MakeFirstCharDistribution(titles,index_to_char)
generator = OnlineTaglineGenerator(training_taglines,char_to_index,end_index)
num_chars = end_index + 1

title_branch = Sequential()
title_branch.add(LSTM(256, return_sequences=True,input_dim=num_chars))
title_branch.add(LSTM(512, return_sequences=False))

tagline_branch = Sequential()
tagline_branch.add(LSTM(256, return_sequences=True,input_dim=num_chars))
tagline_branch.add(LSTM(512, return_sequences=False))

merged_branches = Sequential()
merged_branches.add(Merge([title_branch, tagline_branch], mode='concat', concat_axis=1))
merged_branches.add(Dense(1024,init='he_normal'))
merged_branches.add(PReLU())
merged_branches.add(Dense(num_chars,init='he_normal',activation='softmax'))

merged_branches.compile(loss='categorical_crossentropy', optimizer='Adam')
while True:
    merged_branches.fit_generator(generator,samples_per_epoch=65536,nb_epoch=1)
    generated = index_to_char[ChooseCharacter(first_char_probs)]
    #Now to test a prediction from it
    while True:
        
        input = np.zeros((1,min(len(generated),MAX_LEN),num_chars),dtype=np.bool)
        input_index_offset = max(0,len(generated)-MAX_LEN)
        for i in range(max(0,len(generated)-MAX_LEN),len(generated)):
            input[0,i-input_index_offset,char_to_index[generated[i]]]=1
        prediction = merged_branches.predict(input,batch_size=1,verbose=0)
        nextindex = ChooseCharacter(prediction[0])
        if nextindex == end_index:
            break
        nextchar = index_to_char[nextindex]
        generated = generated + nextchar
    
    print('Title: {}'.format(generated.title()))
            
    
    