#!/usr/bin/python3


import numpy as np
import pdb
import random
import time
import sys
import os
import pickle


from movie_parsing import MakeTitleTraining,MakeFirstCharDistribution

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import PReLU, ELU, SReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from training_functions import GetSamplesPerEpoch,ChooseCharacter,TitleBatchGenerator,GenerateTitle,MakeTrainingAndValidationSets
from keras.optimizers import Adam
        
#Reminder, dimensions go (samples,timesteps,characters)
BATCH_SIZE = 128
MAX_LEN = 12
TITLE_FILE = 'languages_crop'
TAGLINES_FILE = 'taglines_crop'
#oh god here comes the inelegant part
ALLOWED_CHARS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','.',';',':','!','?','-','&',',',' ','"',"'"]
#no-one saw that okay

char_to_index = {ch:i for i,ch in enumerate(ALLOWED_CHARS)}
index_to_char = {i:ch for i,ch in enumerate(ALLOWED_CHARS)}	
end_index = max(index_to_char.keys())+1

training_titles, titles = MakeTitleTraining(MAX_LEN,TITLE_FILE,ALLOWED_CHARS,char_to_index,end_index)

training_data, validation_data = MakeTrainingAndValidationSets(training_titles)

first_char_probs = MakeFirstCharDistribution(titles,index_to_char)
training_generator = TitleBatchGenerator(training_data,BATCH_SIZE,char_to_index,index_to_char,end_index)
validation_generator = TitleBatchGenerator(validation_data,BATCH_SIZE,char_to_index,index_to_char,end_index)
samples_per_epoch = GetSamplesPerEpoch(training_titles,BATCH_SIZE)
num_chars = end_index + 1

model = Sequential()
model.add(LSTM(1024, return_sequences=True,input_dim=num_chars))
#model.add(TimeDistributed(SReLU()))
model.add(LSTM(1024, return_sequences=False))
#model.add(TimeDistributed(SReLU()))
model.add(Dense(1024,init='he_normal'))
model.add(Dropout(0.5))
model.add(SReLU())
model.add(Dense(num_chars))
model.add(Activation('softmax'))

adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam)
checkpointer = ModelCheckpoint(filepath='latestweights.hdf5',verbose=1,save_best_only=False)
lowest_loss = 2 #We don't really care for models above this

while True:
	callback = model.fit_generator(training_generator,validation_data=validation_generator,samples_per_epoch=327680,nb_val_samples=131072,nb_epoch=1,max_q_size=50)
	loss = float(callback.history['loss'][0])
	val_loss = float(callback.history['val_loss'][0])
	if val_loss < lowest_loss - 0.05:
		weightfolder = 'savedmodels/titletraining_weightsatloss_{0:.2f}'.format(val_loss)
		if not os.path.isdir(weightfolder):
			os.makedirs(weightfolder)
		print('Saving {}/weights.h5'.format(weightfolder))
		model.save_weights(weightfolder+'/weights.h5')
		open(weightfolder+'/model.json', 'w').write(model.to_json())
		picklefile = open(weightfolder+'/indices.pickle','wb')
		pickle.dump((char_to_index,index_to_char,first_char_probs),picklefile)
		picklefile.close()
		lowest_loss = val_loss
	generated = GenerateTitle(model,MAX_LEN,first_char_probs,index_to_char,char_to_index,num_chars,end_index)
    
	print('Title: {}'.format(generated.title()))
            
  