import numpy as np
import pdb
import random
import time
import sys
from threading import Thread
from queue import Queue
from copy import deepcopy

def GetSamplesPerEpoch(training_data, batch_size):
    usable_total = 0
    total = 0
    for array in training_data:
        usable_total+=(array.shape[0] // batch_size) * batch_size
        total+=len(array)
    print("{} samples total, {} per epoch.".format(total,usable_total))
    return usable_total

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

def EncodeCharVecsToTrainingArray(charvecs,end_index):
    num_timesteps = charvecs.shape[1]
    num_samples = charvecs.shape[0]
    output = np.zeros((num_samples,num_timesteps,end_index+1),dtype=np.bool)
    indices_dim1 = np.repeat(np.arange(num_samples),num_timesteps)
    indices_dim2 = np.tile(np.arange(num_timesteps),num_samples)
    indices_dim3 = np.reshape(charvecs,num_samples*num_timesteps)
    output[indices_dim1,indices_dim2,indices_dim3] = 1
    return output

def EncodeSingleCharVec(charvec,end_index):
    num_timesteps = len(charvec)
    output = np.zeros((num_timesteps,end_index+1),dtype=np.bool)
    output[np.arange(num_timesteps),charvec]=1
    return output.reshape(1,num_timesteps,end_index+1)

def TitleBatchGenerator(training_data,batch_size,char_to_index,index_to_char,end_index,output_queue=False):
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
		#remember a selection of x means training data of length x + 2
		counter = 0
		while len(selections) > 0:
			array_index = selections.pop()
			startval = pointers[array_index]
			endval = startval + batch_size
			batch = training_data[array_index][startval:endval,:]
			counter+=1
			if counter%1000 == 0:
				debugoutput = ''.join([index_to_char[index] for index in list(batch[0,:])])
				print('Debug: {}'.format(debugoutput))
			encoded_batch = EncodeCharVecsToTrainingArray(batch,end_index)
			features = encoded_batch[:,:-1,:]
			labels = encoded_batch[:,-1,:]
			yield (features,labels)
			pointers[array_index] = endval
		#When you fall off the end of the inner while loop the outer loop restarts 
		#and everything is set back up again
		print('\n\nGenerator resetting! This will take a couple of seconds...\n')
        
        
def OnlineTaglineGenerator(training_data,char_to_index,end_index):
    while True:
        random.shuffle(training_data)
        for training_case in training_data:
            movie_input = EncodeSingleCharVec(training_case[0],end_index)
            tagline = EncodeSingleCharVec(training_case[1],end_index)
            tagline_input = tagline[:,:-1,:]
            tagline_label = tagline[:,-1,:]
            yield [[movie_input,tagline_input],tagline_label]
        print('\n\nGenerator resetting! This will take a couple of seconds...\n')
        
def BatchedTaglineGenerator(training_data,batch_size,char_to_index,index_to_char,end_index,left_padding=False):
	while True:
		selections = []
		pointers = [0 for length_group in training_data]
		for i in range(len(training_data)):
			permutation = np.random.permutation(len(training_data[i][0]))
			training_data[i][0] = training_data[i][0][permutation,:]
			training_data[i][1] = training_data[i][1][permutation,:]
			selections.extend([i for x in range(len(training_data[i][0])//batch_size)])
		random.shuffle(selections) # so we take in a random order
		#remember a selection of x means training data of length x + 2
		counter = 0
		while len(selections) > 0:
			array_index = selections.pop()
			startval = pointers[array_index]
			endval = startval + batch_size
			training_titles = training_data[array_index][0][startval:endval,:]
			training_taglines = training_data[array_index][1][startval:endval,:]
			counter+=1
			#if counter%100 == 0:
			#	index_to_char[end_index]='[END]'
			#	printable_title = ''.join([index_to_char[index] for index in training_titles[12,:].tolist()])
			#	printable_tagline = ''.join([index_to_char[index] for index in training_taglines[12,:].tolist()])
			#	print('\nDebug title: {}'.format(printable_title))
			#	print('Debug tagline: {}'.format(printable_tagline))
			#Note: Because both taglines and titles can have differing lengths we may not want
			#to have a separate list for every tagline_length and title_length combo
			
			#An alternative is for titles to be padded to the nearest multiple of 5 characters and truncated at 30
			#characters. This means there will only be 6 title_length bins instead of 30 or more.
			
			#The left-padding is done with an index one greater than the special end_index, 
			#which is normally the highest index. When one-hot encoded, the padding characters will therefore
			#cause 1s at the positions [sample,timestep,end_index+1], where the array has dimensions
			#[num_samples,num_timesteps,end_index+1]
			
			#At this point we can simply truncate the matrix to [:,:,:end_index] to remove the padding characters,
			#resulting in all-zeros input to the RNN at that point.
			if left_padding:
				encoded_titles = EncodeCharVecsToTrainingArray(training_titles,end_index+1)[:,:,:-1]
			else:
				encoded_titles = EncodeCharVecsToTrainingArray(training_titles,end_index)
			encoded_taglines = EncodeCharVecsToTrainingArray(training_taglines,end_index)
			tagline_features = encoded_taglines[:,:-1,:]
			if tagline_features.shape[1]==0:
				tagline_features = np.zeros_like(encoded_taglines)
			tagline_labels = encoded_taglines[:,-1,:]
			yield ([encoded_titles,tagline_features],tagline_labels)
			#yield ([tagline_features,tagline_labels])
			pointers[array_index] = endval
		#When you fall off the end of the inner while loop the outer loop restarts 
		#and everything is set back up again
		print('\n\nGenerator resetting! This will take a couple of seconds...\n')

def BufferedGenerator(training_data,batch_size,char_to_index,index_to_char,end_index,num_jobs=2):
	thread_list = []
	q = Queue(maxsize=256)
	for i in range(num_jobs):
		thread_list.append(Thread(target=TitleBatchGenerator, args=(deepcopy(training_data),batch_size,char_to_index,index_to_char,end_index,q)))
		thread_list[-1].start()
	while True:
		yield q.get()
	