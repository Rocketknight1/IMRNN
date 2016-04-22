import numpy as np
import pdb
import random
import time
import sys

def GetSamplesPerEpoch(training_data, batch_size):
    usable_total = 0
    total = 0
    for array in training_data:
        usable_total+=(array.shape[0] // batch_size) * batch_size
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
        #remember a selection of x means training data of length x + 2
        while len(selections) > 0:
            array_index = selections.pop()
            startval = pointers[array_index]
            endval = startval + batch_size
            batch = training_data[array_index][startval:endval,:]
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