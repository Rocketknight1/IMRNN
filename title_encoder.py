#!/usr/bin/python3

import pdb,sys
from title_extractor import ExtractTitles
import numpy as np

#We vectorize titles with a sliding-window strategy
#with a minimum length of 1 character and a maximum length
#of max_len characters

#In all cases, the desired output is the subsequent character,
#or a special END character if the input extended to the
#end of the title. The END character uses its own index (i.e. if the
#characters in the titles use indices from 0 to n, then the END character
#uses index n+1).

def VectorizeTitle(title,char_to_index,end_index,max_len):
	#Returns all training vectors that can be produced from a single title
	#In all cases, the label is the final column
    training = []
    title_indices = [char_to_index[char] for char in list(title)]
    title_indices.append(end_index)
    start_char = 0
    target_char = 2
    while target_char <= len(title_indices):
        training.append(title_indices[start_char:target_char])
        target_char+=1
        if target_char - start_char > max_len:
            start_char+=1
    return training

def OneHot(vector,end_index):
	#returns a one-hot encoding of a characters vector with dimensions (num_timesteps,size_of_charset)
	output = np.zeros((len(vector),end_index+1),dtype=np.uint8)
	for i in range(len(vector)):
		output[i,vector[i]]=1
	return output
	
def ListToNumpy(sample_list,num_timesteps,end_index):
    #Take a list of vectorized titles and return a 3D numpy array suitable for training from
	output = np.zeros((len(sample_list),num_timesteps,end_index+1),dtype=np.uint8)
	for i in range(len(sample_list)):
		output[i,:,:] = sample_list[i]
	return output
	
def MakeTrainingData(titles,char_to_index,end_index,max_len):
    training_cases_by_length = dict()
    for length in range(2,max_len+1):
        training_cases_by_length[str(length)]=[]
    for title in titles:
        title_vectors = VectorizeTitle(title,char_to_index,end_index,max_len)
        for vector in title_vectors:
            training_cases_by_length[str(len(vector))].append(OneHot(vector,end_index)) 
    for length in range(2,max_len+1):
        training_cases_by_length[str(length)] = ListToNumpy(training_cases_by_length[str(length)],length,end_index)
    return training_cases_by_length
		
MAX_LEN = 12

titles = ExtractTitles(sys.argv[1]) #Note case is ignored
titlestring = ''.join(titles)
titlechars = set(titlestring)

char_to_index = {ch:i for i,ch in enumerate(titlechars)}
index_to_char = {i:ch for i,ch in enumerate(titlechars)}

end_index = max(index_to_char.keys())+1

training = MakeTrainingData(titles,char_to_index,end_index,MAX_LEN)

np.savez_compressed('training.npz',**training)




    
    

