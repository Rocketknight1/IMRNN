#!/usr/bin/python3

import pdb,sys
import numpy as np
import pickle
from collections import Counter

#We vectorize titles with a sliding-window strategy
#with a minimum length of 1 character and a maximum length
#of max_len characters

#In all cases, the desired output is the subsequent character,
#or a special END character if the input extended to the
#end of the title. The END character uses its own index (i.e. if the
#characters in the titles use indices from 0 to n, then the END character
#uses index n+1).

#We generate a 4-dimensional array as output
#1st dimension is timestep length - 2
#2nd dimension is samples
#3rd dimension is timesteps
#4th dimension is characters

def ExtractTitles(file,allowedchars):
	title_list = []
	rejectlist = []
	f = open(file,encoding='latin_1')
	for line in f:
		if line[0]=='#':
			try:
				title = line[2:line.index('(')]
				title = title.rstrip('" ').lstrip('''#" ''').lower().rstrip()
			except:
				pdb.set_trace()
			if title and all([char in allowedchars for char in list(title)]):
				title_list.append(title)
			else:
				rejectlist.append(title) #this is just for debugging
	return title_list

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
	output = np.zeros((len(vector),end_index+1),dtype=np.bool)
	for i in range(len(vector)):
		output[i,vector[i]]=1
	return output
	
    
def ListToNumpy(sample_list,end_index):
    #Take a list of vectorized titles and return a 3D numpy array suitable for training from
    num_timesteps = sample_list[0].shape[0]
    assert all([sample.shape[0]==num_timesteps for sample in sample_list])
    output = np.zeros((len(sample_list),num_timesteps,end_index+1),dtype=np.bool)
    for i in range(len(sample_list)):
        output[i,:,:] = sample_list[i]
    return output
	
def MakeTrainingData(titles,char_to_index,end_index,max_len):
    training_cases = [[] for i in range(max_len-1)]
    for title in titles:
        title_vectors = VectorizeTitle(title,char_to_index,end_index,max_len)
        for vector in title_vectors:
            training_cases[len(vector)-2].append(OneHot(vector,end_index))
    for i in range(len(training_cases)):
        training_cases[i] = ListToNumpy(training_cases[i],end_index)
    return training_cases

def MakeFirstCharDistribution(titles,index_to_char):
    firstletters = [title[0] for title in titles]
    total_titles = len(titles)
    charcounts = dict(Counter(firstletters))
    charprobs = {key : val/total_titles for key,val in charcounts.items()}
    probarray = []
    for i in range(max(index_to_char.keys())+1):
        if index_to_char[i] in charprobs:
            probarray.append(charprobs[index_to_char[i]])
        else:
            probarray.append(0)
    return np.array(probarray,dtype=np.float64)


def EncodeTitles(file,max_len):
	#oh god here comes the inelegant part
	allowedchars = set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','.',';',':','!','?','-','&',',',' ','"','\''])
	
	char_to_index = {ch:i for i,ch in enumerate(allowedchars)}
	index_to_char = {i:ch for i,ch in enumerate(allowedchars)}
	
	end_index = max(index_to_char.keys())+1
	
	titles = ExtractTitles(file,allowedchars) #Note case is ignored
	first_char_probs = MakeFirstCharDistribution(titles, index_to_char)
    
	training = MakeTrainingData(titles,char_to_index,end_index,max_len)
	
	return (training,char_to_index,index_to_char,end_index,first_char_probs)
