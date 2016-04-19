#!/usr/bin/python3

import pdb,sys
from title_extractor import ExtractTitles

#We vectorize titles with a sliding-window strategy
#with a minimum length of 1 character and a maximum length
#of max_len characters

#In all cases, the desired output is the subsequent character,
#or a special END character if the input extended to the
#end of the title. The END character uses its own index (i.e. if the
#characters in the titles use indices from 0 to n, then the END character
#uses index n+1).

def VectorizeTitle(title,char_to_index,end_index,max_len):
    training = []
    labels = []
    title_indices = [char_to_index[char] for char in list(title)]
    title_indices.append(end_index)
    start_char = 0
    target_char = 1
    while target_char < len(title_indices):
        training.append(title_indices[start_char:target_char-1])
        labels.append(title_indices[target_char])
        target_char = target_char+1
        if target_char - start_char > max_len:
            start_char+=1
    return (training,labels)

def MakeTrainingData():
    pass


titles = ExtractTitles(sys.argv[1])
titlestring = ''.join(titles).lower() #We ignore case
titlechars = set(titlestring)

char_to_index = {ch:i for i,ch in enumerate(charset)}
index_to_char = {i:ch for i,ch in enumerate(charset)}




    
    

