#!/usr/bin/python3

import pdb,sys
import numpy as np
import pickle
from collections import Counter
import re
from scipy import sparse

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
#4th dimension is 

def GetEnglishTitles(file,allowedchars):
	#This gets a list of English language titles to be checked against later
	f = open(file,encoding="latin-1")
	titles = set()
	total = 0
	notenglish = 0
	nomatch = 0
	disallowedchars = 0
	for line in f:
		total+=1
		if not 'English' in line:
			notenglish+=1
			continue
		titlesearch = re.search('(.*)\s\((\d{4}|\?\?\?\?).*?\)',line)
		if not titlesearch is None:
			title = titlesearch.group(1).lower().rstrip('" ').lstrip('" ')
			if all([char in allowedchars for char in list(title)]) and title:
				titles.add(title)
			else:
				disallowedchars+=1
		else:
			nomatch+=1
	print("\nLanguage filter report:")
	print("{} lines total.".format(total))
	print("{} not English.".format(notenglish))
	print("{} failed RE filter (this shouldn't happen)".format(nomatch))
	print("{} had disallowed chars.".format(disallowedchars))
	print("{} titles after merging and filtering.\n".format(len(titles)))
	return titles

def ExtractTaglines(file,allowedchars,english_titles):
    taglines = dict()
    taglinecount = 0
    entrycount = 0
    rejectcount = 0
    rejected_taglines = 0
    f = open(file,encoding="latin-1")
    for line in f:
        if line[0]=='#':
            entrycount+=1
            title = line[2:line.index('(')]
            title = title.rstrip('" ').lstrip('''#" ''').lower().rstrip()
            if title and all([char in allowedchars for char in list(title)]) and title in english_titles:
                taglines[title]=[]
            else:
                rejectcount+=1
                title = False
        elif line[0]=='\t' and title:
            tagline = line.lstrip().rstrip().lstrip('" ').rstrip('" ').lower()
            if all([char in allowedchars for char in list(tagline)]):
                taglines[title].append(tagline)
                taglinecount+=1
            else:
                rejected_taglines+=1
    print("Tagline_extractor: Got {} films with {} taglines".format(len(list(taglines.keys())),taglinecount))
    print("Tagline_extractor: Rejected {} films and {} taglines.".format(rejectcount,rejected_taglines))
    return taglines

def MakeTaglineTraining(max_len,titles_file,taglines_file,allowed_chars,char_to_index,end_index):
    #For each title, make a training case consisting of
    #(full_title,tagline[:n])
    english_titles = GetEnglishTitles(titles_file, allowed_chars)
    
    #Initially let's try online learning to avoid the batching problem here
    taglines_dict = ExtractTaglines(taglines_file,allowed_chars,english_titles)
    training_cases = []
    for title,taglines in taglines_dict.items():
        encoded_title = [char_to_index[char] for char in title]
        for tagline in taglines:
            chopped_tagline = ChopTitle(tagline,max_len,char_to_index,end_index)
            for chop in chopped_tagline:
                training_cases.append((encoded_title,chop))
    return (training_cases, english_titles)
                
def ChopTitle(title,max_len,char_to_index,end_index):
	#Returns all training cases that can be produced from a single title
    training = []
    title_indices = [char_to_index[char] for char in title]
    title_indices.append(end_index)
    start_char = 0
    target_char = 2
    while target_char <= len(title_indices):
        training.append(title_indices[start_char:target_char])
        target_char+=1
        if target_char - start_char > max_len:
            start_char+=1
    return training
	
def MakeTitleTraining(max_len,titles_file,allowed_chars,char_to_index,end_index):
    titles = GetEnglishTitles(titles_file,allowed_chars)
    training_cases_by_length = [[] for i in range(max_len-1)]
    for title in titles:
        chopped_title = ChopTitle(title,max_len,char_to_index,end_index)
        for chop in chopped_title:
            training_cases_by_length[len(chop)-2].append(chop)
    training_cases_by_length = [np.array(training_cases,dtype=np.uint8) for training_cases in training_cases_by_length]
    return training_cases_by_length, titles

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