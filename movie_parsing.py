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
            chopped_tagline = ChopTitle(tagline,max_len,char_to_index,end_index,min_size=1)
            for chop in chopped_tagline:
                training_cases.append((encoded_title,chop))
    return (training_cases, english_titles)

def MakeBatchedTaglineTraining(max_len,titles_file,taglines_file,allowed_chars,char_to_index,end_index,left_padding=False):
    #For each title, make a training case consisting of
    #(full_title,tagline[:n]), using a rolling window across the tagline as with titles.
	english_titles = GetEnglishTitles(titles_file, allowed_chars)
	
	#Batching solution: Most titles are 30 characters or less. Therefore we can divide into batches as follows:
	#6 Bins of width 5 can be used. For example, all titles 26-30 characters long will be a bin. Titles over 30 characters are truncated to 30.
	#Within each bin, there will be 11 bins for different tagline lengths
	
	#The title bin chosen should be [(title_length-1)//5]. This will put lengths 1-5 in bin 0, 6-10 in bin 1 and so on.
	#The tagline chop bin should be [chop_length-1] as 1 is the minimum chop length.
	taglines_dict = ExtractTaglines(taglines_file,allowed_chars,english_titles)
	if left_padding:
		training_cases = [[[] for i in range(max_len)] for j in range(6)]
	else:
		training_cases = [[[] for i in range(max_len)] for j in range(30)]
	for title,taglines in taglines_dict.items():
		encoded_title = [char_to_index[char] for char in title][:30] #truncate to 30 chars if longer than that
		#if left_padding and not len(encoded_title)%5 == 0: #if we need padding, in other words
		#	title_left_padding = 5-(len(encoded_title)%5)
			#We can't pad with 0, that's a character index! Let's pad with end_index + 1 
			#and then strip those out in the generator.
		#	encoded_title = [end_index+1 for x in range(title_left_padding)] + encoded_title
		#	title_bin = (len(encoded_title)//5) - 1
		title_bin = len(encoded_title) - 1
		for tagline in taglines:
			chopped_tagline = ChopTitle(tagline,max_len,char_to_index,end_index,min_size=1)
			for chop in chopped_tagline:
				tagline_bin = len(chop)-1
				training_cases[title_bin][tagline_bin].append([encoded_title,chop])
    #This seems a little perverse, but now I'm going to flatten that out a bit
    #We could have built it directly as a flattened list but I feel the code would have lost
    #significant clarity. I might change my mind about that and refactor this though.
	training_output = []
	for title_length_bin in training_cases:
		for tagline_chop_length_bin in title_length_bin:
			title_training = []
			tagline_training = []
			for case in tagline_chop_length_bin:
				title_training.append(case[0])
				tagline_training.append(case[1]) # I tried doing this with zip but it didn't work. I have no idea why.
			title_training = np.array(title_training,dtype=np.uint8)
			tagline_training = np.array(tagline_training,dtype=np.uint8)
			training_output.append([title_training,tagline_training])

	return (training_output, english_titles)

                
def ChopTitle(title,max_len,char_to_index,end_index,min_size):
	#Returns all training cases that can be produced from a single title
    training = []
    title_indices = [char_to_index[char] for char in title]
    title_indices.append(end_index)
    start_char = 0
    target_char = min_size
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
        chopped_title = ChopTitle(title,max_len,char_to_index,end_index,min_size=2)
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