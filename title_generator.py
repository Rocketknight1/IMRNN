#!/usr/bin/python3

from movie_parsing import MakeTitleTraining,MakeFirstCharDistribution

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import PReLU, ELU, SReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from training_functions import GetSamplesPerEpoch,ChooseCharacter,TitleBatchGenerator,GenerateTitle,MakeTrainingAndValidationSets
from keras.optimizers import Adam
import pickle
import sys

def TitleCase(generated):
	words = generated.split(' ')
	for i in range(len(words)):
		if words[i] in ('ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii','xiii','xiv','xv','uk','usa','u.s.','u.s.a.','u.k.','ii:','iii:','iv:','v:','vi:','vii:','viii:','ix:','x:','xi:','xii:','xiii:','xiv:','xv:'):
			words[i]=words[i].upper()
		else:
			letters = list(words[i])
			if letters:
				letters[0]=letters[0].upper()
				words[i]=''.join(letters)
	return ' '.join(words)

target_folder = sys.argv[1]
if not target_folder[-1]=='/':
	target_folder = target_folder+'/'
	
MAX_LEN = 12
#oh god here comes the inelegant part
#no-one saw that okay

char_to_index, index_to_char, first_char_probs = pickle.load(open(target_folder+'indices.pickle','rb'))
end_index = max(index_to_char.keys())+1
num_chars = end_index + 1

model = model_from_json(open(target_folder+'model.json').read())
model.load_weights(target_folder+'weights.h5')

f = open('titles.txt','w')

while True:
	generated = GenerateTitle(model,MAX_LEN,first_char_probs,index_to_char,char_to_index,num_chars,end_index)
	generated = TitleCase(generated)
	print('Title: {}'.format(generated))
	f.write(generated)
	f.write('\n')