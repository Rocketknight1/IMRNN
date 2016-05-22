#!/usr/bin/python3

import tweepy
import subprocess
import pdb
import os
import sys

#This ensures that the working directory is set to the script's directory
#saving me the very minor hassle of dealing with absolute paths when this is run
#by cron
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def PopTitle(filename):
	#With credit to Saqib on Stackoverflow for this efficient way to pop lines from a file
	#without rewriting the whole thing
	with open(filename, "r+") as file:
		file.seek(0, os.SEEK_END)
		pos = file.tell() - 1
		while pos > 0 and file.read(1) != "\n":
			pos -= 1
			file.seek(pos, os.SEEK_SET)
		if pos > 0:
			file.seek(pos, os.SEEK_SET)
			output = file.read()
			file.seek(pos, os.SEEK_SET)
			file.truncate()
	return output

#This script needs a consumer key, consumer secret, access key and access secret from Twitter.
#Since this isn't really mission-critical high-security stuff it just reads them from a text file
#with one key per line, in the order they're listed above.

MAX_LEN=12
TITLE_FILE = 'titles.txt'
TOKENS_FILE = 'tokens.txt'


with open(TOKENS_FILE) as f:
	tokens = f.read().rstrip().split('\n')
	consumer_key, consumer_secret, access_key, access_secret = tokens

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)
try:
	title = PopTitle(TITLE_FILE).lstrip().rstrip()
	assert(len(title) > 0)
except:
	raise
	
api.update_status('Movie idea: {}'.format(title))