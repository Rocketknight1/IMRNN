#/usr/bin/python3

import tweepy
import subprocess
import pdb
import os

#This script needs a consumer key, consumer secret, access key and access secret from Twitter.
#Since this isn't really mission-critical high-security stuff it just reads them from a text file
#with one key per line, in the order they're listed above.

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


with open('tokens.txt') as f:
	tokens = f.read().rstrip().split('\n')
	consumer_key, consumer_secret, access_key, access_secret = tokens

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)
title = PopTitle('titles.txt').lstrip().rstrip()
api.update_status('Movie idea: {}'.format(title))