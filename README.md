# IMRNN
Throwing RNNs at the IMDB database

##What is this?
A simple-ish Twitter bot and recurrent neural network (RNN) script that learns to generate new and plausible-sounding titles from the IMDB's list of English-language movies. It tweets at [@AuteurBot](https://twitter.com/AuteurBot).

##Why?
I wanted to play with character-generation RNNs. Also sometimes it's extremely hilarious.

##Why do lots of the movie ideas feel like porn?
There's an awful lot of it in the IMDB database. I considered censoring it, but I felt that would defeat the point of the social experiment.

##What's your favourite movie title from it so far?
"Demons & Boyfriends". I think I might legit be able to sell that idea to someone.

##How was this done? Can I run this myself with different input?
Sure! You'll need [Keras](http://keras.io/#installation) and its dependencies (mostly Theano or Tensorflow) for the network itself, and I used Tweepy to handle Twitter auth.

If you want the technical details, characters are represented by one-hot encodings. All titles are presented to the network in lowercase, which means its outputs do not have any case in them. A helper script fixes the capitalization before they get tweeted.

The network uses two long short-term memory (LSTM) layers with 1024 neurons each, with the final output from the second layer passing through a normal feedforward rectifying linear (ReLu) layer. Up to 12 characters are fed into the network, and then it's trained to predict the next character in the sequence. The output layer is a softmax layer over possible characters.
