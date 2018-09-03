from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle


def Load_Clean_sentences(filename):
	return load(open(filename,'rb'))

def save_clean_data(sentences,filename):
	dump(sentences,open(filename,'wb'))
	print('Saved : %s' %filename)

raw_dataset = Load_Clean_sentences('english-german.pkl')


#Demo settings for 10k Instances (Need a better model to enhance it's performance)
n_sentences = 10000
dataset = raw_dataset[:n_sentences,:]
#Random shuffling
shuffle(dataset)
#SPliting into train and testing set
train,test = dataset[:9000],dataset[:1000]
#Saving in pickle
save_clean_data(dataset,'english-german-both.pkl')
save_clean_data(train,'english-german-train.pkl')
save_clean_data(test,'english-german-test.pkl')

