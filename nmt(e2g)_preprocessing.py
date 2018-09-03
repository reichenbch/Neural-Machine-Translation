import re
import string
from pickle import dump
from unicodedata import normalize
from numpy import array


#Loading Data into memory as blob of text to perserve Unicode German Characters
def load_doc(filename):
	file = open(filename,mode='rt',encoding='utf-8')
	text = file.read()
	file.close()

	return text

#Spliting loaded Document into Sentences(English and german Seperately)
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in lines]
	return pairs

#Data Cleaning Operations
def clean_pairs(lines):
	cleaned = list()

	#preparing regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	#Prepare translation table for removing punctuation
	table = str.maketrans('','',string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			#Normalize unicode characters
			line = normalize('NFD',line).encode('ascii','ignore')
			line = line.decode('UTF-8')
			#Tokenizing on White Space
			line = line.split()

			line = [word.lower() for word in line]
			#remove punctuation from each token
			line = [word.translate(table) for word in line]
			#removing non-printable characters from each token
			line = [re_print.sub('',w) for w in line]
			#remove tokens with number in them
			line = [word for word in line if word.isalpha()]
			#Store as String
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

#Saving the clean sentences to a file
def save_clean_data(sentences,filename):
	dump(sentences,open(filename,'wb'))
	print('Saved: %s' % filename)

filename = 'deu.txt'
doc = load_doc(filename)
pairs = to_pairs(doc)
clean_pairs = clean_pairs(pairs)
save_clean_data(clean_pairs,'english-german.pkl')

for i in range(100):
	print('[%s] ==> [%s]' % (clean_pairs[i,0],clean_pairs[i,1]))
