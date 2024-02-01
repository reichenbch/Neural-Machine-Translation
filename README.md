# Neural-Machine-Translation
Implementing neural machine translation from scratch using Python,Keras.Here,we will building a model to translate German phrases to English !

## Dataset

The dataset is available from the http://www.manythings.org/ website, with examples drawn from the Tatoeba Project. The dataset is comprised of German phrases and their English counterparts and is intended to be used with the Anki flashcard software.

Dataset for the system can be downloaded here : http://www.manythings.org/anki/deu-eng.zip

Download the dataset to your current working directory and decompress; for example:
unzip deu-eng.zip

You will have a file called deu.txt that contains 152,820 pairs of English to German phases, one pair per line with a tab separating the language.

## Sequence of files to run :
1. nmt_training_split.py
2. nmt(e2g)_preprocessing.py
3. nmt(e2g)_translation.py
4. nmt(e2g)_evaluation.py

## Things to be done more for building a better system :

Data Cleaning - Different data cleaning operations could be performed on the data, such as not removing punctuation or normalizing case, or perhaps removing duplicate English phrases.

Vocabulary - The vocabulary could be refined, perhaps removing words used less than 5 or 10 times in the dataset and replaced with “unk“.

More Data - The dataset used to fit the model could be expanded to 50,000, 100,000 phrases, or more.

Input Order - The order of input phrases could be reversed, which has been reported to lift skill, or a Bidirectional input layer could be used.

Layers - The encoder and/or the decoder models could be expanded with additional layers and trained for more epochs, providing more representational capacity for the model.

Units -  The number of memory units in the encoder and decoder could be increased, providing more representational capacity for the model.

Regularization - The model could use regularization, such as weight or activation regularization, or the use of dropout on the LSTM layers.

Pre-trained Word Vectors - Pre-trained word vectors could be used in the model.

Recursive Model - A recursive formulation of the model could be used where the next word in the output sequence could be conditional on the input sequence and the output sequence generated so far.

