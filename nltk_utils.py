import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    

def stem(word):
    return stemmer.stem(word.lower())



def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "Hello", "I", "you", "bye", "thank", "cool"]
    bag = [0, 1, 0, 1, 0, 0, 0]
    """

    ps = PorterStemmer()
    tokenized_sentence = [ps.stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:  # Fixed variable name from 'tokenize' to 'tokenized_sentence'
            bag[idx] = 1.0
    return bag

# sentence = ["hello", "hey", "hi", "good day", "greetings", "what's up?", "how is it going"]
# words = ["hi", "hello"]
# bag = bag_of_words(sentence, words)
# print(bag)
