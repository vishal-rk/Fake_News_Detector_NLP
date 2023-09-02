 
 # src/preprocessing.py

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

def preprocess_text(text):
    ps = PorterStemmer()
    ele = re.sub('[^a-zA-Z]', ' ', text)
    ele = ele.lower()
    ele = ele.split()
    ele = [ps.stem(word) for word in ele if not word in stopwords.words('english')]
    ele = ' '.join(ele)
    return ele

def preprocess_and_pad(texts, voc_size, max_len):
    corpus = []
    ps = PorterStemmer()

    for text in texts:
        ele = re.sub('[^a-zA-Z]', ' ', text)
        ele = ele.lower()
        ele = ele.split()
        ele = [ps.stem(word) for word in ele if not word in stopwords.words('english')]
        ele = ' '.join(ele)
        corpus.append(ele)

    onehot_rep = [one_hot(word, voc_size) for word in corpus]
    padded_text = pad_sequences(onehot_rep, padding='post', maxlen=max_len)
    return padded_text