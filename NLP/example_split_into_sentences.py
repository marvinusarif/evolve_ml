
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into sentences
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
sentences = sent_tokenize(text)
print(sentences[0])