from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


with open('./data/train.en','r',encoding="utf8") as f:
    rawData = f.read()
    X_train = rawData.split('\n')
print(X_train)