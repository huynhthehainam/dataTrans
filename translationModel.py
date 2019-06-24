import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import io
import string
import re
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()


def preprocess(sentence):
    exclude = []
    sent = sentence.lower()
    sent = sent.strip()
    sent = re.sub("'", " ", sent)
    sent = re.sub("\s+", " ", sent)
    sent = ''.join([char for char in sent if char not in exclude])
    sent = "<start> " + sent + " <end>"
    return sent

print(string.digits)
