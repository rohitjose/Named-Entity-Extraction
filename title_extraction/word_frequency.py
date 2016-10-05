import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer


def check_word_capitalization(word):
    """Checks whether a word is a capitalized word"""
    return_value = False
    if (len(word) > 1):
        return_value = True if (word[0].isupper() and word[1].islower()) else False
    return return_value

def build_training_features(training_data):
    # Load the data set
    with open(str(training_data), 'rb') as f:
        training_set = pickle.load(f)

    columns_values = ["word","POStag","TitleMapping","isUpper", "isCapitalized", "isNNP", "isNN", "isJJ","isPrecedingDT","isPrecedingIN","isTitle"]
    preceding_pos_tag = None
    for sentence in training_set:
        for word in sentence:
            # WORD TEXT RELATED FEATURES
            # Check for a word in uppercase
            word.append(1.0 if (word[0].isupper()) else 0.0)
            # Check for word capitalization
            word.append(1.0 if (check_word_capitalization(word[0])) else 0.0)

            # WORD POS TAGGING RELATED FEATURES
            # Check if the POS tagging is NNP - (noun, proper, singular)
            word.append(1.0 if (word[1] == 'NNP') else 0.0)
            # Check if the POS tagging is NN - (noun, common, singular or mass)
            word.append(1.0 if (word[1] == 'NN') else 0.0)
            # Check if the POS tagging is JJ - (adjective or numeral, ordinal)
            word.append(1.0 if (word[1] == 'JJ') else 0.0)

            # POS TAGGING OF PRECEDING WORD
            # Check if the preceding POS tagging was DT
            word.append(1.0 if (preceding_pos_tag == 'DT') else 0.0)
            # Check if the preceding POS tagging was IN (preposition or conjunction, subordinating)
            word.append(1.0 if (preceding_pos_tag == 'IN') else 0.0)

            # TARGET COLUMN
            word.append(1 if(word[2]=='TITLE') else 0)
            preceding_pos_tag = word[1]

    # Flat list of all the words
    word_list = [item for sublist in training_set for item in sublist]
    # Pandas Data frame
    word_frame = pd.DataFrame(word_list,columns=columns_values)
    # Remove the first 3 columns [word, POS tag, target]
    data_frame = word_frame[columns_values[3:]]


    return data_frame

def get_freq_of_tokens(word):
    tokens = {}
    tokens[word] = 1
    return tokens

training_data = sys.argv[1]
# Load the data set
with open(str(training_data), 'rb') as f:
    training_set = pickle.load(f)



word_list = [item for sublist in training_set for item in sublist]

features_and_labels = []
for token in word_list:
    features_and_labels.append((get_freq_of_tokens(token[0]), token[-1]))


encoder = LabelEncoder()
vectorizer = DictVectorizer(dtype=float, sparse=True)
X, y = list(zip(*features_and_labels))
X = vectorizer.fit_transform(X)
y = encoder.fit_transform(y)
print(type(X))
print(X[0])
print(X[0][0])

print("Only the words")
nb = LogisticRegression()
nb.fit(X, y)
print(nb.score(X, y))

# print("All the features")
# data = build_training_features(training_data)
# combined_features = [X,data]
# all_features = pd.concat(combined_features)
# print(all_features[0])
# nb = LogisticRegression()
# nb.fit(all_features, y)
# print(nb.score(all_features, y))

