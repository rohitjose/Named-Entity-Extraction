import sys
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

def check_word_capitalization(word):
    """Checks whether a word is a capitalized word"""
    return_value = False
    if (len(word) > 1):
        return_value = True if (word[0].isupper() and word[1].islower()) else False

def build_training_features(training_data):
    # Load the data set
    with open(str(training_data), 'rb') as f:
        training_set = pickle.load(f)

    columns_values = ["word","POStag","TitleMapping","isUpper", "isCapitalized", "isNNP", "isNN", "isJJ","isPrecedingDT","isPrecedingIN","isTitle"]
    preceding_pos_tag = None

    # to lematize a word
    lmtzr = WordNetLemmatizer()

    features_and_labels =[]
    for sentence in training_set:
        for word in sentence:
            token_dict = {}
            # Map the word into the dictionary
            token_dict[lmtzr.lemmatize(word[0])] = 1
            # WORD TEXT RELATED FEATURES
            # Check for a word in uppercase
            token_dict["isUpper"] = (1 if (word[0].isupper()) else 0)
            # Check for word capitalization
            token_dict["isCapitalized"] = (1 if (check_word_capitalization(word[0])) else 0)

            # WORD POS TAGGING RELATED FEATURES
            # Check if the POS tagging is NNP - (noun, proper, singular)
            token_dict["isNNP"] = (1 if (word[1] == 'NNP') else 0)
            # Check if the POS tagging is NN - (noun, common, singular or mass)
            token_dict["isNN"] = (1 if (word[1] == 'NN') else 0)
            # Check if the POS tagging is JJ - (adjective or numeral, ordinal)
            token_dict["isJJ"] = (1 if (word[1] == 'JJ') else 0)

            # POS TAGGING OF PRECEDING WORD
            # Check if the preceding POS tagging was DT
            token_dict["isPrecedingDT"] = (1 if (preceding_pos_tag == 'DT') else 0)
            # Check if the preceding POS tagging was IN (preposition or conjunction, subordinating)
            token_dict["isPrecedingIN"] = (1 if (preceding_pos_tag == 'IN') else 0)

            # Map the POS tag of the preceding token
            preceding_pos_tag = word[1]

            # Append values to features and labels
            features_and_labels.append((token_dict, word[2]))
    # return the features and labels list
    return features_and_labels

if __name__=="__main__":
    training_data = sys.argv[1]
    features_and_labels = build_training_features(training_data)

    print(features_and_labels[0])

    encoder = LabelEncoder()
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    X, y = list(zip(*features_and_labels))
    X = vectorizer.fit_transform(X)
    y = encoder.fit_transform(y)

    print("done transformation")

    arr = X[10].toarray()
    for i in range(len(arr[0])):
        if arr[0][i] > 0:
            print('{}:{}'.format(vectorizer.feature_names_[i], arr[0][i]))

    print("Only the words")
    nb = LogisticRegression()
    nb.fit(X, y)
    print(nb.score(X, y))