import pickle
import sys
import pandas
from sklearn.linear_model import LogisticRegression


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
            word.append(1 if (word[0].isupper()) else 0)
            # Check for word capitalization
            word.append(1 if (check_word_capitalization(word[0])) else 0)

            # WORD POS TAGGING RELATED FEATURES
            # Check if the POS tagging is NNP - (noun, proper, singular)
            word.append(1 if (word[1] == 'NNP') else 0)
            # Check if the POS tagging is NN - (noun, common, singular or mass)
            word.append(1 if (word[1] == 'NN') else 0)
            # Check if the POS tagging is JJ - (adjective or numeral, ordinal)
            word.append(1 if (word[1] == 'JJ') else 0)

            # POS TAGGING OF PRECEDING WORD
            # Check if the preceding POS tagging was DT
            word.append(1 if (preceding_pos_tag == 'DT') else 0)
            # Check if the preceding POS tagging was IN (preposition or conjunction, subordinating)
            word.append(1 if (preceding_pos_tag == 'IN') else 0)

            # TARGET COLUMN
            word.append(1 if(word[2]=='TITLE') else 0)
            preceding_pos_tag = word[1]

    # Flat list of all the words
    word_list = [item for sublist in training_set for item in sublist]
    # Pandas Data frame
    word_frame = pandas.DataFrame(word_list,columns=columns_values)
    # Remove the first 3 columns [word, POS tag, target]
    data_frame = word_frame[columns_values[3:]]

    return data_frame

def trainRegressionModel(X,y):
    """Trains the logistic regression model based on the extracted feature values given"""
    # instantiate a logistic regression model, and fit with X and y
    model = LogisticRegression()
    model = model.fit(X, y)
    # check the accuracy on the training set
    print(model.score(X, y))


if __name__ == "__main__":
    # Get the training data set path
    training_data = sys.argv[1]
    columns_values = ["isUpper", "isCapitalized", "isNNP", "isNN", "isJJ","isPrecedingDT", "isPrecedingIN", "isTitle"]
    data = build_training_features(training_data)
    X = data[columns_values[:-1]]
    y = data["isTitle"]
    trainRegressionModel(X,y)

