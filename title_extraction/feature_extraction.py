import pickle
import sys
import nltk
import pandas

def build_training_features(training_data):
    # Load the data set
    with open(str(training_data), 'rb') as f:
        training_set = pickle.load(f)

    print(training_set[0])
    columns = ["isUpper","isCapitalized"]
    for sentence in training_set:
        for word in sentence:
            # Check for Uppercased word
            if (word[0].isupper()):
                word.append(1)
            else:
                word.append(0)

            # Check for capitalization
            if(word[0][0].isupper()):
                word.append(1)
            else:
                word.append(0)

    frame = [item for sublist in training_set for item in sublist]
    for word in frame:
        if(word[-1]==1):
            print(word)

if __name__=="__main__":
    # Get the training data set path
    training_data = sys.argv[1]

    build_training_features(training_data)



