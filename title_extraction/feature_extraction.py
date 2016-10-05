import pickle
import sys
import pandas
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn import metrics


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
    word_frame = pandas.DataFrame(word_list,columns=columns_values)
    # Remove the first 3 columns [word, POS tag, target]
    data_frame = word_frame[columns_values[3:]]


    return data_frame

def trainRegressionModel(X,y):
    """Trains the logistic regression model based on the extracted feature values given"""
    # # instantiate a logistic regression model, and fit with X and y
    # model = LogisticRegression()
    # model = model.fit(X, y)
    # # check the accuracy on the training set
    # print(model.score(X, y))
    #X['intercept'] = 1.0
    #del X['isCapitalized']
    #del X['isNN']
    #del X['isNNP']
    #del X['isJJ']
    #del X['isUpper']
    #del X['isPrecedingIN']
    logit = sm.Logit(y, X)
    result = logit.fit()
    print(result.summary())
    print(result.conf_int())
    model = LogisticRegression()
    model = model.fit(X, y)
    print(model.score(X, y))
    print(y.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)
    # predict class labels for the test set
    predicted = model.predict(X_test)
    print(predicted)
    for i in predicted:
        if i==1:
            print("Test:"+str(i))
    print(max(predicted))
    #generate class probabilities
    probs = model2.predict_proba(X_test)
    print(probs)
    # generate evaluation metrics
    print("Accuracy: "+str(metrics.accuracy_score(y_test, predicted)))
    print("AUC: "+str(metrics.roc_auc_score(y_test, probs[:, 1])))
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.classification_report(y_test, predicted))

    from sklearn.cross_validation import cross_val_score
    # evaluate the model using 10-fold cross-validation
    scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print(scores)
    print(scores.mean())



if __name__ == "__main__":
    # Get the training data set path
    training_data = sys.argv[1]
    columns_values = ["isUpper", "isCapitalized", "isNNP", "isNN", "isJJ","isPrecedingDT", "isPrecedingIN", "isTitle"]
    data = build_training_features(training_data)
    X = data[columns_values[:-1]]
    y = data["isTitle"]
    print(data.groupby("isTitle").mean())
    trainRegressionModel(X,y)

