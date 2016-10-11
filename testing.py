import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from extract_features import build_training_features

if __name__ == "__main__":
    path_to_testing_data = sys.argv[1]
    path_to_classifier = sys.argv[2]
    path_to_results = sys.argv[3]

    # Testing
    test_data = build_training_features(path_to_testing_data)
    encoder = LabelEncoder()
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    X_test, y_test = list(zip(*test_data))
    X_test = vectorizer.fit_transform(X_test)
    y_test = encoder.fit_transform(y_test)


    # Import the logistic regression model
    with open(str(path_to_classifier), 'rb') as f:
        logitModel = pickle.load(f)
    predicted = logitModel.predict(X_test)
    # print(predicted)
    probs = logitModel.predict_proba(X_test)

    # for item in predicted:
    #     if item != 0:
    #         print(item)
    #
    # from sklearn import metrics
    #
    # print("Accuracy")
    # print(metrics.accuracy_score(y_test, predicted))
    # print(metrics.roc_auc_score(y_test, probs[:, 1]))

    with open(str(path_to_testing_data), 'rb') as f:
        training_set = pickle.load(f)



    results = []
    counter = 0
    for sentence in training_set:
        result_sentence = []
        for words in sentence:
            result_sentence.append((words[0],'TITLE' if(predicted[counter]==1)else 'O'))
            counter+=1
        results.append(result_sentence)

    print(results)

    # Dump the results
    with open(path_to_results, 'wb') as f:
        pickle.dump(results, f)
