import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from extract_features import build_training_features



if __name__ == "__main__":
    training_data = sys.argv[1]
    classifier_path = sys.argv[2]
    features_and_labels = build_training_features(training_data)


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
    logitModel = LogisticRegression()
    logitModel.fit(X, y)
    print(logitModel.score(X, y))

    with open(classifier_path, 'wb') as f:
        pickle.dump(logitModel, f)
