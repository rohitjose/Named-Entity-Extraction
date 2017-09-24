# Named Entity Extraction



## Approach

The input file provided for training contains a word-tokenized list of the sentences. Each token or word has the word content, POS (part-of-speech) tag and indicates if the word is a title or not. The properties that influence the word being 'Title' entity relates more to the placement of the word in the sentence, in relation to a named entity (PERSON/ORGANIZATION), that it tries to describe. A 'Title' is usually a prefix or suffix added to someone's name in certain contexts. It may signify either veneration, an official position or a professional or academic qualification.

The features selected are based on the following properties of each token in the sentence:

1. The distance of the word from a named entity present in the sentence
  (difference of the indices of the word and the named entity)
2. The POS tagging of the token.
3. The content of the token such as the capitalization or case (upper/lower).
4. The POS tagging of the preceding token in the sentence sequence.
5. A lookup for the token content against a lemmatized reference list for
  known titles (Gazetteer method).

## Feature extraction and engineering
### Maximum Distance from a Named Entity
The relative distance of the token index, from the indices of identified Named Entities is taken as a feature. The named entities in the sentence are identified with the _nltk.ne_chunk_ module. This module parses the POS tagged list to identify named entities. The indices of these named entities are identified and their difference against the index of each token is taken. The maximum value of this difference is assigned as a feature value for the token.
```
maxRelativeDistanceNE = max(d_i)
d_i = absolute([index_token − index_namedEntity_i])
```
### POS Tagging of the Token
Upon examination of the POS tags of the tokens in the training data, the 'Title' named entity occurs in a particular sequence of POS tags.

| Pattern      | Count | Example           |
| ------------ | ----- | ----------------- |
| `NNP`        | 517   | MP                |
| `NN`         | 319   | president         |
| `<NNP><NNP>` | 315   | Vice President    |
| `<JJ><NN>`   | 105   | Regional Director |

This feature is evaluated by parsing the POS tagged token sequence through a regular expression (utilizing the _nltk.RegexpParser_ Module). The regular expression defined for this is given below:
```
title_grammar = r """
                TITLE :
                { < NNP >+ < NN > < NN >+}
                { < NNP > < NNP >+}
                { < NN >? < JJ > < NN | NNP >+}
                { < NNP > < IN > < NNP >}"""
```
For each token we determine a value of "1" if the token was identified as a title after parsing (else "0") for the feature "isParsedTitle".
### Word Capitalization/Casing
The capitalization of a word can indicate it being a title. For example, considering the sentence: _"Prime Minister Malcolm Turnbull MP visited UNSW yesterday."_. Here the _Title_ entities are _Prime_ and _Minister_ both of which are capitalized. Titles can tend to be in upper case such as MP in the sentence considered here.
```
isCapitalized = {1 ⇒ isupper(token[0])/0 ⇒ ∼ isupper(token[0])}
isU pper      = {1 ⇒ isupper(token)/0    ⇒ ∼ isupper(token)
```
### POS tagging of the Preceding Token
The POS tag of the preceding token can indicate if the token is a title. The preceding tag can be a determiner such as "a doctor" or a preposition "Associate of Arts". The feature _isPrecedingDT_ checks if the preceding token is a determiner. It evaluates to _1_ if the preceding token has a POS tag of `<DT>` else _0_. The feature _isPrecedingIN_ checks if the preceding token is a preposition. It evaluates to _1_ if the preceding token has a POS tag of `<IN>` else _0_.
### Lookup against a Gazetteer
The model considers the presence of the token in a gazetteer (reference list for the title tokens) as a feature - _isInGazetteer_. The gazetteer contains a reference list of the lemmatized form of tokens that were identified as titles. If the token is present in the Gazetteer list, the feature is evaluated as _1_ else _0_. The gazetteer list is
built with the unique tokens identified from the combination training data set and the possible titles from the Wikipedia page (for title). Along with this lookup, the token is also matched against a list of stop words. The reference list for stop words is built from the module _stopwords_ in _nltk.corpus_.

## Improving the Classifier
The baseline F1 score for the classifier was 0.947191042924. The F1 score of the regression model on the training data set was 0.971727113768. This implies that the model has a significant improvement on the F1 score. The feature selection can be improved by methods like cross-validation or recursive feature elimination. The scikit-learn library _sklearn.feature_selection.RFE_ provides an in-built API to perform recursive feature elimination. It performs RFE (Recursive Feature Elimination) in a cross-validation loop to find the optimal number of features. The use of the API on the model and training data reveals the ranking of each of the features. The use of the API on the features reveals the ranking of each of the features:

| Feature               | Ranking |
| --------------------- | ------- |
| isUpper               | 5       |
| isInGazetteer         | 1       |
| isCapitalized         | 3       |
| isParsedTitle         | 1       |
| maxRelativeDistanceNE | 2       |
| isPrecedingDT         | 1       |
| isPrecedingIN         | 4       |

This implies that the model works really well when only the features with a ranking "1" is included:

1. isInGazetteer
2. isParsedTitle
3. maxRelativeDistanceNE

These features can be selected and the rest of the features can be eliminated from the model since they have a lower ranking. This also slightly improves the F1 score of the model to 0.972685994698.

## References

1. Logistic Regression in Python - http://blog.yhat.com/posts/logisticregression-and-python.html
2. Title - https://en.wikipedia.org/wiki/Title
3. Feature Selection in Python with Scikit-Learn - http://machinelearningmastery.com/feature-selection-in-python-withscikit-learn/
4. SkLearn RFE - http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html