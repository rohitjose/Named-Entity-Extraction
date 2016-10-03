import pickle
import sys
import nltk
import pandas

training_data = sys.argv[1]

with open(str(training_data),'rb') as f:
    training_set =  pickle.load(f)

# get the person name from the data set
pos = training_set[0]
print(pos)
print("\n\nParsed data:")

namedEnt = nltk.ne_chunk(pos, binary = False)
print(namedEnt)

#sentence creation
sentence = " ".join([word[0] for word in pos])

print(sentence)

# Root data frame from the training set - (word,POS, Title info)
frame = [item for sublist in training_set for item in sublist]
print(frame[0])

#frame = [item if(str.isalpha(item[0])) else [] for item in frame] # for removing word with special character - might not be the right approach

# Data frame conversion
data_frame = pandas.DataFrame(frame,columns=["word","pos_tagging","isTitle"])
print(data_frame.head(20))

#data_frame["isTitle"] = [1 if(item=='TITLE') else 0 for item in data_frame["isTitle"]]

data_frame["isTitle"] = (data_frame.isTitle=='TITLE').astype(int)


print(data_frame.head(20))

# extracts the categorical value of the POS tagging in relation to the word
df = data_frame
dummy_ranks = pandas.get_dummies(df['pos_tagging'], prefix='POS')
print(dummy_ranks.head(20))

print(dummy_ranks.dtypes)
