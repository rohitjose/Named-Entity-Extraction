import pickle
import sys
import nltk
import pandas


# Get the training data set path
training_data = sys.argv[1]

# Load the data set
with open(str(training_data),'rb') as f:
    training_set =  pickle.load(f)


# # Parse the words list to build the
# for pos in training_set:
#     # sentence creation
#     sentence = " ".join([word[0] for word in pos])
#     print(sentence)
#     namedEnt = nltk.ne_chunk(pos, binary=False)
#     print(namedEnt)


frame = [item for sublist in training_set for item in sublist]

count = {}

for item in frame:
    if(item[2]!='TITLE'):
        if(item[1] in count):
            count[item[1]] += 1
        else:
            count[item[1]] = 1
        if(item[1]=='JJ'):
            print(item)

print(count)


total_count = 0
for d in count:
    total_count+=count[d]

print(total_count)






