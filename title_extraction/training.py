import pickle
import sys
import nltk
import pandas

# def calculate_NE_relative_index:
#


# Get the training data set path
training_data = sys.argv[1]

# Load the data set
with open(str(training_data),'rb') as f:
    training_set =  pickle.load(f)

columns = ['isCapitalized','isUpperCase','isPreceedingCharacterPreposition','isPreceedingCharacterDeterminer','isNN','isNNP','isJJ']
#Parse the words list to build the features
# count = 0
# for pos in training_set:
#     # sentence creation
#     sentence = " ".join([word[0] for word in pos])
#     #print(sentence)
#     namedEnt = nltk.ne_chunk(pos, binary=False)
#     #print(namedEnt)
#     person = []
#     for subtree in namedEnt.subtrees(filter=lambda t: t.label() in ['PERSON','ORGANIZATION','GPE','LOCATION','DATE','TIME','MONEY','PERCENT','FACILITY','GSP']):
#         for leave in subtree.leaves():
#             person.append(leave)
#     if (person==[]):
#         print(namedEnt)
#         print(sentence)
#         count += 1

# pos = training_set[4]
# sentence = " ".join([word[0] for word in pos])
# person = []
# namedEnt = nltk.ne_chunk(pos, binary=False)
# for subtree in namedEnt.subtrees(filter=lambda t: t.label() in ['PERSON','ORGANIZATION','GPE']):
#     for leave in subtree.leaves():
#         person.append(leave)
# print(person)
# print(namedEnt)
# for word in person:
#     print(str(word)+str(pos.index(word)))

#print(len(pos))

print(len(training_set))


frame = [item for sublist in training_set for item in sublist]