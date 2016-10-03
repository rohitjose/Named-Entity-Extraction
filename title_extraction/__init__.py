import pickle
import sys
import nltk
import pandas

training_data = sys.argv[1]

with open(str(training_data),'rb') as f:
    training_set =  pickle.load(f)

pos = training_set[0]
print(pos)
print("\n\nParsed data:")

namedEnt = nltk.ne_chunk(pos, binary = False)
print(namedEnt)

sentence = " ".join([word[0] for word in pos])

print(sentence)

frame = [item for sublist in training_set for item in sublist]
print(frame[0])

data_frame = pandas.DataFrame(frame,columns=["word","pos_tagging","isTitle"])
#
print(data_frame.head(20))


