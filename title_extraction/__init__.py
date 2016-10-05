import pickle
import sys
import nltk
import pandas

training_data = sys.argv[1]

with open(str(training_data),'rb') as f:
    training_set =  pickle.load(f)

# # get the person name from the data set
# pos = training_set[10]
# print(pos)
# print("\n\nParsed data:")
#
# namedEnt = nltk.ne_chunk(pos, binary = False)
# print(namedEnt)
#
# #sentence creation
# sentence = " ".join([word[0] for word in pos])
#
# print(sentence)
#
# # Root data frame from the training set - (word,POS, Title info)
# frame = [item for sublist in training_set for item in sublist]
# print(frame[0])
#
# #frame = [item if(str.isalpha(item[0])) else [] for item in frame] # for removing word with special character - might not be the right approach
#
# # Data frame conversion
# data_frame = pandas.DataFrame(frame,columns=["word","pos_tagging","isTitle"])
# print(data_frame.head(20))
#
# #data_frame["isTitle"] = [1 if(item=='TITLE') else 0 for item in data_frame["isTitle"]]
#
# data_frame["isTitle"] = (data_frame.isTitle=='TITLE').astype(int)
#
#
# print(data_frame.head(20))
#
# # extracts the categorical value of the POS tagging in relation to the word
# df = data_frame
# dummy_ranks = pandas.get_dummies(df['pos_tagging'], prefix='POS')
# print(dummy_ranks.head(20))

# from sklearn.datasets import load_iris
# iris = load_iris()
# X, y = iris.data[:-1,:], iris.target[:-1]
# print(type(iris))

# from nltk.parse.stanford import StanfordDependencyParser
#
# dep_parser=StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
# print([parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")])

frame = [item for sublist in training_set for item in sublist]
counter = {}
counter['title']=0
counter['notitle']=0
for word in frame:
    if word[2]=='TITLE':
        counter['title'] +=1
    else:
        counter['notitle'] += 1

print(str(counter['title']/counter['notitle']))
print(str(counter['notitle']/(counter['notitle']+counter['title'])))