import pickle
import sys
import nltk
import pandas
from nltk.stem.wordnet import WordNetLemmatizer

def build_sentence_tree(tagged_sentence):
    """Builds the sentence tree based on the IOB tags for person and date"""
    phrase=[]
    label = ""
    token_list = []
    for token in tagged_sentence:
        iob = token[2]
        word = token[:-1]
        if(iob=='O'):
            if(phrase!=[]):
                token_list.append(nltk.Tree(label,phrase))
                label=""
                phrase=[]
                token_list.append(word)
            else:
                token_list.append(word)
        else:
            if(iob[2:] in ["PERSON","DATE"]):
                if(label==iob[2:] or label==""):
                    label = iob[2:]
                    phrase.append(word)
                else:
                    token_list.append(nltk.Tree(label, phrase))
                    label = ""
                    phrase = []
                    phrase.append(word)

    if (phrase != []):
        token_list.append(nltk.Tree(label, phrase))

    return token_list

training_data = sys.argv[1]

with open(str(training_data),'rb') as f:
    training_set =  pickle.load(f)

# get the person name from the data set
pos = training_set
sentence_pos_tags = {}
for sentence in pos:
    # print(sentence)
    #print("\n\nParsed data:")

    # to lematize a word
    lmtzr = WordNetLemmatizer()

    # namedEnt = nltk.ne_chunk(sentence, binary = False)
    # #print(namedEnt)
    # title_List = []
    # for token in namedEnt:
    #     print(token)

    title_list=[]
    title_phrase = []
    # for token in sentence:
    #     if(token[2]=='O' and title_phrase==[]):
    #         title_list.append(token)
    #     elif(token[2]=='TITLE'):
    #         title_phrase.append(tuple(token[:-1]))
    #     else:
    #         title_list.append(nltk.Tree('TITLE',title_phrase))
    # print(title_list)
    namedEnt = nltk.ne_chunk(sentence, binary=False)
    # print(namedEnt)
    named_entity_tags = ""
    for f in sentence:
        if(type(f)==type([]) and f[-1]=='TITLE'):
            named_entity_tags+="|"+f[1]+"|"

    if(named_entity_tags in sentence_pos_tags):
        sentence_pos_tags[named_entity_tags] +=1
    else:
        sentence_pos_tags[named_entity_tags] = 1

print(sentence_pos_tags)



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

# frame = [item for sublist in training_set for item in sublist]
# counter = {}
# counter['title']=0
# counter['notitle']=0
# for word in frame:
#     if word[2]=='TITLE':
#         counter['title'] +=1
#     else:
#         counter['notitle'] += 1
#
# print(str(counter['title']/counter['notitle']))
# print(str(counter['notitle']/(counter['notitle']+counter['title'])))
