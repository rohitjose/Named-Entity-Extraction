import pickle
from nltk.stem.wordnet import WordNetLemmatizer



set_words = {'Congressman', 'Representative', 'press', 'rabbi', 'Associate',
             'Presidency', 'editor', 'branch', 'elect', 'Enterprises', 'executives', 'lady',
             'agricultural', 'mayoralty', 'economist', 'electrical', 'speaker', 'Lord', 'Provincial',
             'operations', 'pseudonym', 'financial', 'doctor', 'governor', 'Administrator', 'aide',
             'commanding', 'journalist', 'Dr.', 'General', 'bishops', 'Governor', 'CEO', 'quarterback',
             'CFO', 'chair', 'Agriculture', 'assemblyman', 'Presidents', 'intelligence', 'prosecutor',
             'Ministers', 'professors', 'general', 'Plenipotentiary', 'priest', 'Executive', 'rector',
             'engineer', 'Senator', 'backer', 'founder', 'actress', 'Home', 'investment', 'writer', 'elders',
             'Commander', 'judge', 'Prophet', 'valedictorian', 'presidents', 'Agricultural', 'Queen', 'senior',
             'principal', 'developer', 'singer', 'federal', 'ministers', 'senator', 'scientist', 'chairperson',
             'media', 'cabinet', 'practitioner', 'congressman', 'King', 'Legislative', 'project', 'head', 'athletic',
             'Finance', 'Prince', 'Health', 'Gov.', 'mayor', 'president', 'Police', 'Crown', 'ambassadors', 'State',
             'acting', 'Acting', 'alongside', 'Minister', 'candidate', 'chief', 'Sen.', 'minister', 'Lifetime',
             'commanders', 'Chair', 'were', 'Inspector', 'Presidential', 'premier', 'Relations', 'Transport',
             'finance', 'senators', 'foreign', 'justice', 'competitors', 'Election', 'managing', 'Responsible',
             'prince', 'defence', 'associate', 'manager', 'generals', 'architect', 'High', 'Managing', 'staff',
             'deputy', 'technical', 'Ambassador', 'Assistant', 'envoys', 'Chairwoman', 'presidential', 'Officers',
             'Defense', 'archbishop', 'counsel', 'producer', 'business', 'painter', 'Bishop', 'Honorary', 'leader',
             'analyst', 'Majority', 'Comrade', 'independent', 'party', 'Professor', 'services', 'director', 'Doctor',
             'Prime', 'Cabinet', 'assistant', 'energy', 'Defence', 'communications', 'technology', 'cardinal',
             'negotiator', 'announcer', 'attorneys', 'advocate', 'Democrat', 'professor', 'interior', 'Pope',
             'vice', 'crown', 'Staff', 'MPs', 'directors', 'postmaster', 'Vice', 'Foreign', 'administration',
             'secretary', 'administrative', 'Political', 'Senior', 'coach', 'Ordinance', 'treasurer', 'Security',
             'Oilman', 'Education', 'activist', 'regional', 'Secretary', 'Premier', 'for', 'publisher', 'Founder',
             'until', 'budget', 'bank', 'operating', 'security', 'board', 'Mrs', 'Operating', 'national', 'officer',
             'Leader', 'chiefs', 'chancellor', 'software', 'Judge', 'Speaker', 'Rep.', 'Attorney', 'commentator',
             'cleric', 'Prosecutor', 'Second', 'disciplinary', 'Mrs.', 'lawyer', 'screenwriter', 'Chancellor',
             'chairwoman', 'king', 'author', 'portfolio', 'blogger', 'Captain', 'chairman', 'Environment',
             'Justice', 'Chief', 'governors', 'campaign', 'Chairman', 'Gen.', 'attorney', 'artist', 'sultan',
             'Miss', 'ambassador', 'works', 'Arts', 'interim', 'Parliamentary', 'Officer', 'partisan',
             'fascist', 'Mayor', 'President', 'defense', 'first', 'historian', 'businesswoman', 'prime', 'Ministership',
             'Directorate', 'commissioner', 'Deputy', 'Member', 'Editor', 'environmental', 'Emperor', 'institutions',
             'First', 'relations', 'pontiff', 'executive', 'columnist', 'industrialist', 'police', 'Atty.', 'Director',
             'Whip', 'public', 'Financial', 'captain', 'commander'}

# no = [',','.','-','/','!','@','#','$',' ','co','MP','Ms','in','TV','Mr','of','Rt','PM','were',
# 'for','until','environmental','envoys','First','technical','party']
no = [',', '.', '-', '/', '!', '@', '#', '$', ' ', 'co', 'MP', 'Ms', 'in', 'TV', 'Mr', 'of', 'Rt', 'PM', 'were',
      'for', 'until', 'environmental', 'envoys', 'First', 'technical', 'party', 'and']


def suffix(training_set):
    for lists in training_set:
        for sublist in lists:
            if sublist[0][-3] == 'ent' or sublist[0][-3] == 'tor':
                return True
            else:
                return False


def check_word_capitalization(word):
    """Checks whether a word is a capitalized word"""
    return_value = False
    if (len(word) > 1):
        # print(word,"1")
        return_value = True if (word[0].isupper() and word[1].islower()) else False


def build_training_features(training_data):
    # Load the data set
    with open(str(training_data), 'rb') as f:
        training_set = pickle.load(f)

    columns_values = ["suffix", "word", "POStag", "TitleMapping", "isUpper", "isCapitalized", "isNNP", "isNN", "isJJ",
                      "isPrecedingDT", "isPrecedingIN", "isTitle"]
    preceding_pos_tag = None

    # to lematize a word
    lmtzr = WordNetLemmatizer()

    features_and_labels = []
    for sentence in training_set:
        for word in sentence:
            # print(word,word[0],"word")
            token_dict = {}
            # Map the word into the dictionary

            #token_dict[lmtzr.lemmatize(word[0])] = 1
            token_dict["suffix"] = (1 if (suffix(training_set)) else 0)
            # WORD TEXT RELATED FEATURES
            # Check for a word in uppercase
            token_dict["isUpper"] = (1 if (word[0].isupper()) else 0)

            token_dict["fullwordcheck"] = (1 if (word[0] in set_words and word[0] not in no) else 0)
            # Check for word capitalization
            token_dict["isCapitalized"] = (1 if (check_word_capitalization(word[0])) else 0)

            # WORD POS TAGGING RELATED FEATURES
            # Check if the POS tagging is NNP - (noun, proper, singular)
            token_dict["isNNP"] = (1 if (word[1] == 'NNP') else 0)
            # Check if the POS tagging is NN - (noun, common, singular or mass)
            token_dict["isNN"] = (1 if (word[1] == 'NN') else 0)
            # Check if the POS tagging is JJ - (adjective or numeral, ordinal)
            token_dict["isJJ"] = (1 if (word[1] == 'JJ') else 0)

            # POS TAGGING OF PRECEDING WORD
            # Check if the preceding POS tagging was DT
            token_dict["isPrecedingDT"] = (1 if (preceding_pos_tag == 'DT') else 0)
            # Check if the preceding POS tagging was IN (preposition or conjunction, subordinating)
            token_dict["isPrecedingIN"] = (1 if (preceding_pos_tag == 'IN') else 0)
            # Map the POS tag of the preceding token
            preceding_pos_tag = word[1]

            # Append values to features and labels
            features_and_labels.append((token_dict, word[2]))

    # return the features and labels list
    return features_and_labels
