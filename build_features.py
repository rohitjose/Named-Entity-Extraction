import pickle
import numpy as np
import nltk
from nltk import RegexpParser
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

title_reference_list = ['Solicitor General', 'crown', 'Lady', 'Mother Superior', 'counsel', 'Colonel General', 'BEd',
                        'Flag Officer', 'MDrama ', 'Prime', 'MPl ', 'Chief Risk Officer', 'Agricultural', 'Presidency',
                        'Chief Marketing Officer', 'Chief Operating Officer', 'Pandit', 'DA ', 'Captain General',
                        'D.Prof ', 'Staff', 'administration', 'Ordinance', 'ambassador', 'BEng', 'Privy Counsellor',
                        'Lord High Almoner', 'Tetrarch', 'technical', 'Kabaka', 'MPhys ', 'budget', 'MPs',
                        'Plenipotentiary', 'founder', 'Tsar', 'fascist', 'Cardinal', 'High', 'independent', 'Bishop',
                        'bank', 'Satrap', 'singer', 'MScBMC ', 'Customs officer', 'Justice of the Peace', 'Sergeant',
                        'producer', 'Friar', 'Responsible', 'Mahdi', 'BDes', 'relation', 'Officer', 'Ph.D.',
                        'BDentTech', 'MEd ', 'Empress', 'professor', 'Primate', 'Ustad', 'Marquis', 'advocate', 'MSE ',
                        'Legate', 'Democrat', 'operation', 'Prosecutor', 'until', 'postmaster', 'Bachelor', 'Acting',
                        'Officers', 'head', 'treasurer', 'managing', 'Chairwoman', 'press', 'party', 'Madam', 'CFO',
                        'Parliamentary', 'manager', 'Founder', 'quarterback', 'Corporate officer', 'MEng ',
                        'Professional Engineer', 'MLA ', 'BMus', 'Dato', 'commissioner', 'national', 'Education',
                        'Emir', 'AS', 'Lifetime', 'Attorney', 'Chair', 'President', 'Nath', 'cardinal', 'painter',
                        'Lieutenant', 'Petty Officer', 'economist', 'Tirthankar', 'First Mate', 'Prof', 'secretary',
                        'aide', 'Bailiff', 'BFA', 'Almoner', 'Chief Financial Officer', 'Master of the Rolls',
                        'Constable', 'first', 'Foreign', 'Whip', 'director', 'Admiral', 'Sultan', 'Commander',
                        'General Officer', 'branch', 'federal', 'actress', 'MPH ', 'BPhil', 'technology', 'Buddha',
                        'senator', 'Election', 'valedictorian', 'MMath ', 'Gen.', 'interim', 'attorney', 'Corporal',
                        'MSc ', 'Presidential', 'Doge', 'officer', 'doctor', 'journalist', 'pseudonym',
                        'Lord Justice Clerk', 'Secretary General', 'columnist', 'Secretary', 'Lord Protector',
                        'Saoshyant', 'BSc', 'Viscount', 'Miss', 'Rep.', 'presidential', 'Chief Knowledge Officer',
                        'Vardapet', 'General', 'Shah', 'mayoralty', 'ThM ', 'Decemvir', 'analyst', 'MDS ', 'police',
                        'Master Chief Petty Officer', 'Chief Information Security Officer', 'Maharani', 'Senator',
                        'CEO', 'Principal', 'envoy', 'DO', 'Doctorandus', 'chairman', 'judge', 'Imperator', 'Docent',
                        'Agent', 'Governor General', 'Representative', 'operating', 'S.T.M. ', 'business', 'MArch ',
                        'Prince', 'Pope', 'Nanny', 'Mayor', 'Nawab', 'Agriculture', 'Eagle Scout', 'foreign', 'Eze',
                        'Speaker', 'Ms', 'board', 'Captain', 'BDS', 'editor', 'mayor', 'Venerable',
                        'Lieutenant General', 'governor', 'Senior Officer', 'MPhil ', 'MSRE ', 'artist', 'energy',
                        'MPS ', 'Archduchess', 'Obi', 'Patil', 'Chief Petty Officer', 'institution',
                        'Lord President of the Court of Session', 'Dame', 'Congressman', 'Comrade', 'MC ', 'acting',
                        'minister', 'senior', 'Presbyter', 'priest', 'Enterprises', 'Saint', 'MET ', 'Asantehene',
                        'Almamy', 'Archdeacon', 'Magistrate', 'Prelate', 'Lord Justice of Appeal', 'prime', 'assistant',
                        'Crown', 'activist', 'Dom', 'Queen Guide', 'Lictor', 'Ship Officer', 'Christ', 'deputy',
                        'BBiotech', 'Surgeon General', 'Chief Technical Officer', 'MURP ', 'Rosh HaYeshiva', 'Senior',
                        'Dr.', 'Chief Security Officer', 'Blessed', 'Arts', 'prosecutor', 'captain', 'Major General',
                        'Emira', 'BSN', 'competitor', 'portfolio', 'author', 'Archduke', 'Mr', 'intelligence',
                        'Promagistrate', 'Oilman', 'Oba', 'interior', 'Grand Mufti', 'Health', 'MB', 'BEnvd', 'public',
                        'work', 'Major', 'Home', 'writer', 'Executive', 'Ayatollah', 'publisher', 'MHist ', 'Commodore',
                        'Regina', 'Imam', 'Admiralty Judge', 'Political Officer', 'Patriarch', 'Chief Strategy Officer',
                        'JD ', 'associate', 'Triumvir', 'executive', 'Nizam', 'Lord Justice General', 'Financial',
                        'PsyD ', 'cleric', 'elect', 'MDiv ', 'Mx', 'Catholicos', 'Viceroy', 'DPA ', 'commander',
                        'leader', 'Editor', 'Legislative', 'Professional Nurse', 'Honorary', 'First', 'Pharm.D. ',
                        'coach', 'Chief Business Development Officer', 'Chief Scout', 'Pastor', 'Marquess', 'Mirza',
                        'Abbot', 'Junior Warrant Officer', 'Warrant Officer.', 'Brother', 'announcer', 'BArch', 'chief',
                        'Majority', 'Chief Academic Officer', 'BA', 'Bodhisattva', 'Registered Nurse', 'Pharaoh',
                        'Master Warrant Officer', 'Reader', 'First Officer', 'D.Min. ', 'archbishop', 'Canon', 'Second',
                        'Flying Officer', 'Chief analytics officer', 'Master', 'Hakham', 'MPA ', 'Sen.', 'Mwami',
                        'architect', 'Mother', 'Environment', 'Maharajah', 'DMA ', 'electrical', 'Advocate General AG',
                        'Assistant', 'Leader', 'elder', 'Baron', 'Chief', 'industrialist', 'general', 'Nurse',
                        'MPharm ', 'Premier', 'congressman', 'Defense', 'administrative', 'Atty.', 'sultan', 'D.D. ',
                        'D.Sc. ', 'Mahatma', 'Barrister', 'Caesar', 'MSW ', 'defense', 'Rabbi', 'MAL ', 'Seneschal',
                        'president', 'MRes ', 'bishop', 'Brigadier', 'for', 'Maid', 'Police', 'assemblyman',
                        'practitioner', 'EngD or DEng ', 'Chancellor', 'Finance', 'Administrator', 'Vice', 'Queen',
                        'screenwriter', 'Sheikh', 'Grand Duchess', 'chair', 'BD', 'Lord Chief Justice', 'security',
                        'Father', 'pontiff', 'Priest', 'Mrs.', 'Director General', 'Reverend Mother', 'Defence',
                        'Chairman', 'partisan', 'Lord Lieutenant', 'Staff Officer', 'MFA ', 'BMath', 'Associate',
                        'Swami', 'Security', 'athletic', 'DFA ', 'Princess', 'Solicitor', 'alongside', 'D.Phil. ',
                        'BSBA', 'Generalissimo', 'MHA ', 'Ministership', 'communication', 'were', 'rector',
                        'General of the Army', 'Advocate', 'BDiv', 'businesswoman', 'MA ', 'Resident General',
                        'Basileus', 'Count', 'AAS', 'Wizard', 'Magister ', 'Operating', 'Grand Duke', 'Malik',
                        'Relations', 'Deacon', 'agricultural', 'candidate', 'principal', 'Th.D. ', 'Officer of State',
                        'MLitt', 'Servant of God', 'Police Officer', 'State', 'BVSc', 'Lama', 'Coach', 'Duce',
                        'Chaplain', 'Cabinet', 'Consul', 'Rebbe', 'Professor', 'Mate', 'Sapa Inca', 'Druid', 'Omukama',
                        'Monsignor', 'chairperson', 'D.Mus. ', 'Reeve', 'Duke', 'environmental', 'Countess', 'Scout',
                        'king', 'Sultana', 'Chief Information Officer', 'AA', 'service', 'Archdruid', 'backer',
                        'Colonel', 'Vicereine', 'lady', 'DBA ', 'Field officer', 'Ministers', 'investment',
                        'Tor Tiv of Tiv', 'Dean', 'Baroness', 'Mikado', 'Governor', 'Member', 'defence', 'blogger',
                        'Khagan', 'Mullah', 'Chief Credit Officer', 'Inspector', 'Ambassador', 'Counsel', 'King',
                        'Roman dictator', 'historian', 'campaign', 'prince', 'rabbi', 'Deputy', 'speaker', 'Prophet',
                        'MD ', 'Chief Executive Officer', 'Elder', 'Vicar General', 'Intelligence Officer',
                        'disciplinary', 'vice', 'justice', 'BN', 'chancellor', 'commentator', 'Judge', 'Tsarina',
                        'Caliph', 'Presidents', 'developer', 'Emperor', 'Pilot Officer', 'Provincial', 'Gov.',
                        'Flight Lieutenant', 'Mufti', 'chairwoman', 'Sister', 'Air Officer', 'LL.D. ', 'lawyer', 'STB',
                        'commanding', 'MChem ', 'Registered Engineer', 'cabinet', 'financial', 'BTh', 'medium', 'Lord',
                        'Acolyte', 'BBA', 'Revenue Officer', 'Chief Warrant Officer', 'Archon', 'Private', 'Saopha',
                        'regional', 'Minister', 'Negus', 'staff', 'Transport', 'finance', 'Abbess', 'Adjutant General',
                        'project', 'Political', 'negotiator', 'Ed.D. ', 'Doctor', 'Chief Mate', 'premier', 'BSW',
                        'M.Des ', 'Führer', 'software', 'BChD', 'scientist', 'Kohen', 'MBA ', 'First Lieutenant', 'Dr',
                        'Mrs', 'Attorney General', 'Vicar', 'engineer', 'Justice', 'Managing', 'LLB', 'Tribune', 'Earl',
                        'Director', 'Directorate', 'Sir', 'Caudillo', 'Khan']

# Setting the corpus for the words that cannot be a title
not_title = [',', '.', '-', '/', '!', '@', '#', '$', ' ']




def check_word_capitalization(word):
    """Checks whether a word is a capitalized word"""
    return_value = False
    if (len(word) > 1):
        # print(word,"1")
        return_value = True if (word[0].isupper() and word[1].islower()) else False

def parse_title(sentence):
    title_grammar = r"""
              TITLE:
                {<NNP>+<NN><NN>+}
                {<NNP><NNP>+}
                {<NN>?<JJ><NN|NNP>+}
                {<NNP><IN><NNP>}"""
    namedEnt = ne_chunk(sentence, binary=False)
    cp = RegexpParser(title_grammar)
    return cp.parse(namedEnt)


def download_nltk_packages():
    """Sets up the necessary NLTK packages"""
    run_status = False

    try:
        # Download the NLTK packages if not present
        nltk.download("averaged_perceptron_tagger")
        nltk.download("punkt")
        nltk.download("stopwords")
        run_status = True
    except:
        pass

    stop_words = set(stopwords.words('english'))
    not_title.extend(stop_words)

    return run_status


def isparsed_title(sentence_list):
    """ Parse the tree for entities"""
    # Parse the string to identify TITLE sequences
    entities = {}
    sentence_string = " ".join([word[0] for word in sentence_list])
    words = word_tokenize(sentence_string)
    tokens = nltk.pos_tag(words)
    sentence_parse_tree = parse_title(tokens)
    title_list = []
    for titles in sentence_parse_tree.subtrees(filter=lambda t: t.label() == 'TITLE'):
        title_list.append(titles.leaves())

    named_entities = []
    for names in sentence_parse_tree.subtrees(filter=lambda t: t.label() in ['ORGANIZATION','PERSON','GPE','LOCATION','FACILITY']):
        named_entities.append(names.leaves())

    entities['title'] = [word[0][0] for word in title_list]
    entities['named'] = [name[0] for name in named_entities]

    names_list = []
    for names in named_entities:
        for name in names:
            names_list.append(name[0])
    entities['named'] = names_list

    return entities

def get_named_entity_index(sentence,named_entities):
    """Retrieve the index values for the named enitites"""
    index_list = []
    counter = 0
    for word in sentence:
        if word[0] in named_entities:
            index_list.append(counter)
        counter += 1
    return index_list


def build_training_features(training_data):
    """ Build the features for the data set"""
    # Load the data set
    with open(str(training_data), 'rb') as f:
        training_set = pickle.load(f)

    preceding_pos_tag = None

    #Check if the required packages are available
    check_named_entities = download_nltk_packages()


    features_and_labels = []
    for sentence in training_set:
        if (check_named_entities):
            # Get the titles and the Named entities in the sentence
            entities = isparsed_title(sentence)
            # Index list of the named entities in the sentence
            index_list = get_named_entity_index(sentence, entities['named'])
        for word in sentence:
            # Initialize the dictionary
            token_dict = {}

            # WORD TEXT RELATED FEATURES
            # Check for a word in uppercase
            token_dict["isUpper"] = (1 if (word[0].isupper()) else 0)

            token_dict["wordMapping"] = (1 if (word[0] in title_reference_list and word[0] not in not_title) else 0)
            # Check for word capitalization
            token_dict["isCapitalized"] = (1 if (check_word_capitalization(word[0])) else 0)

            if(check_named_entities):
                # WORD POS TAGGING RELATED FEATURES
                # Check is the pattern in text is recognized as a title
                token_dict["isParsedTitle"] = (1 if(word[0] in entities['title']) else 0)
                if(index_list!=[]):
                    token_dict["maxRelativeDistanceNE"] = max([ abs(index-sentence.index(word)) for index in index_list])
                else:
                    token_dict["maxRelativeDistanceNE"] = 0

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

def make_flat(data):
    return [item for sublist in data for item in sublist]

def make_data(data, target):
    final_data = []
    for sentence_index, sentence in enumerate(data):
        sentence_data = []
        for word_index, word in enumerate(sentence):
            sentence_data.append((word, target[sentence_index][word_index]))
        final_data.append(sentence_data)
    return make_flat(final_data)

def score(classifier,X_train,y_train,state=0):
    kf = KFold(n_folds=5, shuffle=True, random_state=state)
    f1_array = np.zeros(5)
    fold = 0
    for itrain, itest in kf:
        print('Folding: {}'.format(fold + 1))
        Xtr, Xte = np.array(X_train)[itrain], np.array(X_train)[itest]
        ytr, yte = np.array(y_train)[itrain], np.array(y_train)[itest]
        build = make_data(Xtr, ytr)
        classifier.train(build)
        val = make_flat(Xte)
        val_true = make_flat(yte)
        val_pred = classifier.classify_many(val)
        enc = LabelEncoder().fit(classifier.labels())
        f1_array[fold] = f1_score(enc.transform(val_true), enc.transform(val_pred))
        fold += 1
    return np.mean(f1_array)