import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    sequences = test_set.get_all_sequences()
#    xlengths = test_set.get_all_Xlengths()
#    item_seq1 = test_set.get_item_sequences(1)
    
    probabilities = []
    guesses = []
    
    for word_id in sequences:
        prob = {}
        for word in models:
            xes, lengths = test_set.get_item_Xlengths(word_id)
            score = models[word].score(xes, lengths)
            prob[word] = score
#            print('probality that word {} is {}: {}'.format(word_id, word, score))
        
        guess = max(prob, key=prob.get)
        probabilities.append(prob)
        guesses.append(guess)
        print('guess for word {}: {}, score = {}'.format(word_id, guess, prob[guess]))
    
    return probabilities, guesses
