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
    probabilities = []
    guesses = []

    for word_id in test_set.get_all_sequences():
        prob = {}
        for word in models:
            xes, lengths = test_set.get_item_Xlengths(word_id)
            try:
                score = models[word].score(xes, lengths)
            except Exception:
                score = float('-inf')

            prob[word] = score

        guess = max(prob, key=prob.get)
        probabilities.append(prob)
        guesses.append(guess)
        # print('guess for word {}: {}, score = {}'.format(word_id, guess, prob[guess]))
    
    return probabilities, guesses
