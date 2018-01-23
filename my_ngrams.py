import arpa

def one_gram_probs(probs: list):
    models = arpa.loadf("data/ukn.1.lm")
    print('models: {}'.format(models))
    alpha = 10.

    lm = models[0]
    
    one_gram_probs = []
    one_gram_guesses = []
    
    for prob in probs:
        one_gram_prob = {}
        for word in prob:
            one_gram_key = word
            if not one_gram_key in lm.vocabulary() and one_gram_key[-1:].isdigit():
                one_gram_key = one_gram_key[:-1]
            if not one_gram_key in lm.vocabulary():
                one_gram_key = '[UNKNOWN]'
                print('unknown word found: '.format(word))
                
            frequency = lm.log_p(one_gram_key)
            one_gram_prob[word] = prob[word] + alpha * frequency
    #        print('word {}: prob {}, frequency {}, new prob {}'.format(word, prob[word], frequency, one_gram_prob[word]))
            
        one_gram_probs.append(one_gram_prob)
            
    for one_gram_prob in one_gram_probs:
        one_gram_guess = max(one_gram_prob, key=one_gram_prob.get)
        one_gram_guesses.append(one_gram_guess)
        
    return one_gram_probs, one_gram_guesses

def two_gram_guesses(test_set, probs: list):
#    models = arpa.loadf("data/devel-lm-M2.sri.lm")
    models = arpa.loadf("data/ukn.2.lm")

    lm = models[0]
    sentences = []
    
    for video_num in test_set.sentences_index:
        sentence_probs = [probs[word_index] for word_index in test_set.sentences_index[video_num]]
        
        sentence = sentence_guess(sentence_probs, lm)
        sentences.extend(sentence)
    
    return sentences

def sentence_guess(probs: list, lm):
#    print('handling sentence {}'.format(probs))
    probs_begin_end = [{'<s>': 0.}] + probs + [{'</s>': 0.}]
    begin_candidates = [(['<s>'], 0.)]
    
    best_guesses = viterbi_path(probs_begin_end, lm, begin_candidates, 5, 10.)
    print('guess: {}'.format(best_guesses[0]))
#    for i in range(0, 3):
#        print('guess {}: {}'.format(i, best_guesses[i]))
    
    sentence = best_guesses[0][0]
    
    return sentence[1:-1]
    
def viterbi_path(probs: list, lm, candidates: list, breadth: int, alpha: float):
    '''
    :param candidates: list of tuples of the form (path, value)
    '''
    word_count = len(candidates[0][0])
    if word_count >= len(probs):
        return candidates
    
    prob = probs[word_count]
    new_candidates = []
    for candidate in candidates:
        path = candidate[0]
        value = candidate[1]
        for word in prob:
            words_for_lm = lm_word(path[-1], lm) + ' ' + lm_word(word, lm)
            ngram_value = lm.log_p(words_for_lm)
            word_prob = prob[word]
            word_value = alpha * ngram_value + word_prob
            new_path = path + [word]
            new_value = value + word_value
            new_candidates.append((new_path, new_value))
            
    new_candidates.sort(key = lambda tup: tup[1], reverse = True)
    new_candidates = new_candidates[:breadth]
    
    return viterbi_path(probs, lm, new_candidates, breadth, alpha)
    
def lm_word(word, lm):
    if word in lm.vocabulary():
        return word
    word = word[:-1]
    if word in lm.vocabulary():
        return word
    return '[UNKNOWN]'


def three_gram_guesses(test_set, probs: list):
#    models = arpa.loadf("data/devel-lm-M3.sri.lm")
    models = arpa.loadf("data/ukn.3.lm")

    lm = models[0]
    sentences = []
    
    for video_num in test_set.sentences_index:
        sentence_probs = [probs[word_index] for word_index in test_set.sentences_index[video_num]]
        
#        if video_num == 21:
        sentence = three_gram_sentence_guess(sentence_probs, lm)
#        else:
#            sentence = ['GO'] * len(sentence_probs)
        sentences.extend(sentence)
        
    return sentences

def three_gram_sentence_guess(probs: list, lm):
    probs_begin_end = [{'<s>': 0.}] + probs + [{'</s>': 0.}]
    begin_candidates = [(['<s>'], 0.)]
    
    best_guesses = three_gram_viterbi_path(probs_begin_end, lm, begin_candidates, 5, 10.)
    print('guess: {}'.format(best_guesses[0]))
#    for i in range(0, 3):
#        print('guess {}: {}'.format(i, best_guesses[i]))
    
    sentence = best_guesses[0][0]
    
    return sentence[1:-1]
    
def three_gram_viterbi_path(probs: list, lm, candidates: list, breadth: int, alpha: float):
    '''
    :param candidates: list of tuples of the form (path, value)
    '''
    word_count = len(candidates[0][0])
    if word_count >= len(probs):
        return candidates
    
    prob = probs[word_count]
    new_candidates = []
    for candidate in candidates:
        path = candidate[0]
        value = candidate[1]
        for word in prob:
            if len(path) == 1:
                words_for_lm = lm_word(path[-1], lm) + ' ' + lm_word(word, lm)
            else:
                words_for_lm = lm_word(path[-2], lm) + ' ' + lm_word(path[-1], lm) + ' ' + lm_word(word, lm)
            
            ngram_value = lm.log_p(words_for_lm)
#            print('path: {}, word: {}, ngram: {} -> {}'.format(path, word, words_for_lm, ngram_value))
            word_prob = prob[word]
            word_value = alpha * ngram_value + word_prob
            new_path = path + [word]
            new_value = value + word_value
            new_candidates.append((new_path, new_value))
#            print('new path: {}, new value: {}'.format(new_path, new_value))
            
    new_candidates.sort(key = lambda tup: tup[1], reverse = True)
    new_candidates = new_candidates[:breadth]
#    print('10 best new candidates: {}'.format(new_candidates))
    
    return three_gram_viterbi_path(probs, lm, new_candidates, breadth, alpha)






