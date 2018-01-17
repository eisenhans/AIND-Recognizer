import math
import statistics
import warnings
import sys

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        bics = {}
        for num_states in range(self.min_n_components, self.max_n_components + 1):
           
            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        
                score = hmm_model.score(self.X, self.lengths)
                bic = self.bic(score, 4, num_states)
                bics[num_states] = bic                
                
            except Exception as e:
                if self.verbose:
                    print("failure on {} with {} states, error: {}".format(self.this_word, num_states, e))
        
        min_key = min(bics, key = bics.get)                
        print('bics: {}, min: {}'.format(bics, min_key))
        
        try:
            chosen_model = GaussianHMM(n_components=min_key, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            return chosen_model

        except Exception as e:
            if self.verbose:
                print("failure on {} with {} states, error: {}".format(self.this_word, min_key, e))
        
        best_num_components = self.n_constant
        return self.base_model(best_num_components)
    
    def bic(self, log_likelihood, num_features, num_states):
        # TODO: understand this formula
        p = np.square(num_states) + 2 * num_states * num_features - 1
        bic = p * np.log(num_features) - 2 * log_likelihood
#        print('log_likelihood: {}, num_features: {}, num_states: {}, p: {}, penalty: {}, bic: {}'.format(
#                log_likelihood, num_features, num_states, p,  p * np.log(num_features), bic))
        
        return bic

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        dics = {}
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        
                score = hmm_model.score(self.X, self.lengths)
#                print('score for this word {} ({} states): {}'.format(self.this_word, num_states, score))
                other_scores = []
                better_scores = 0
                for word in self.hwords:
                    if word != self.this_word:
                        xes, lengths = self.hwords[word]
                        other_score = hmm_model.score(xes, lengths)
                        other_scores.append(other_score)
                        if other_score > score:
                            better_scores += 1
                            
#                print('{} out of {} words fit the model better than this word'.format(better_scores, len(self.hwords), self.this_word))
                dic = self.dic(score, other_scores)
                dics[num_states] = dic
                
            except Exception as e:
                if self.verbose:
                    print("failure on {} with {} states, error: {}".format(self.this_word, num_states, e))
        
        max_key = max(dics, key = dics.get)                
        print('dics: {}, max: {}'.format(dics, max_key))
        
        try:
            chosen_model = GaussianHMM(n_components=max_key, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            return chosen_model

        except Exception as e:
            if self.verbose:
                print("failure on {} with {} states, error: {}".format(self.this_word, max_key, e))
        
        best_num_components = self.n_constant
        return self.base_model(best_num_components)
    
    def dic(self, log_likelihood, other_log_likelihoods):
        # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
#        print('logL: {}, other logLs: {}, mean other logLs: {}'.format(log_likelihood, other_log_likelihoods, np.mean(other_log_likelihoods)))
        return log_likelihood - np.mean(other_log_likelihoods)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if len(self.sequences) < 3:
            print('SelectorCV does not work with only {} samples'.format(len(self.sequences)))
            return None
        
        split_method = KFold()
        
        mean_scores = {}
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
#                print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds                  
            
                train_x, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                test_x, test_lengths = combine_sequences(cv_test_idx, self.sequences)
            
#                print('len(train_samples): {}, train_lengths: {}, len(test_samples): {}, test_lengths: {}'.format(
#                        len(train_x), train_lengths, len(test_x), test_lengths))
                    
                try:
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_x, train_lengths)
            
                    score = hmm_model.score(test_x, test_lengths)
                    scores.append(score)
                    
#                    if self.verbose:
#                        print("model created for {} with {} states".format(self.this_word, self.n_constant))
                except Exception as e:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, self.n_constant))
                        print('error: ', e)
                
            mean_score = np.mean(scores)
            mean_scores[num_states] = mean_score
            print('scores for {} states: {}, mean score: {}'.format(num_states, scores, mean_score))
        
        max_key = max(mean_scores, key=mean_scores.get)                
        print('mean scores: {}, max: {}'.format(mean_scores, max_key))
        
        try:
            chosen_model = GaussianHMM(n_components=max_key, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            return chosen_model
#                    if self.verbose:
#                        print("model created for {} with {} states".format(self.this_word, self.n_constant))
        except Exception as e:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, self.n_constant))
                print('error: ', e)
        
        best_num_components = self.n_constant
        return self.base_model(best_num_components)
