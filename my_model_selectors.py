import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class DummyHMM(GaussianHMM):
    def score(self, xes, lengths):
        return float('-inf')


class ModelSelector(object):
    """
    base class for model selection (strategy design pattern)
    """
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=15,
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
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return self.dummy_model()

    @staticmethod
    def dummy_model():
        return DummyHMM()
    
    @staticmethod
    def score(model, xes, lengths):
        try:
            return model.score(xes, lengths)
        except Exception:
            return float('-inf')


class SelectorConstant(ModelSelector):
    """
    select the model with value self.n_constant
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
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = self.dummy_model()
        num_data_points, num_features = self.X.shape
        best_num_states = -1
        min_bic = float('inf')
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(num_states)
            score = self.score(model, self.X, self.lengths)
            bic = self.bic(score, num_data_points, num_features, num_states)
            if bic < min_bic:
                min_bic = bic
                best_model = model
                best_num_states = num_states
                
        # print('selected model for word {}: {} states'.format(self.this_word, best_num_states))
        return best_model

    @staticmethod
    def bic(log_likelihood, num_data_points, num_features, num_states):
        """
        Number of parameters of our HMM (num_states =: n, num_features =: d):
        n - 1 start probabilities (the probabilities must add up to 1, therefore the '-1')
        n * (n - 1) transition probabilities (again the probabilities must add up to 1)
        2 * d * n for the mean and variance of the Gaussian distributions for each feature and each state
        
        So the sum is:
        p = n - 1 + n * (n - 1) + 2 * d * n = n^2 + 2 * d * n - 1
        """
        p = np.square(num_states) + 2 * num_features * num_states - 1
        alpha = .3
        bic = alpha * p * np.log(num_data_points) - 2 * log_likelihood
        return bic


class SelectorDIC(ModelSelector):
    """
    select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = self.dummy_model()
        best_num_states = -1
        max_dic = float('-inf')
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(num_states)
            score = self.score(model, self.X, self.lengths)
            other_scores = []
            for word in self.hwords:
                if word != self.this_word:
                    xes, lengths = self.hwords[word]
                    other_score = self.score(model, xes, lengths)
                    if other_score != float('-inf'):
                        other_scores.append(other_score)

            dic = self.dic(score, other_scores)
            if dic > max_dic:
                max_dic = dic
                best_model = model
                best_num_states = num_states

        # print('selected model for word {}: {} states'.format(self.this_word, best_num_states))
        return best_model

    @staticmethod
    def dic(log_likelihood, other_log_likelihoods):
        if log_likelihood == float('-inf'):
            return float('-inf')

        alpha = 1.
        return log_likelihood - alpha * np.mean(other_log_likelihoods)


class SelectorCV(ModelSelector):

    """
    Creates a model from the given parameters, or a dummy model if an exception occurs.
    """
    def create_model(self, num_states, xes, xlengths):
        try:
            return GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                               random_state=self.random_state, verbose=False).fit(xes, xlengths)
        except Exception as e:
            if self.verbose:
                print("failure on {} with {} states, error: {}".format(self.this_word, num_states, e))
            return self.dummy_model()

    """
    select best model based on average log Likelihood of cross-validation folds
    """
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        split_method = KFold()
        
        best_score = float('-inf')
        best_num_states = -1
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            if len(self.sequences) < 3:
                # Not enough data for cross validation. Create the model from all the data - this is the best we can do.
                model = self.base_model(num_states)
                score = self.score(model, self.X, self.lengths)
            else:
                scores = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_x, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_x, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                    # Use the training data to create the model
                    model = self.create_model(num_states, train_x, train_lengths)
                    # Use the test data to evaluate this model
                    score = self.score(model, test_x, test_lengths)
                    scores.append(score)
                        
                score = np.mean(scores)
#                print('scores for {} states: {}, mean score: {}'.format(num_states, scores, score))
                
            if score > best_score:
                best_score = score
                best_num_states = num_states
                
        # print('selected model for word {}: {} states'.format(self.this_word, best_num_states))
        # Return a model based on all data
        return self.base_model(best_num_states)
