# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:49:36 2018

@author: Markus
"""
import timeit
import numpy as np
import pandas as pd
from asl_data import AslDb
import warnings
import hmmlearn
from hmmlearn.hmm import GaussianHMM
from my_model_selectors import (SelectorCV, SelectorBIC, SelectorDIC)
from asl_utils import combine_sequences
import math
from matplotlib import (cm, pyplot as plt, mlab)

asl = AslDb() # initializes the database
#print('asl head: {}'.format(asl.df.head()))
#
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
#asl.df.head()
#
## collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
# #show a single set of features for a given (video, frame) tuple
#[asl.df.ix[98,1][v] for v in features_ground]

#training = asl.build_training(features_ground)
#print("Training words: {}".format(training.words))
#seqs = training.get_all_sequences()
#print('John sequence: {}'.format(seqs['JOHN']))
#print('john xlengths: {}'.format(training.get_word_Xlengths('JOHN')))

#john = training.get_word_sequences('JOHN')

#print('john: {}'.format(john))
#print(type(john))
#first = john[0]
#second = john[1]

df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

asl.df['right-x-mean'] = asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean'] = asl.df['speaker'].map(df_means['right-y'])
asl.df['left-x-mean'] = asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean'] = asl.df['speaker'].map(df_means['left-y'])

asl.df['right-x-std'] = asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std'] = asl.df['speaker'].map(df_std['right-y'])
asl.df['left-x-std'] = asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std'] = asl.df['speaker'].map(df_std['left-y'])

asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

asl.df['polar-rr'] = np.sqrt(asl.df['grnd-rx']**2 + asl.df['grnd-ry']**2)
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = np.sqrt(asl.df['grnd-lx']**2 + asl.df['grnd-ly']**2)
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

df_delta = asl.df[['right-x', 'right-y', 'left-x', 'left-y']].diff().fillna(0)
asl.df['delta-rx'] = df_delta['right-x']
asl.df['delta-ry'] = df_delta['right-y']
asl.df['delta-lx'] = df_delta['left-x']
asl.df['delta-ly'] = df_delta['left-y']
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

df_norm_delta = asl.df[['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']].diff().fillna(0)
asl.df['norm-delta-rx'] = df_norm_delta['norm-rx']
asl.df['norm-delta-ry'] = df_norm_delta['norm-ry']
asl.df['norm-delta-lx'] = df_norm_delta['norm-lx']
asl.df['norm-delta-ly'] = df_norm_delta['norm-ly']
# better than features_delta_norm
# idea: First normalize so that they are comparable. Then delta.
features_norm_delta = ['norm-delta-rx', 'norm-delta-ry', 'norm-delta-lx', 'norm-delta-ly',]


df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

asl.df['delta-rx-mean'] = asl.df['speaker'].map(df_means['delta-rx'])
asl.df['delta-ry-mean'] = asl.df['speaker'].map(df_means['delta-ry'])
asl.df['delta-lx-mean'] = asl.df['speaker'].map(df_means['delta-lx'])
asl.df['delta-ly-mean'] = asl.df['speaker'].map(df_means['delta-ly'])

asl.df['delta-rx-std'] = asl.df['speaker'].map(df_std['delta-rx'])
asl.df['delta-ry-std'] = asl.df['speaker'].map(df_std['delta-ry'])
asl.df['delta-lx-std'] = asl.df['speaker'].map(df_std['delta-lx'])
asl.df['delta-ly-std'] = asl.df['speaker'].map(df_std['delta-ly'])

asl.df['delta-norm-rx'] = (asl.df['delta-rx'] - asl.df['delta-rx-mean']) / asl.df['delta-rx-std']
asl.df['delta-norm-ry'] = (asl.df['delta-ry'] - asl.df['delta-ry-mean']) / asl.df['delta-ry-std']
asl.df['delta-norm-lx'] = (asl.df['delta-lx'] - asl.df['delta-lx-mean']) / asl.df['delta-lx-std']
asl.df['delta-norm-ly'] = (asl.df['delta-ly'] - asl.df['delta-ly-mean']) / asl.df['delta-ly-std']
# worse than features_norm_delta
features_delta_norm = ['delta-norm-rx', 'delta-norm-ry', 'delta-norm-lx', 'delta-norm-ly']
#
#print(asl.df.head())

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()

#def train_a_word(word, num_hidden_states, features):
#    
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    training = asl.build_training(features)  
#    X, lengths = training.get_word_Xlengths(word)
#    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
#    logL = model.score(X, lengths)
#    return model, logL
#
#demoword = 'BOOK'
#model, logL = train_a_word(demoword, 3, features_ground)
#print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
#print("logL = {}".format(logL))

#words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
words_to_train = ['BOOK']
training = asl.build_training(features_delta_norm)  # Experiment here with different feature sets defined in part 1
# sequences and xlengths contain the same information in different form. sequences is more
# human-friendly, xlengths is for efficient calculation.
sequences = training.get_all_sequences()
xlengths = training.get_all_Xlengths()

#print('sequences: {}'.format(sequences))
#print('\nXlengths: {}'.format(Xlengths))
# one element of john_sequence is the word 'JOHN' signed once, i.e. a list of quadruples
word_sequence = sequences[words_to_train[0]]
# john_xlength is a tuple, the first part of this tuple is a list of quadruples where all the
# elements of john_sequence have been concatenated, i.e. it starts with the elements of
# john_sequence[0], then the elements of john_sequence[1] follow etc.
word_xlength = xlengths[words_to_train[0]]
#print('word sequence: {}'.format(word_sequence))
#print('word xlength: {}'.format(word_xlength))
#print('done')
#
for word in words_to_train:
    print('training word {}'.format(word))
    start = timeit.default_timer()
    model = SelectorDIC(sequences, xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14, verbose = True).select()
    end = timeit.default_timer()-start
    
#    if model:
#        visualize(word, model)
#    if model is not None:
#        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
#    else:
#        print("Training failed for {}".format(word))

