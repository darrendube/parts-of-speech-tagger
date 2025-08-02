'''
This module implements a parts-of-speech tagger based on a Hidden Markov Model. 

After training, the model takes in a sentence and outputs it's (predicted) parts-of-speech
tags.

Author: Darren Dube
Website: https://darrendube.github.io/
Copyright (c) 2025 Darren Dube 
'''

import numpy as np
np.seterr(divide="ignore") 

class DiscreteRV:
    '''
    A Discrete random variable.

    Usage:
    1. Initialise the random variable by passing in an array of items, and 
    an array of probabilities associated with those items. 
    2. Use `pmf()` to get the probability associated with an item
    3. Use `sample()` to sample an item from the distribution
    '''

    def __init__(self, tokens:np.ndarray, probabilities:np.ndarray):
        assert tokens.shape[0] == probabilities.shape[0], "Mismatch of lengths"
        assert abs(sum(probabilities) - 1.0) < 1e-7, "Probabilities \
         do not add up to 1"
        self.tokens = tokens
        self.probabilities = probabilities
        self._dict = dict(zip(self.tokens, self.probabilities))


    def pmf(self, token):
        '''Get the probability associated with an item'''
        return self._dict.get(token, 0.0)


    def sample(self, n=1):
        '''Sample an item (or items if `n`>1) from the distribution'''
        return np.random.choice(self.tokens, size=n, p=self.probabilities)



class POSTagger:
    '''
    Parts of Speech Tagger based on the Hidden Markov Model.

    Usage:
    1. Initialise the class.
    2. Either use `load()` to load a transition matrix and list of emission
    distributions, or train the model by passing training data to `fit()`.
    3. Use `get_tags()` to tag a given text with its parts of speech (uses Viterbi
    decoding).
    4. Use `sample` to generate sample texts/sentences. NOTE: the sentences generated 
    are likely to be nonsensical as this class doesn't model syntax, semantics, or context.
    '''
    
    def fit(self, signals:list, tags:list):
        '''
        Train the POS Tagger using the training data provided in `signal` and 
        `tags`.

        Parameters
        ----------
        signals : list
            An list of training data, consisting of numpy arrays of several different training
            sequences.

        tags : list
            An list of numpy arrays of tags corresponding to each token in `signal`
        '''
        lowercase_signals = [np.char.lower(arr) for arr in signals]
        self._unique_tags = np.unique(np.concatenate(tags))
        self.A = self._estimate_trans_matrix(tags)
        self.state_dists = self._estimate_state_dists(lowercase_signals, tags)
        

    def load_hmm(self, transition_matrix, state_dists):
        '''
        Initialise the POS Tagger using an already-generated transition matrix and state
        dsitributions.

        Parameters
        ----------
        transition_matrix : ndarray
            A transition matrix. The last row must correspond to an "INIT" state, and the 
            last column must correspond to a "TERMINAL" state.
        state_dists : list
            A list of `DiscreetRV` objects corresponding to the tags.
        '''
        self.A = transition_matrix
        self.state_dists = state_dists


    def get_tags(self, signal:np.ndarray):
        '''
        Assign parts-of-speech tags to a given sentence by performing Viterbi decoding.

        Parameters
        ----------
        signal : ndarray
            A 1D numpy array of String tokens - a token can be a word or punctuation mark.

        Returns
        -------
        ndarray
            Assigned parts-of-speech tags
        '''
        signal = np.char.lower(signal)
        np.seterr(invalid="ignore") 
        N = len(self.state_dists)
        T = signal.shape[0]
        Delt = np.zeros((N, T))
        Backp = np.zeros((N, T), dtype=int)

        # initialisation
        for j in range(N):
            Delt[j,0] = np.log(self.A[-1,j]) + np.log(self.state_dists[j].pmf(signal[0]))
            Backp[j,0] = -1

        # recursion
        for t in range(1,T):
            for j in range(N):
                vals = []
                for i in range(N):
                    vals += [Delt[i,t-1] + np.log(self.A[i,j])]
                Delt[j, t] = np.log(self.state_dists[j].pmf(signal[t])) + np.max(vals)
                Backp[j, t] = int(np.argmax(vals))

        # termination
        vals = []
        for j in range(N):
            vals += [np.log(self.A[j, N]) + Delt[j,T-1]]
        b_T = np.argmax(vals)

        # get optimal state sequence
        seq = [b_T]
        t = T - 1
        curr = b_T
        while True:
            if Backp[int(curr), t] == -1:
                break
            seq += [int(Backp[int(curr), t])]
            curr = seq[-1]
            t -= 1

        return self._int_to_tag(np.array(seq)[::-1])


    def _estimate_trans_matrix(self, tags):
        '''
        Estimate the transition matrix using the training tag data.

        Parameters
        ----------
        tags : list
            A list of numpy arrays of tags corresponding to the sentences in the
            training data

        Returns
        -------
        ndarray
            The estimated transition matrix
        '''
        N = self._unique_tags.shape[0]
        k = 1 # k value for Laplace smoothing
        state_seqs = [self._tag_to_int(tag_seq) for tag_seq in tags]
        transition_counts = np.zeros((N+1, N+1))

        # pad each state sequence with initial and terminal state
        for i in range(len(state_seqs)):
            state_seqs[i] = np.insert(state_seqs[i], 0, -1)
            state_seqs[i] = np.append(state_seqs[i], N)

        # accumulate transition counts
        for seq in state_seqs:
            for i in range(seq.shape[0] - 1):
                transition_counts[seq[i], seq[i+1]] += 1
        transition_counts += k # add k for Laplace smoothing

        return transition_counts / transition_counts.sum(axis=1, keepdims=True)


    def _estimate_state_dists(self, signals, tags):
        '''
        Estimate the word distributions of each state from the training data.
        For each state (tag), this function computes the relative proportions of each 
        word in the training data assigned that tag. 

        Parameters
        ----------
        signals : list
            A list of numpy arrays, each array corresponding to a sentence
        tags : list
            A list of numpy arrays, each array containing the tags of its corresponding
            sentence in `signals`

        Returns
        -------
        list
            A list of `DiscreteRV` objects, one for each state
        '''
        signals_flat = np.concatenate(signals)
        tags_flat = np.concatenate(tags)
        grouped_vals = {}
        state_dists = []
        for tag in self._unique_tags:
            grouped_vals[tag] = signals_flat[np.where(tags_flat == tag)[0]]

        # compute the estimated emission distribution of each state
        for tag in self._unique_tags:
            unique_tokens, counts = np.unique(grouped_vals[tag], return_counts=True)
            probs = counts / counts.sum()
            state_dists += [DiscreteRV(unique_tokens, probs)]

        return state_dists


    def _int_to_tag(self, val):
        '''Convert a tag\'s int code to its string representation'''
        return self._unique_tags[val]


    def _tag_to_int(self, tag):
        '''
        Convert a tag/'s string representation to its integer representation
        (working with integers makes some computations neater/easier).
        '''
        to_index = np.vectorize(lambda x: np.where(self._unique_tags == x)[0][0])
        return to_index(tag)

    
    def sample(self):
        '''
        Generate a new sentence from the learned parts-of-speech distribution.
        
        Essentially reverses the POS tagging process to generate sentences that follow the
        general tag structure learned by the model (in a probabilistic sense). However, some output 
        sentences may be nonsensical because this model only learns syntax structure, not context,
        verb agreement, or "meaning" in general.

        Returns
        -------
        str ndarray
            An array of tokens (words/punctuation that form the sampled sentence)
        str ndarray
            An array of parts-of-speech tags corresponding to the tokens in the above array
        '''
        samples = []
        states = []
        curr_state = -1

        while True:
            state_trans_probs = self.A[curr_state]
            # sample next state based on current transition probabilities
            curr_state = np.random.choice(len(self._unique_tags) + 1, p=state_trans_probs)
            if curr_state == len(self.state_dists):
                states = np.array(states)
                return np.array(samples), self._int_to_tag(states)
            states += [curr_state]
            samples += [self.state_dists[curr_state].sample()]

        return samples, states

