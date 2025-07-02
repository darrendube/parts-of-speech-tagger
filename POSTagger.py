import numpy as np
np.seterr(divide="ignore") # TODO: check if this is even necessary


# XXX: NOTE that this case (parts-of-speech tagging) is a supervised problem, 
# unlike the signature assignmetn in CS315 which was unsurpervied. in POS tagging,
# we are given the tags, so we can speed up training by simply counting transitions and
# estimating emission probabilities from the data. We can then load this information
# through some function like load(). So the main point of this class is to find the most
# likely state sequence for an unknown, test sequence, using viterbi
#
# XXX: in the load phase, remember to smooth after counting - in real data some 
# state transitions or word emission smay never appear in training, resulting in 
# a zero count, which makes porbabilities zero and thus certain paths impossible
# (even though they may be possible in real life). Avoid this by adding a small value 
# k to counts before normlisation 
#

class DiscreteRV:
    def __init__(self, tokens:ndarray, probabilities:ndarray):
        assert tokens.shape[0] == probabilities.shape[0], "Mismatch of lengths"
        assert abs(sum(probabilities) - 1.0) < 1e-7, "Probabilities \
         do not add up to 1"
        self.tokens = tokens
        self.probabilities = probabilities
        self._dict = dict(zip(self.tokens, self.probabilities))

    def pmf(self, token):
        return self._dict.get(token, 0.0)

    def sample(self, n=1):
        return np.random.choice(self.tokens, size=n, p=self.probabilities)


class POSTagger:
    def __init__(self, n_states):
        self.n_states = n_states
        # TODO: just initialise some local params e.g. max number of iterations, 
        # error teolerance for viterbi decoding
    
    def fit(self, signals, tags):
        '''
        Train the POS Tagger using the training data provided in `signal` and 
        `tags`.

        Parameters
        ----------
        signals : str ndarray
            An array of training data, consisting of several different training
            sequences. Columns correspond to individual sequences

        tags : str ndarray
            An array of tags corresponding to each token in `signal`
        '''
        # TODO: count tag transitions and estimate state emission probabilities
        # from training data
        self._unique_tags = np.unique(tags)
        self.A = POSTagger._estimate_trans_matrix(tags)
        self.state_dists = POSTagger._estimate_state_dists(signals, tags)
        

    def load_hmm(self, transition_matrix, state_dists):
        self.A = transition_matrix
        self.state_dists = state_dists

    def get_tags(self, signal:np.ndarray):
        # TODO: store tags in an array then simply work with ints
        np.seterr(invalid="ignore") # XXX: check if this is even necessary
        N = len(self.state_dists)
        T = signal.shape[0]
        
        Delt = np.zeros((N, T))
        Backp = np.zeros((N, T), dtype=int)

        # initialisation
        for j in range(N):
            Delt[j,0] = np.log(self.A[-1,j]) + self.state_dists[j].prob(signal[0])
            Backp[j,0] = -1

        # recursion
        for t in range(1,T):
            for j in range(N):
                vals = []
                for i in range(N):
                    vals += [Delt[i,t-1] + np.log(self.A[i,j])]
                Delt[j, t] = self.state_dists[j].prob(signal[t]) + np.max(vals)
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

        return self._int_to_tag(np.array(seq)[::-1]])

    @staticmethod
    def _estimate_trans_matrix(tags):
        N = len(self.state_dists)
        k = 1 # k value for smoothing
        state_seqs = self._tag_to_int(tags)

        transition_counts = np.zeros((N+1, N+1))

        # pad each state sequence with 
        for i in range(tags.shape[1]):
            state_seqs[:,i] = np.insert(state_seqs[:,i], 0, -1)
            state_seqs[:,i] = np.append(state_seqs, N)

        for seq in state_seqs.T:
            for i in range(seq.shape[0] - 1):
                transition_counts[seq[i], seq[i+1]] += 1
        transition_counts += k # add smoothing k

        return transition_counts / transition_counts.sum(axis=1, keepdims=True)



    @staticmethod
    def _estimate_state_dists(signals, tags):
        # TODO: collect an array of tokens for each tag, then calculate 
        # probabilities of words based on relative proportions, plus maybe a 
        # small smoothing value k???
        pass

    def _int_to_tag(val):
        return self._unique_tags[val]

    def _tag_to_int(tag):
        return np.where(self._unique_tags == tag)[0][0]


