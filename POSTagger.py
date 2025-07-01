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
# k to counts before normlisation§
#

class POSTagger:
    def __init__(self, n_states):
        self.n_states = n_states
        # TODO: just initialise some local params e.g. max number of iterations, 
        # error teolerance for viterbi decoding

    def load_hmm(self, transition_matrix, state_dists):
        self.A = transition_matrix
        self.state_dists = state_dists

    def viterbi(self, signal:np.ndarray):
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

        return np.array(seq)[::-1]



    @staticmethod
    def _init_transition_matrix(n_states):
        pass
