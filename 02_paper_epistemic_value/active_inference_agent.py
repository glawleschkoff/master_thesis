import numpy as np
from src.math_utils import softmax, safelog


class ActiveInferenceAgent:
    def __init__(self, model, alpha, beta, T, pi_gamma_loop_steps = 8):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.pi_gamma_loop_steps = pi_gamma_loop_steps

        self.reset()

    def reset(self):
        self.s_hat_t_minus_1 = safelog(self.model.D)
        self.a_t_minus_1 = None
        self.gamma_hat = self.alpha / self.beta

    def _s_hat_t(self, o_t, t):
        log_likelihood = safelog(self.model.A.T) @ o_t
        if t == 0:
            log_prior = safelog(self.model.D)
        else:
            log_prior = safelog(self.model.B(self.a_t_minus_1) @ self.s_hat_t_minus_1)

        return softmax(log_likelihood + log_prior)

    def _s_hat_tau(self, s_hat_t, u_seq):
        s_hat_tau = s_hat_t.copy()
        for u in u_seq:
            s_hat_tau = self.model.B(u) @ s_hat_tau

        return s_hat_tau

    def _o_hat_tau(self, s_hat_tau):
        return self.model.A @ s_hat_tau

    def _Q_tau(self, s_hat_tau, o_hat_tau):
        predicted_uncertainty = np.sum(
            (self.model.A * safelog(self.model.A)) @ s_hat_tau
        )
        predicted_divergence = (
            safelog(o_hat_tau) - safelog(self.model.C)
        ).T.flatten() @ o_hat_tau.flatten()

        return predicted_uncertainty - predicted_divergence

    def _Q(self, s_hat_t, policies_t, t):
        Q = []
        for pi in policies_t:
            Q_pi = 0.0
            for tau in range(t + 1, self.T):
                u_seq = pi[0 : tau - t]
                s_hat_tau = self._s_hat_tau(s_hat_t, u_seq)
                o_hat_tau = self._o_hat_tau(s_hat_tau)
                Q_tau = self._Q_tau(s_hat_tau, o_hat_tau)
                Q_pi += Q_tau
            Q.append(Q_pi)

        return np.array(Q)

    def _pi_hat(self, Q):
        return softmax(self.gamma_hat * Q)

    def _gamma_hat(self, Q, pi_hat):
        return self.alpha / (self.beta - Q @ pi_hat)

    def infer_action(self, o_t, policies_t, t):
        s_hat_t = self._s_hat_t(o_t, t)
        Q = self._Q(s_hat_t, policies_t, t)
        pi_hat = None
        for _ in range(self.pi_gamma_loop_steps):
            pi_hat = self._pi_hat(Q)
            self.gamma_hat = self._gamma_hat(Q, pi_hat)
        pi_hat_stable = np.nan_to_num(pi_hat, nan=1.0 / len(pi_hat))
        pi_hat_stable /= np.sum(pi_hat_stable)
        pi_index = np.random.choice(len(pi_hat_stable), p=pi_hat_stable)
        if not policies_t[pi_index]:
            a_t = None
        else:
            a_t = policies_t[pi_index][0]
        self.s_hat_t_minus_1 = s_hat_t.copy()
        self.a_t_minus_1 = a_t
        
        return a_t
