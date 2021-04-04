import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=1)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=1)


def onehot(y):
    assert len(y.shape) <= 2, 'Invalid response.'
    if len(y.shape) > 1:
        assert y.shape[1] == 1, 'Invalid response shape.'
    y = y.astype(np.int)
    y_oh = np.zeros((y.size, y.max() + 1))
    y_oh[np.arange(y.size), y.squeeze()] = 1

    return y_oh


class HME:
    def __init__(self, n_feature, n_expert1, n_expert2, n_level,
                 batch_size=32, lr=1., algo='gd', l1_coef=0.001,
                 l2_coef=0.001):
        assert algo in ['fista', 'gd'], f'Invalid algorihm: {algo}.'

        self.n_feature = n_feature
        self.n_expert1 = n_expert1
        self.n_expert2 = n_expert2
        self.n_level = n_level
        self.lr = lr
        self.algo = algo
        self.batch_size = batch_size
        self.l1_coef = l1_coef
        self.l21_coef = l2_coef

        self.data = None
        self.loc = 0
        self.init_param()

    def init_param(self):
        self.params = { }
        self.init_gate1()
        self.init_gate2()
        self.init_expert()
        self.state = None
        self.grads = None
        if self.algo == 'fista':
            self.theta = 1.    # For FISTA algorithm
            self.params_wo_momentum = self.params

    def init_gate1(self):
        # todo: should we add intercept in logistic regression?
        self.params['gamma1'] = np.random.normal(
            size=(self.n_expert1, self.n_feature, 1)) * .1

    def init_gate2(self):
        self.params['gamma2'] = np.random.normal(
            size=(self.n_expert2, self.n_expert1, self.n_feature, 1)) * .1

    def init_expert(self):
        # expert result: [>=K, >= K-1, ..., >= 1]
        self.params['expert_w'] = np.random.normal(
            size=(self.n_expert2, self.n_expert1, self.n_feature, 1)) * .1

        expert_b = np.random.uniform(
            size=(self.n_expert2, self.n_expert1, 1, self.n_level)) * .1
        expert_b = np.cumsum(expert_b, axis=-1)
        expert_b[..., -1] = 0.
        self.params['expert_b'] = expert_b

    def set_data(self, x, y):
        assert x.shape[1] == self.n_feature, 'Invalid feature number.'
        assert x.shape[0] == y.shape[0], 'Inconsistent sample number.'
        assert len(np.unique(y)) == self.n_level, 'Invalid level number.'
        self.data = { 'x': x, 'y': y.squeeze().astype(np.int) }

    def get_batch(self):

        start = self.loc
        end = min(self.loc + self.batch_size, self.data['x'].shape[0])

        x = self.data['x'][start:end, ...]
        y = self.data['y'][start:end, ...]

        if end == self.data['x'].shape[0]:
            self.shuffle_data()
            self.loc = 0
        else:
            self.loc = end

        return x, y

    def shuffle_data(self):
        idx = np.arange(self.data['x'].shape[0])
        np.random.shuffle(idx)
        self.data['x'] = self.data['x'][idx]
        self.data['y'] = self.data['y'][idx]

    def forward(self, params, x):
        score1 = x @ params['gamma1']
        g1 = softmax(score1, axis=0)
        score2 = x @ params['gamma2']
        g2 = softmax(score2, axis=0)

        cum_score = x @ params['expert_w'] + params['expert_b']
        cum_prob = sigmoid(cum_score)
        # cum_prob: [p(>=K+1), p(>= K-1), ..., p(>= 1)]
        cum_prob = np.c_[np.zeros(cum_prob.shape[:-1] + (1,)), cum_prob]
        cum_prob[..., -1] = 1.
        # lvl_prob: [p(=k), p(=k-1), ..., p(k=1)]
        lvl_prob = np.diff(cum_prob, axis=-1)
        prob = g1 * g2 * lvl_prob
        assert ((g1 * g2).sum(0) - g1).max() < 1e-10
        lvl_prob = lvl_prob.swapaxes(0, 1)
        prob = prob.swapaxes(0, 1)

        state = { 'score1': score1, 'g1': g1, 'score2': score2, 'g2': g2,
            'cum_score': cum_score, 'cum_prob': cum_prob, 'lvl_prob': lvl_prob,
            'prob': prob
            }
        return state  # self.state = state

    def backward(self, params, state, x, y):
        assert self.data is not None
        grads = { }
        self._update_grad_gamma1(grads, params, state, x, y)
        self._update_grad_gamma2(grads, params, state, x, y)
        self._update_grad_expert(grads, params, state, x, y)

        return grads

    def _update_grad_gamma1(self, grads, params, state, x, y):
        grad_gamma1 = np.zeros_like(params['gamma1'])
        for i, (gamma_i, g_i) in enumerate(zip(params['gamma1'], state['g1'])):
            dg_i = g_i * (1 - g_i) * x
            dg_i = dg_i.T[..., np.newaxis]
            factor_dg_i = state['lvl_prob'][i].sum((0), keepdims=1)
            dg_i = dg_i * factor_dg_i
            dg_ls = np.zeros_like(dg_i)
            for l in range(self.n_expert2):
                if l != i:
                    dg_l = g_i * state['g1'][l] * x
                    dg_l = dg_l.T[..., np.newaxis]
                    factor_dg_l = state['lvl_prob'][l].sum(0, keepdims=1)
                    dg_ls += dg_l * factor_dg_l
            grad_gi = (dg_i + dg_ls)[..., ::-1][..., np.arange(y.size), y]
            grad_gi = grad_gi.mean(-1, keepdims=True)
            assert grad_gi.shape == gamma_i.shape
            grad_gamma1[i] = grad_gi

        grads['gamma1'] = grad_gamma1

    def _update_grad_gamma2(self, grads, params, state, x, y):
        grad_gamma2 = np.zeros_like(self.params['gamma2'].swapaxes(0, 1))
        for i, (gamma_i, g_i) in enumerate(zip(params['gamma1'], state['g1'])):
            for j, (gamma_ij, g_ij) in enumerate(
                    zip(params['gamma2'].swapaxes(0, 1)[i],
                            state['g2'].swapaxes(0, 1)[i])):
                dg_ij = g_ij * (1 - g_ij) * x
                dg_ij = dg_ij.T[..., np.newaxis]
                factor_dg_ij = state['lvl_prob'][i, j]
                dg_ij = dg_ij * factor_dg_ij
                dg_ils = np.zeros_like(dg_ij)
                for l in range(self.n_expert2):
                    if l != j:
                        dg_il = g_ij * state['g1'][i, l] * x
                        dg_il = dg_il.T[..., np.newaxis]
                        factor_dg_il = state['lvl_prob'][i, l]
                        dg_ils += dg_il * factor_dg_il
                grad_gij = (dg_ij + dg_ils)[..., ::-1][
                    ..., np.arange(y.size), y]
                grad_gij = grad_gij.mean(-1, keepdims=True)
                assert grad_gij.shape == gamma_ij.shape
                grad_gamma2[i, j] = grad_gij

        grads['gamma2'] = grad_gamma2.swapaxes(0, 1)

    def _update_grad_expert(self, grads, params, state, x, y):
        self._update_grad_expert_w(grads, params, state, x, y)
        self._update_grad_expert_b(grads, params, state, x, y)

    def _update_grad_expert_w(self, grads, params, state, x, y):
        grad_w = np.zeros_like(params['expert_w'].swapaxes(0, 1))
        for i, (gamma_i, g_i) in enumerate(zip(params['gamma1'], state['g1'])):
            for j, (w_ij, g_ij, sig_ij) in enumerate(
                    zip(params['expert_w'].swapaxes(0, 1)[i],
                        state['g2'].swapaxes(0, 1)[i],
                        state['cum_prob'].swapaxes(0, 1)[i])):
                # dw_ij: [K, ..., 1]
                dw_ij = np.diff(sig_ij * (1 - sig_ij))
                factor_dw_ij = g_i * g_ij
                grad_wij = (dw_ij * factor_dw_ij)[..., ::-1][
                    ..., np.arange(y.size), y]
                grad_wij = grad_wij[..., np.newaxis] * x
                grad_wij = grad_wij.T.mean(-1, keepdims=True)
                assert grad_wij.shape == w_ij.shape
                grad_w[i, j] = grad_wij
        grads['expert_w'] = grad_w.swapaxes(0, 1)

    def _update_grad_expert_b(self, grads, params, state, x, y):
        grad_b = np.zeros_like(params['expert_b'].swapaxes(0, 1))
        for i, (gamma_i, g_i) in enumerate(zip(params['gamma1'], state['g1'])):
            for j, (b_ij, g_ij, sig_ij) in enumerate(
                    zip(params['expert_b'].swapaxes(0, 1)[i],
                        state['g2'].swapaxes(0, 1)[i],
                        state['cum_prob'].swapaxes(0, 1)[i])):
                # dw_ij: [K, ..., 1]
                _db_ij_ = sig_ij * (1 - sig_ij)
                # The first column is the (K+1)
                db_ij_k = _db_ij_[..., 1:]

                db_ij_kn = -_db_ij_[..., :-1]
                db_ij = np.array([db_ij_kn, db_ij_k])

                factor_db_ij = g_i * g_ij

                grad_bij_ = (db_ij * factor_db_ij)[..., ::-1]
                grad_bij_val = grad_bij_[..., np.arange(y.size), y].T
                grad_bij = np.zeros(shape=(y.size, self.n_level + 1))
                # Store b_k and b_k+1 gradient
                grad_bij[np.arange(y.size), y] = grad_bij_val[:, 1]
                grad_bij[np.arange(y.size), y + 1] = grad_bij_val[:, 0]
                # Flip graded axis and drop the K+1 column
                grad_bij = grad_bij[..., ::-1][..., 1:]
                grad_bij = grad_bij.mean(0, keepdims=True)
                assert grad_bij.shape == b_ij.shape
                grad_b[i, j] = grad_bij

        grads['expert_b'] = grad_b.swapaxes(0, 1)

    def predict(self, params=None, x=None):
        if params is None:
            params = self.params
        state = self.forward(params, x)
        prob = state['prob'][..., ::-1].sum((0, 1))
        return prob.argmax(-1)[..., np.newaxis]

    def loss(self, params, x=None, y=None, state=None):
        if x is None:
            x = self.data['x']
        if y is None:
            y = self.data['y']
        if state is None:
            state = self.forward(params=params, x=x)

        prob = state['prob']
        prob = np.clip(prob, a_min=1e-10, a_max=1.)
        llk = np.log(prob.sum((0, 1)))
        llk = llk[..., ::-1][..., np.arange(y.size), y].mean()
        norm = 0.
        for param_name, param in params.items():
            norm += np.abs(param).sum()

        return norm * self.l1_coef, -llk

    def zero_grad(self):
        self.grads = None

    def zero_state(self):
        self.state = None

    def lasso_proximalty(self, param, threshold):
        tmp_param = np.zeros_like(param)
        tmp_param[param > threshold] = param[param > threshold] - threshold
        tmp_param[param < -threshold] = param[param <- threshold] + threshold
        return tmp_param

    def update(self, grads, params, state, x, y):
        if self.algo == 'fista':
            return self._update_fista(grads, params, state, x, y)
        if self.algo == 'gd':
            return self._update_gd(grads, params, state, x, y)
        # self.lr *= 0.999

    def _update_fista(self, grads, params, state, x, y):
        lr = self.lr
        l1_lambda = self.l1_coef
        l21_lambda = self.l21_coef

        _, f_old = self.loss(self.params_wo_momentum, x, y)

        tmp_params_wo_momentum = {}
        feat_coef_normsq = np.zeros((self.n_feature, 1))
        for param_name, param in params.items():
            grad = -grads[param_name]
            param_wo_momentum = self.params_wo_momentum[param_name]
            assert grad.shape == param.shape == param_wo_momentum.shape
            new_param = self.lasso_proximalty(param - lr * grad, lr * l1_lambda)
            tmp_params_wo_momentum[param_name] = new_param

            if param_name != 'expert_b':
                summand = np.arange(len(new_param.shape))
                feat_coef_normsq += (new_param ** 2).sum(tuple(summand[:-2]))
        feat_glasso_coef = l21_lambda * lr / np.sqrt(feat_coef_normsq)
        feat_glasso_coef = np.clip(1 - feat_glasso_coef, a_min=0., a_max=None)

        tmp_params = {}
        for param_name, param in tmp_params_wo_momentum.items():
            if param_name != 'expert_b':
                param = param * feat_glasso_coef
                tmp_params_wo_momentum[param_name] = param
            param_wo_momentum = self.params_wo_momentum[param_name]
            momentum = (self.theta - 1) / (self.theta + 1) * \
                       (param - param_wo_momentum)
            tmp_params[param_name] = param + momentum

        _, f_new = self.loss(tmp_params_wo_momentum, x, y)
        if f_new < f_old:
            self.params_wo_momentum = tmp_params_wo_momentum
            self.theta = (1 + np.sqrt(1 + 4 * self.theta ** 2)) / 2
        else:
            self.params_wo_momentum = tmp_params
            self.theta = 1.

        return tmp_params

    def _update_gd(self, grads, params, state, x, y):
        lr = self.lr
        l1_lambda = self.l1_coef
        l21_lambda = self.l21_coef

        tmp_params = { }
        feat_coef_normsq = np.zeros((self.n_feature, 1))
        for param_name, param in params.items():
            grad = -grads[param_name]
            assert grad.shape == param.shape
            new_param = self.lasso_proximalty(param - lr * grad, lr * l1_lambda)
            tmp_params[param_name] = new_param
            if param_name != 'expert_b':
                summand = np.arange(len(new_param.shape))
                feat_coef_normsq += (new_param ** 2).sum(tuple(summand[:-2]))

        feat_glasso_coef = l21_lambda * lr / np.sqrt(feat_coef_normsq)
        feat_glasso_coef = np.clip(1 - feat_glasso_coef, a_min=0., a_max=None)
        for param_name, param in tmp_params.items():
            if param_name != 'expert_b':
                tmp_params[param_name] = param * feat_glasso_coef

        return tmp_params

    def fit(self, x, y, max_iter=100, silent=False, stop_thre=np.inf):
        self.set_data(x, y)
        self.init_param()
        prev_accu = - np.inf
        stop = 0
        log_interval = 500

        self.theta = 1.

        params = self.params
        for i in range(1, max_iter + 1):
            x_batch, y_batch = self.get_batch()
            state = self.forward(params, x_batch)
            grads = self.backward(params, state, x_batch, y_batch)
            params = self.update(grads, params, state, x_batch, y_batch)

            if i % log_interval == 0:
                y_pred = self.predict(params, x)
                accu = (y_pred == y).mean()
                l1loss, llk_loss = self.loss(params=params)
                if accu > prev_accu:
                    prev_accu = accu
                    stop = 0
                else:
                    stop += 1
                if not silent:
                    print(f'[accu: {accu}], [loss: {l1loss}, [{llk_loss}]')

                if stop > stop_thre:
                    print(f'stop increasing accuracy at iter: {i}')
                    break

        self.params = params
