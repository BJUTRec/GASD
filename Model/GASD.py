import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import os
import sys
import random as rd
import pickle
import scipy.sparse as sp
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utility.helper import *
from utility.batch_test import *
from tqdm import tqdm


class Mymodel(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        # self.training_user = data_config['trainUser']
        # self.training_item = data_config['trainItem']

        self.n_fold = 100
        self.plain_adj = data_config['plain_adj']  #A
        self.norm_adj = data_config['norm_adj']    #D-1/2*A*D-1/2

        self.A_in_shape = self.plain_adj.tocoo().shape
        self.n_nonzero_elems = self.plain_adj.count_nonzero()
        print(self.n_nonzero_elems)

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = eval(args.node_dropout)
        self.mess_dropout = eval(args.mess_dropout)

        self.ssl_ratio = args.ssl_ratio
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg

        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {tf.float32: 4e-3, tf.float64: 1e-5}
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        # tf.compat.v1.disable_eager_execution()
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        # create models
        self.ua_embeddings, self.ia_embeddings, self.hyper_ua_embeddings, self.hyper_ia_embeddings = self.GASD()
        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings1 = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.u_g_embeddings2 = tf.nn.embedding_lookup(self.hyper_ua_embeddings, self.users)

        self.pos_i_g_embeddings1 = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.pos_i_g_embeddings2 = tf.nn.embedding_lookup(self.hyper_ia_embeddings, self.pos_items)

        self.neg_i_g_embeddings1 = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.neg_i_g_embeddings2 = tf.nn.embedding_lookup(self.hyper_ia_embeddings, self.neg_items)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        # Inference for the testing phase.
        self.batch_ratings = tf.matmul(self.u_g_embeddings1, self.pos_i_g_embeddings1, transpose_a=False, transpose_b=True)

        # Generate Predictions & Optimize via BPR loss.
        self.mf_loss, self.ssl_loss, self.label_loss = self.create_loss(self.u_g_embeddings1, self.u_g_embeddings2, self.pos_i_g_embeddings1, self.pos_i_g_embeddings2, self.neg_i_g_embeddings1, self.neg_i_g_embeddings2)

        self.loss = self.mf_loss + self.ssl_loss + self.label_loss #+ self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        #initializer = tf.keras.initializers.glorot_normal()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')

        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True, name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True, name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        return all_weights

    def GASD(self):
        adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = []

        for k in range(0, self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        Hyper_embeddings = all_embeddings
        Hyper_embeddings = self.expmap0(Hyper_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        hyper_u_g_embeddings, hyper_i_g_embeddings = tf.split(Hyper_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings, hyper_u_g_embeddings, hyper_i_g_embeddings

    def create_loss(self, user1, user2, pos_item1, pos_item2, neg_item1, neg_item2):
        pos_scores = tf.reduce_sum(tf.multiply(user1, pos_item1), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(user1, neg_item1), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # mf_loss = tf.negative(tf.reduce_mean(maxi))

        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        ## In the third version,correct and high effitiveness
        #mf_loss = tf.reduce_sum(-tf.log_sigmoid(pos_scores - neg_scores))
        emb_loss = self.decay * regularizer
        mf_loss += emb_loss

        hyper_pos_scores = 1 / self.sqdist(user2, pos_item2, 1)
        hyper_neg_scores = 1 / self.sqdist(user2, neg_item2, 1)

        #####################adaptive###################
        score = tf.concat([pos_scores, neg_scores], axis=0)
        hyper_score = tf.concat([hyper_pos_scores, hyper_neg_scores], axis=0)

        max_score_idx = tf.argmax(score)
        max_hyper_score_idx = tf.argmax(hyper_score)
        min_score_idx = tf.argmin(score)
        min_hyper_score_idx = tf.argmin(hyper_score)

        max_score = tf.nn.embedding_lookup(score, max_score_idx)
        max_score_hyper = tf.nn.embedding_lookup(hyper_score, max_hyper_score_idx)
        min_score = tf.nn.embedding_lookup(score, min_score_idx)
        min_score_hyper = tf.nn.embedding_lookup(hyper_score, min_hyper_score_idx)

        lamda = max_score-min_score / max_score_hyper-min_score_hyper
        ###################################################################
        pos_label = tf.multiply(tf.nn.sigmoid(lamda*hyper_pos_scores/self.ssl_temp), tf.log(tf.nn.sigmoid(pos_scores/self.ssl_temp)+1e-8))
        neg_label = tf.multiply(tf.nn.sigmoid((1-lamda*hyper_neg_scores)/self.ssl_temp), tf.log(tf.nn.sigmoid((1-neg_scores)/self.ssl_temp)+1e-8))
        kd_loss = self.ssl_reg*(-tf.reduce_mean(pos_label + neg_label))

        hyper_pos_scores_l = self.sqdist(user2, pos_item2, 1)
        hyper_neg_scores_l = self.sqdist(user2, neg_item2, 1)
        gap = tf.nn.relu((hyper_pos_scores_l - hyper_neg_scores_l + 1.0))  #best=1.0
        label_loss = 0.00 * tf.reduce_mean(gap) #best == 0.05

        return mf_loss, kd_loss, label_loss
########################################################################################################################   Hyperbolic
    def safe_norm(self, x, eps=1e-12, axis=None, keep_dims=False):
        # return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis, keep_dims=keep_dims)+eps)
        return tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=axis, keep_dims=keep_dims) + eps)

    def minkowski_dot(self, x, y, c, keepdim=True):
        hyp_norm_x = tf.sqrt(self.sqdist(x, tf.zeros_like(x), c))
        hyp_norm_y = tf.sqrt(self.sqdist(y, tf.zeros_like(y), c))
        norm_mul = tf.multiply(hyp_norm_x, hyp_norm_y)
        denominator = tf.multiply(tf.norm(x, axis=-1), tf.norm(y, axis=-1))
        denominator = tf.clip_by_value(denominator, clip_value_min=self.eps[denominator.dtype],
                                           clip_value_max=tf.reduce_max(denominator))
        cos_theta = tf.reduce_sum(tf.multiply(x, y), axis=-1) / denominator
        res = tf.multiply(norm_mul, cos_theta)
        res = tf.reshape(res, [-1])
        if keepdim:
           res = tf.expand_dims(res, -1)
        return res

    def minkowski_norm(self, u, c, keepdim=True):
        dot = self.minkowski_dot(u, u, c, keepdim=keepdim)
        return tf.sqrt(tf.clip_by_value(dot, clip_value_min=self.eps[u.dtype], clip_value_max=tf.reduce_max(dot)))

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = self.artanh(sqrt_c * self.safe_norm(self.mobius_add(-p1, p2, c, dim=-1), axis=-1, keep_dims=False))
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = tf.reduce_sum(tf.pow(x, 2), axis=-1, keep_dims=True)
        result = 2 / (1. - c * x_sqnorm)
        max_value = tf.reduce_max(result)
        return tf.clip_by_value(result, clip_value_min=self.min_norm, clip_value_max=max_value)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= tf.pow(lambda_p, 2)
        return dp

    def proj(self, x, c):
        norm_x = self.safe_norm(x, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(norm_x)
        norm = tf.clip_by_value(norm_x, clip_value_min=self.min_norm, clip_value_max=max_value)
        maxnorm = (1 - 4e-3) / (c ** 0.5)
        cond = norm > maxnorm
        if len(x.shape) == 4:
            cond = tf.tile(cond, [1, 1, 1, x.shape[3]])
        elif len(x.shape) == 3:
            cond = tf.tile(cond, [1, 1, x.shape[2]])
        elif len(x.shape) == 2:
            cond = tf.tile(cond, [1, x.shape[1]])
        elif len(x.shape) == 1:
            cond = tf.tile(cond, [x.shape[0]])
        else:
            raise ValueError('invalid shape!')
        projected = x / norm * maxnorm
        result = tf.where(cond, projected, x)
        return result

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = self.safe_norm(u, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(u_norm)
        u_norm = tf.clip_by_value(u_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        second_term = (self.tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm) * u / (sqrt_c * u_norm))
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = self.safe_norm(sub, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(sub_norm)
        sub_norm = tf.clip_by_value(sub_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        result = 2 / sqrt_c / lam * self.artanh(sqrt_c * sub_norm) * sub / sub_norm
        return result

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = self.safe_norm(u, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(u_norm)
        u_norm = tf.clip_by_value(u_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        gamma_1 = self.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return self.proj(gamma_1, c)

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = self.safe_norm(p, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(p_norm)
        p_norm = tf.clip_by_value(p_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        scale = 1. / sqrt_c * self.artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = tf.reduce_sum(tf.pow(x, 2), axis=dim, keep_dims=True)
        y2 = tf.reduce_sum(tf.pow(y, 2), axis=dim, keep_dims=True)
        xy = tf.reduce_sum(x * y, axis=dim, keep_dims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        max_value = tf.reduce_max(denom)
        denom = tf.clip_by_value(denom, clip_value_min=self.min_norm, clip_value_max=max_value)
        result = num / denom
        return result

    def mobius_matvec(self, x, m, c, sparse=False):
        sqrt_c = c ** 0.5
        x_norm = self.safe_norm(x, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(x_norm)
        x_norm = tf.clip_by_value(x_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        if sparse:
            mx = tf.sparse_tensor_dense_matmul(x, m)
        else:
            mx = tf.matmul(x, m)
        mx_norm = self.safe_norm(mx, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(mx_norm)
        mx_norm = tf.clip_by_value(mx_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        res_c = self.tanh(mx_norm / x_norm * self.artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = tf.reduce_all(tf.equal(mx, 0), axis=-1, keep_dims=True)
        cond = tf.tile(cond, [1, res_c.shape[1]])
        res_0 = tf.zeros_like(res_c)
        res = tf.where(cond, res_0, res_c)
        return self.proj(res, c)

    def _gyration(self, u, v, w, c, dim=-1):
        u2 = tf.reduce_sum(tf.pow(u, 2), axis=dim, keep_dims=True)
        v2 = tf.reduce_sum(tf.pow(v, 2), axis=dim, keep_dims=True)
        uv = tf.reduce_sum(u * v, axis=dim, keep_dims=True)
        uw = tf.reduce_sum(u * w, axis=dim, keep_dims=True)
        vw = tf.reduce_sum(v * w, axis=dim, keep_dims=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        max_value = tf.reduce_max(d)
        d = tf.clip_by_value(d, clip_value_min=self.min_norm, clip_value_max=max_value)
        result = w + 2 * (a * u + b * v) / d
        return result

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
           v = u
        lambda_x = self._lambda_x(x, c)
        uv = tf.reduce_sum(u * v, axis=-1, keep_dims=keepdim)
        return lambda_x ** 2 * uv

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        result = self._gyration(y, -x, u, c) * lambda_x / lambda_y
        return result

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        result = self._gyration(y, -x, u, c) * lambda_x / lambda_y
        return result

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        max_value = tf.reduce_max(lambda_x)
        lambda_x = tf.clip_by_value(lambda_x, clip_value_min=self.min_norm, clip_value_max=max_value)
        result = 2 * u / lambda_x
        return result

    def to_hyperboloid(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        sqnorm = self.safe_norm(x, axis=1, keep_dims=True) ** 2
        return sqrtK * tf.concat([K + sqnorm, 2 * sqrtK * x], axis=1) / (K - sqnorm)

########################################################################################################################   Hyperbolic
########################################################################################################################   Math
    def cosh(self, x, clamp=15):
        return tf.cosh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

    def sinh(self, x, clamp=15):
        return tf.sinh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

    def tanh(self, x, clamp=15):
        return tf.tanh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))

    def arcosh(self, x):
        max_value = tf.reduce_max(x)
        if x.dtype == tf.float32:
            max_value = tf.cond(max_value < 1.0 + 1e-7, lambda: 1.0 + 1e-7, lambda: max_value)
            result = tf.acosh(tf.clip_by_value(x, clip_value_min=1.0 + 1e-7, clip_value_max=max_value))  # 1e38
        elif x.dtype == tf.float64:
            max_value = tf.cond(max_value < 1.0 + 1e-16, lambda: 1.0 + 1e-16, lambda: max_value)
            result = tf.acosh(tf.clip_by_value(x, clip_value_min=1.0 + 1e-16, clip_value_max=max_value))
        else:
            raise ValueError('invalid dtype!')
        return result

    def arsinh(self, x):
        result = tf.asinh(x)
        return result

    def artanh(self, x):
        if x.dtype == tf.float32:
            result = tf.atanh(tf.clip_by_value(x, clip_value_min=tf.constant([-1], dtype=tf.float32) + 1e-7,
                                                   clip_value_max=tf.constant([1], dtype=tf.float32) - 1e-7))
        elif x.dtype == tf.float64:
            result = tf.atanh(tf.clip_by_value(x, clip_value_min=tf.constant([-1], dtype=tf.float64) + 1e-16,
                                                   clip_value_max=tf.constant([1], dtype=tf.float64) - 1e-16))
        else:
            raise ValueError('invalid dtype!')
        return result

########################################################################################################################    Math
    def model_save(self, ses):
        save_pretrain_path = 'D:\wt\SSL-mycode\pretrain\gowalla_0.5'
        np.savez(save_pretrain_path, user_embed=np.array(self.weights['user_embedding'].eval(session=ses)),
                 item_embed=np.array(self.weights['item_embedding'].eval(session=ses)))

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

def load_best(name="best_model"):
    #pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, name)
    pretrain_path = "D:\wt\DGCF复现\pretain\yelp2018-layer1-64.npz"
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the best model:', name)
    except Exception:
        pretrain_data = None
    return pretrain_data

def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)
    return all_h_list, all_t_list, all_v_list


if __name__ == '__main__':
    whether_test_batch = True
    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    print("************************************************************************************")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    # config['trainUser'] = data_generator.trainUser
    # config['trainItem'] = data_generator.trainItem

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    #all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)
    #u_u_mat, i_i_mat = data_generator.get_ii_uu_mat()

    config['plain_adj'] = plain_adj   #A
    config['norm_adj'] = pre_adj      #D-1/2*A*D-1/2


    t0 = time()
    """
    *********************************************************
    pretrain = 1: load embeddings with name such as embedding_xxx(.npz), l2_best_model(.npz)
    pretrain = 0: default value, no pretrained embeddings.
    """
    if args.pretrain == 1:
        print("Try to load pretain: ", args.embed_name)
        pretrain_data = load_best(name=args.embed_name)
        if pretrain_data == None:
            print("Load pretrained model(%s)fail!!!!!!!!!!!!!!!" % (args.embed_name))
    else:
        pretrain_data = None

    model = Mymodel(data_config=config, pretrain_data=pretrain_data)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, ssl_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in tqdm(range(n_batch)):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_ssl_loss = sess.run([model.opt, model.loss, model.mf_loss, model.ssl_loss],
                                                                                    feed_dict={model.users: users,
                                                                                               model.pos_items: pos_items,
                                                                                               model.neg_items: neg_items})
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            #emb_loss += batch_emb_loss / n_batch
            ssl_loss += batch_ssl_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            print(mf_loss, emb_loss, ssl_loss)
            sys.exit()

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, ssl_loss)
            print(perf_str)

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.show_step != 0:
            #if args.verbose > 0 and epoch % args.verbose == 0:
            # Skip testing
            continue

        # Begin test at this epoch.
        loss_test, mf_loss_test, emb_loss_test, ssl_loss_test = 0., 0., 0., 0.
        for idx in tqdm(range(n_batch)):
            users, pos_items, neg_items = data_generator.sample_test()
            batch_loss_test, batch_mf_loss_test, batch_ssl_loss_test = sess.run(
                [model.loss, model.mf_loss, model.ssl_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items})

            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            #emb_loss_test += batch_emb_loss_test / n_batch
            ssl_loss_test += batch_ssl_loss_test / n_batch

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True, batch_test_flag=whether_test_batch)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, ssl_loss_test, ret['recall'][0],
                       ret['recall'][-1],
                       ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                       ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=args.early)
        # early stopping when cur_best_pre_0 is decreasing for given steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            model.model_save(sess)
            print('save the model with performance: ', cur_best_pre_0)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
