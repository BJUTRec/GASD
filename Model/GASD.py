import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utility.helper import *
from utility.batch_test import *
from tqdm import tqdm
from utility.math_utils import *


class GASD(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

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

        self.skd_temp = args.skd_temp
        self.skd_reg = args.skd_reg

        self.manifold = Poincare()

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
        self.mf_loss, self.skd_loss, self.label_loss = self.create_loss(self.u_g_embeddings1, self.u_g_embeddings2, self.pos_i_g_embeddings1, self.pos_i_g_embeddings2, self.neg_i_g_embeddings1, self.neg_i_g_embeddings2)

        self.loss = self.mf_loss + self.skd_loss + self.label_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

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
        Hyper_embeddings = self.manifold.expmap0(Hyper_embeddings, 1)

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

        hyper_pos_scores = 1 / self.manifold.sqdist(user2, pos_item2, 1)
        hyper_neg_scores = 1 / self.manifold.sqdist(user2, neg_item2, 1)

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
        pos_label = tf.multiply(tf.nn.sigmoid(lamda*hyper_pos_scores/self.skd_temp), tf.log(tf.nn.sigmoid(pos_scores/self.skd_temp)+1e-8))
        neg_label = tf.multiply(tf.nn.sigmoid((1-lamda*hyper_neg_scores)/self.skd_temp), tf.log(tf.nn.sigmoid((1-neg_scores)/self.skd_temp)+1e-8))
        skd_loss = self.skd_reg*(-tf.reduce_mean(pos_label + neg_label))

        hyper_pos_scores_l = self.manifold.sqdist(user2, pos_item2, 1)
        hyper_neg_scores_l = self.manifold.sqdist(user2, neg_item2, 1)
        gap = tf.nn.relu((hyper_pos_scores_l - hyper_neg_scores_l + 1.0))  #best=1.0 useless
        label_loss = 0.00 * tf.reduce_mean(gap) #best == 0.05 useless

        return mf_loss, skd_loss, label_loss

########################################################################################################################    Math
    def model_save(self, ses):
        save_pretrain_path = '../output_parameters/ml1m'
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
    pretrain_path = "../pretrain_parameters/ml1m_emb.npz"
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


    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
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

    model = GASD(data_config=config, pretrain_data=pretrain_data)

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
        loss, mf_loss, emb_loss, skd_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in tqdm(range(n_batch)):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_skd_loss = sess.run([model.opt, model.loss, model.mf_loss, model.skd_loss],
                                                                                    feed_dict={model.users: users,
                                                                                               model.pos_items: pos_items,
                                                                                               model.neg_items: neg_items})
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            skd_loss += batch_skd_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            print(mf_loss, emb_loss, skd_loss)
            sys.exit()

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, skd_loss)
            print(perf_str)

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.show_step != 0:
            #if args.verbose > 0 and epoch % args.verbose == 0:
            # Skip testing
            continue

        # Begin test at this epoch.
        loss_test, mf_loss_test, emb_loss_test, skd_loss_test = 0., 0., 0., 0.
        for idx in tqdm(range(n_batch)):
            users, pos_items, neg_items = data_generator.sample_test()
            batch_loss_test, batch_mf_loss_test, batch_skd_loss_test = sess.run(
                [model.loss, model.mf_loss, model.skd_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items})

            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            skd_loss_test += batch_skd_loss_test / n_batch

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
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, skd_loss_test, ret['recall'][0],
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
