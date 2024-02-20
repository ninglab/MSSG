import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import bottleneck as bn
import pdb
import math
import time
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname, isTrain):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            if isTrain == 0:
                user_train[user] = User[user][:-1]
                user_test[user] = User[user][-1:]
            else:
                user_train[user] = User[user][:-2]
                user_test[user] = User[user][-2:-1]
    return [user_train, user_valid, user_test, usernum, itemnum]

def batch_evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    valid_user = 0
    computing_time = 0

    atts_all   = []
    st_all     = []
    lt_all     = []
    hiddens_all= []

    num_batch  = len(train) // args.batch_size if (len(train) % args.batch_size) == 0 else (len(train) // args.batch_size) + 1
    Recalls = {}
    NDCGs = {}

    def data_prep(train):
        seq = []
        for i in range(1, len(train)+1):
            if len(train[i]) < args.maxlen:
                seq.append([0] * (args.maxlen - len(train[i])) + train[i])
            else:
                seq.append(train[i][-args.maxlen:]) 

        return np.asarray(seq).astype(int)

    def sample_batch(seq, test, batch_idx, batch_size=args.batch_size):
        start = batch_idx * batch_size
        end = min((batch_idx+1) * batch_size, len(seq))
        batch_seq = seq[start:end]
        batch_users = np.arange(start, end) + 1

        return batch_seq, batch_users, start

    
    seq = data_prep(train)
    for batch_idx in range(num_batch):
        batch_seq, batch_users, start = sample_batch(seq, test, batch_idx)
        bt = time.time()
        #atts, st, lt and hiddens are used for analysis
        predictions, atts, st, lt, hiddens = model.predict(batch_users, batch_seq)
        computing_time += (time.time() - bt)
        predictions = -predictions.cpu().detach().numpy()

        #mask 0 and existing items as that in the literature
        for i in range(len(batch_seq)):
            predictions[i, np.array(train[i+start+1]+[0])] = 1000

        predictions[:,0] = 1000
        
        if batch_idx == 0:
            pred = predictions
        else:
            pred = np.append(pred, predictions, axis=0)

    for k in [5, 10, 20]:

        pred_idx = np.argsort(pred, axis=-1)[:,:k]
        Recall = 0.0
        NDCG = 0.0
        recalls, ndcgs = [], []
        
        for u in range(len(test)):
            #u in test starts from 0
            set_truth = set(test[u+1])
            set_idx = set(pred_idx[u])

            Recall += len(set_truth & set_idx) / (len(set_truth) + 0.0)
            recalls.append(len(set_truth & set_idx) / (len(set_truth) + 0.0))

            #calculate ndcg
            topk = min(k, len(set_truth))
            idcg_k = sum([1.0/math.log(ii+2, 2) for ii in range(topk)])
            dcg_k = sum([int(pred_idx[u][j] in set_truth) / math.log(j+2, 2) for j in range(k)])
            NDCG += dcg_k / idcg_k
            ndcgs.append(dcg_k / idcg_k)
        
        Recalls[k], NDCGs[k] = Recall / len(test), NDCG / len(test)

    if args.inference_only:
        return (NDCGs, Recalls, len(test)), computing_time

    return NDCGs, Recalls, len(test)
