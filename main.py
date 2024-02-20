import os
import time
import torch
import argparse
import sys
torch.manual_seed(123)

from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--attn_dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--model', default='RAM', type=str)
parser.add_argument('--isTrain', default=0, type=int)
parser.add_argument('--store_model', default=1, type=int)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)


args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset, args.isTrain)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    if args.model == 'MSSG':
        from models.MSSG import MSSG
        model = MSSG(usernum, itemnum, args).to(args.device)
    elif args.model == 'MSSGU':
        from models.MSSGU import MSSGU
        model = MSSGU(usernum, itemnum, args).to(args.device)

    if args.isTrain == 0:
        #the path to save models during testing
        args.saveRoot = 'models_test/'+args.model+'/'+args.dataset+'/'+args.dataset+'_'+str(args.hidden_units)+'_'+str(args.maxlen)+'_'+str(args.num_heads)+'_'+str(args.num_blocks)+'_'+str(args.batch_size)+'_'+str(args.lr)+'_'+str(args.attn_dropout_rate)
    
    model.train()
    
    epoch_start_idx = 1
    
    if args.inference_only:
        model = torch.load(args.saveRoot+'_'+str(args.num_epochs))
        model.eval()
        t_test, ct = batch_evaluate(model, dataset, args)
        print('test (NDCG@5: %.4f, HR@5: %.4f)' % (t_test[0][5], t_test[1][5]))
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0][10], t_test[1][10]))
        print('test (NDCG@20: %.4f, HR@20: %.4f)' % (t_test[0][20], t_test[1][20]))
        print('Computing time: %.3f' % ct)
        sys.exit("Done inference")
    
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    lambda1 = lambda epoch: 0.995 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(adam_optimizer, lr_lambda=lambda1)
    
    T = 0.0
    t0 = time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices] + 1e-6, pos_labels[indices])
            loss += bce_criterion(neg_logits[indices] + 1e-6, neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            loss.backward()
            adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
        scheduler.step()
    
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t3 = time.time()
            t_test = batch_evaluate(model, dataset, args)
            et = time.time() - t3
            
            print('epoch:%d, time: %f(s), evaluate time: %f(s), test (NDCG@10: %.4f, Recall@10: %.4f)'
                    % (epoch, T, et, t_test[0][10], t_test[1][10]))

            f.write(str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            if args.isTrain == 0 and args.store_model==1:
                torch.save(model, args.saveRoot+'_'+str(epoch))  
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'MSSG.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    
    f.close()
    sampler.close()
    print("Done")
