import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import argparse
import math
import timeit
import os
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
import torch.distributed as dist
import torch.utils.data.distributed
from utils import args2json, save_model, save_history, show_progress

def toseq(tensor):
    return ''.join(map(str, tensor.tolist()))

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        # print(gold)
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_dataset = TranslationDataset(
        src_word2idx=data['dict']['src'],
        tgt_word2idx=data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'])

    if opt.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=(train_sampler is None),
        sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, train_sampler, valid_loader

def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    epoch_start = timeit.default_timer()

    for batch in training_data:
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        # print('-----------')
        # print(f'src_seq : {src_seq.shape}')
        # print(f'src_pos : {src_pos.shape}')
        # print(f'tgt_seq : {tgt_seq.shape}')
        # print(f'tgt_pos : {tgt_pos.shape}')
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    print(' time %4.1f sec' % (timeit.default_timer() - epoch_start))

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in validation_data:
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, train_sampler, validation_data, optimizer, device, opt):
    ''' Start training '''
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    history = []
    valid_accus = []
    for e in range(opt.epoch):
        if opt.world_size > 1:
            train_sampler.set_epoch(e)

        with torch.autograd.profiler.profile(enabled=opt.profile) as prof:
            train_loss, train_accu = train_epoch(model, training_data, optimizer, device, smoothing=opt.label_smoothing)

        if opt.profile:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            exit()

        valid_loss, valid_accu = eval_epoch(model, validation_data, device)

        history.append([train_loss, train_accu, valid_loss, valid_accu])
        valid_accus += [valid_accu]

        if valid_accu >= max(valid_accus) and opt.rank == 0:
            save_model(model, opt.result_dir)
            print('[Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file and opt.rank == 0:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=e, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=e, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

        show_progress(e+1, opt.epoch, train_loss, valid_loss, train_accu, valid_accu)

    if opt.rank == 0:
        save_history(history, opt.result_dir)

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='../pssp-data/dataset.pt')

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=20)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    # distributed training
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training.')
    parser.add_argument('-dist_url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-dist_backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('-world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('-rank', default=0, type=int,
                        help='node rank for distributed training')

    parser.add_argument('-log', default=None)
    parser.add_argument('-result_dir', type=str, default='./result')
    # parser.add_argument('-save_model', type=str, default='model')
    # parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-profile', action='store_true', default=False)
    parser.add_argument('-num_workers', type=int, default=0)

    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model

    os.makedirs(opt.result_dir, exist_ok=True)
    args2json(opt, opt.result_dir)

    #========= Init Distributed ========#
    torch.manual_seed(opt.seed)
    opt.distributed = opt.world_size > 1

    if opt.distributed:
        print("Init distributed backend {}: world size {}; rank {}"
              .format(opt.dist_backend, opt.world_size, opt.rank))
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method=opt.dist_url,
            world_size=opt.world_size,
            rank=opt.rank)

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, train_sampler, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    if opt.distributed:
        transformer = torch.nn.parallel.DistributedDataParallel(transformer, find_unused_parameters=True)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, train_sampler, validation_data, optimizer, device, opt)

if __name__ == '__main__':
    main()
