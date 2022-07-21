import argparse
import json
import os
from datetime import datetime, timedelta
import pathlib

import numpy as np
import mlflow
import torch
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
#from apex.optimizers import FusedAdam
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import GradScaler
#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp
# replace amp with pytorch mixed precision

from sequence_models.convolutional import ByteNetLM
#from model import ByteNetLM
#from sequence_models.constants import MASK
from dms.constants import PROTEIN_ALPHABET, PAD
from dms.utils import Tokenizer
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
#from torch.utils.data import SubsetRandomSampler
from sequence_models.datasets import UniRefDataset
#from dms.data import UNIREF50
from dms.collaters import SimpleCollater, OAMaskCollater, DMsMaskCollater
#from sequence_models.collaters import LMCollater, MLMCollater
#from sequence_models.losses import MaskedCrossEntropyLoss
from losses import MaskedCrossEntropyLoss, AustinLoss
#from torch.nn import MSELoss
from sequence_models.metrics import MaskedAccuracy
from sequence_models.utils import warmup, transformer_lr

### SET RANDOM SEEDS ###
random_seed = 1
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.empty_cache() # empty caches

home = str(pathlib.Path.home())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fpath')
    parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-off', '--offset', default=0, type=int,
                        help='Number of GPU devices to skip.')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--task', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--aml', action='store_true')  # Set true to do multi-node training on amlk8s
    parser.add_argument('-sd', '--state_dict', default=None)
    parser.add_argument('--zero_mask', action='store_true') # Set to true to use a masking scheme
    #parser.add_argument('--decay', action='store_true')
    parser.add_argument('--final_norm', action='store_true')
    parser.add_argument('--mini_run', action='store_true') # Set to True if running on subset of data
    parser.add_argument('--mask', type=str, default='autoreg')  # Set to True if running on subset of data
    parser.add_argument('--warmup', action='store_true')  # Set to True if running on subset of data


    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    if args.aml:
        pass
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8889'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    _ = torch.manual_seed(0)
    if args.aml:
        args.nr = int(os.environ['OMPI_COMM_WORLD_RANK'])
        print(args.nr)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)
    torch.cuda.set_device(gpu + args.offset)
    device = torch.device('cuda:' + str(gpu + args.offset))
    with open(args.config_fpath, 'r') as f:
        config = json.load(f)
    n_tokens = len(PROTEIN_ALPHABET)
    d_embed = config['d_embed']
    d_model = config['d_model']
    n_layers = config['n_layers']
    if 'slim' in config:
        slim = config['slim']
    else:
        slim = True
    if 'activation' in config:
        activation = config['activation']
    else:
        activation = 'relu'
    kernel_size = config['kernel_size']
    r = config['r']
    bucket_size = config['bucket_size']
    max_tokens = config['max_tokens']
    max_batch_size = config['max_batch_size']
    epochs = config['epochs']
    lr = config['lr']
    opt_level = config['opt_level']
    warmup_steps = config['warmup_steps']
    if 'rank' in config:
        weight_rank = config['rank']
    else:
        weight_rank = None
    if args.task is not None:
        config['task'] = args.task
    if args.dataset is not None:
        config['dataset'] = args.dataset
    try:
        data_dir = os.getenv('PT_DATA_DIR') + '/'
        ptjob = True
    except:
        data_dir = home + '/Desktop/DMs/data/'
        ptjob = False
    data_dir += config['dataset'] + '/'
    if args.mini_run:
        mini_size = 1000 # For troubleshooting
    # ----------------------------------------------------------
    ### DEFINE COLLATOR ###
    # ----------------------------------------------------------
    if args.mask == 'autoreg':
        simple_collater = SimpleCollater()
        collater = OAMaskCollater(simple_collater, inputs_padded=False)
    elif args.mask == 'blosum':
        simple_collater = SimpleCollater()
        collater = DMsMaskCollater(simple_collater, tokenizer=Tokenizer(), inputs_padded=False, masking_scheme="BLOSUM",
                                num_timesteps=500)
    elif args.mask == 'random':
        print('not implemented yet')
    else:
        print("Using autoreg masking scheme")
        simple_collater = SimpleCollater()
        collater = OAMaskCollater(simple_collater, inputs_padded=False)
    causal = False
    # ----------------------------------------------------------
    ### DATALOADER ###
    # ----------------------------------------------------------
    metadata = np.load(data_dir + 'lengths_and_offsets.npz')
    ds_train = UniRefDataset(data_dir, 'train', structure=False)

    train_idx = ds_train.indices
    if args.mini_run:
        len_train = np.sort(np.random.choice(metadata['ells'][train_idx], size=mini_size))[::-1]
    else:
        len_train = metadata['ells'][train_idx]
    train_sortish_sampler = SortishSampler(len_train, bucket_size, num_replicas=args.world_size, rank=rank)
    train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, len_train)

    dl_train = DataLoader(dataset=ds_train,
                          batch_sampler=train_sampler,
                          num_workers=16,
                          collate_fn=collater)

    if rank == 0:
        ds_valid = UniRefDataset(data_dir, 'valid', structure=False)
        valid_idx = ds_valid.indices
        if args.mini_run:
            len_valid = np.sort(np.random.choice(metadata['ells'][valid_idx], size=mini_size))[::-1]
        else:
            len_valid = metadata['ells'][valid_idx]
        valid_sortish_sampler = SortishSampler(len_valid, 1000, num_replicas=1, rank=0)
        valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens // 2, max_batch_size, len_valid)
        dl_valid = DataLoader(dataset=ds_valid,
                              batch_sampler=valid_sampler,
                              num_workers=8,
                              collate_fn=collater)

    # ----------------------------------------------------------
    # Initiate model
    # ----------------------------------------------------------
    if args.zero_mask:
        padding_idx = Tokenizer().tokenize(PAD)[0] #PROTEIN_ALPHABET.index(PAD)
    else:
        #padding_idx = None
        padding_idx = Tokenizer().tokenize(PAD)[0] #PROTEIN_ALPHABET.index(PAD)
    print('Using {} as padding index'.format(padding_idx))
    model = ByteNetLM(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                      causal=causal, padding_idx=padding_idx, rank=weight_rank, dropout=args.dropout,
                      tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    outputs = os.listdir(args.out_fpath)
    if len(outputs) > 0:
       last_epoch = 0
       for output in outputs:
           if 'checkpoint' in output:
               epoch = int(output.split('checkpoint')[-1][:-4])
               if epoch > last_epoch:
                   args.state_dict = args.out_fpath + output
                   last_epoch = epoch
    if args.state_dict is not None:
       print('Loading weightsweights from ' + args.state_dict + '...')
       sd = torch.load(args.state_dict, map_location=torch.device('cpu'))
       msd = sd['model_state_dict']
       #print([len(k.split('module.')) for k,v in msd.items()])
       msd = {k.split('module.')[0]: v for k,v in msd.items()}
       model.load_state_dict(msd)
       optimizer.load_state_dict(sd['optimizer_state_dict'])
       initial_epoch = sd['epoch'] + 1
       total_steps = sd['step']
       total_tokens = sd['tokens']
    else:
       initial_epoch = 0
       total_steps = 0
       total_tokens = 0
    model = model.to(device)
    scaler = GradScaler()
    # ----------------------------------------------------------
    # Loss Function
    # ----------------------------------------------------------
    if args.warmup:
        scheduler = LambdaLR(optimizer, warmup(warmup_steps))
    if args.mask == 'autoreg':
        loss_func = MaskedCrossEntropyLoss(reweight=True) # FOR ODARDMS
    elif args.mask == 'blosum':
        # loss_func = MaskedCrossEntropyLoss(reweight=False)
        loss_func = AustinLoss() # for markov
    elif args.mask == 'random':
        print('not working yet')
    accu_func = MaskedAccuracy()
    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------
    def epoch(model, train, current_step=0, current_tokens=0):
        start_time = datetime.now()
        if train:
            model = model.train()
            loader = dl_train
            t = 'Training:'
        else:
            model = model.eval()
            loader = dl_valid
            t = 'Validating:'
        losses = []
        accus = []
        ns = []
        chunk_time = datetime.now()
        n_seen = 0
        tokens_trained = current_tokens
        if train:
            n_total = len(len_train)/args.gpus #len(ds_train)
        else:
            n_total = len(len_valid) #len(ds_valid)
        for i, batch in enumerate(loader):
            print("Batch", i)
            print("rank", rank)
            print("Tokens", tokens_trained)
            # restarting from a checkpoint
            if train and i == 1 and e == initial_epoch and args.state_dict is not None:
                print("Restarting from checkpoint")
                optimizer.load_state_dict(sd['optimizer_state_dict'])
                scheduler.load_state_dict(sd['scheduler_state_dict'])
            new_loss, new_accu, new_n, new_seqs, new_processed = step(model, batch, train)
            losses.append(new_loss.item())
            accus.append(new_accu.item())
            ns.append(new_n.item())
            n_seen += new_seqs.item()
            total_n = sum(ns)
            rloss = sum(losses) / total_n
            raccu = sum(accus) / total_n
            if train:
                nsteps = current_step + i + 1
                tokens_trained += new_processed.item()
            else:
                nsteps = i
            if rank == 0:
                if ptjob:
                    end = '\n'
                    start = ''
                else:
                    start = ''
                    end = '\n'
                print(start + '%s Epoch %d of %d Step %d ntokens %d Example %d of %d loss = %.4f accu = %.4f'
                      % (t, e + 1, epochs, nsteps, tokens_trained, n_seen, n_total, rloss, raccu),
                      end=end)
            if train:
                losses = losses[-999:]
                accus = accus[-999:]
                ns = ns[-999:]
                if n_seen == n_total:
                    if rank == 0:
                        with open(args.out_fpath + 'train-metrics.csv', 'a') as f:
                            f.write(
                                ','.join([str(rloss), str(raccu), str(int(current_tokens)), str(current_step), str(e)]))
                            f.write('\n')
                        print('Training complete in ' + str(datetime.now() - chunk_time))
                        with torch.no_grad():
                            if rank == 0:
                                ckpt_fpath = args.out_fpath + 'checkpoint%d.tar' % nsteps
                                torch.save({
                                    'step': nsteps,
                                    'tokens': tokens_trained,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'epoch': e
                                }, ckpt_fpath)
                                _ = epoch(model, False, current_step=nsteps, current_tokens=tokens_trained)
                        chunk_time = datetime.now()
        if not train:
            if n_seen == n_total:
                if rank == 0:
                    with open(args.out_fpath + 'valid-metrics.csv', 'a') as f:
                        f.write(','.join([str(rloss), str(raccu), str(int(current_tokens)), str(current_step), str(e)]))
                        f.write('\n')

                print('Validation complete in ' + str(datetime.now() - start_time))
        elif rank == 0:
            print('Epoch complete in ' + str(datetime.now() - start_time))
        return i, tokens_trained

    def step(model, batch, train):
        if args.mask == 'blosum':
            src, timestep, tgt, mask,q_x = batch
            q_x = q_x.to(device)
            #print(q_x.shape)
        else:
            src, timestep, tgt, mask = batch
        print("Batchsize", len(timestep))
        src = src.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)
        input_mask = (src != padding_idx).float()
        n_tokens = mask.sum()
        n_processed = input_mask.sum()

        if train:
            optimizer.zero_grad() # reset gradients of model parameters
            with torch.cuda.amp.autocast():
                outputs = model(src, input_mask=input_mask.unsqueeze(-1))
                #print(outputs.shape)
                if args.mask == 'blosum':
                    loss = loss_func(q_x, outputs, tgt, mask, timestep) * n_tokens
                else:
                    loss = loss_func(outputs, tgt, mask, timestep) * n_tokens
                accu = accu_func(outputs, tgt, mask) * n_tokens

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_scheduler = (scale > scaler.get_scale())
            if not skip_scheduler:
                scheduler.step()

        else:
            outputs = model(src, input_mask=input_mask.unsqueeze(-1))
            #print(outputs[0])
            if args.mask == 'blosum':
                loss = loss_func(q_x, outputs, tgt, mask, timestep) * n_tokens
            else:
                loss = loss_func(outputs, tgt, mask, timestep) * n_tokens
            accu = accu_func(outputs, tgt, mask) * n_tokens

        n_seqs = torch.tensor(len(src), device=device)
        return loss, accu, n_tokens, n_seqs, n_processed

    if rank == 0:
        if not ptjob:
            mlflow.set_experiment(config['experiment'])
            mlflow.log_params(config)
    n_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print('%d model parameters' %n_parameters)
        print('%d training sequences' %len(len_train))
        print('%d validation sequences' %len(len_valid))
    for e in range(initial_epoch, epochs):
        print("epoch ", e)
        s, t = epoch(model, True, current_step=total_steps, current_tokens=total_tokens)
        total_steps += s
        total_tokens += t
        print(total_steps, total_tokens)

if __name__ == '__main__':
    main()
