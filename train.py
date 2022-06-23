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
#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp
# replace amp with pytorch mixed precision

#from sequence_models.convolutional import ByteNetLM # TODO: FIGURE OUT BYTENET
from model import ByteNetLM
#from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK # to do update
from dms.constants import PROTEIN_ALPHABET, PAD, MASK
from sequence_models.samplers import SortishSampler, ApproxBatchSampler # TODO reimplement kevins sampler
from torch.utils.data import SubsetRandomSampler
#from sequence_models.datasets import UniRefDataset # TODO reimplement kevins dataset - faster
from dms.data import UNIREF50
from dms.collaters import SimpleCollater, OAMaskCollater
#from sequence_models.collaters import LMCollater, MLMCollater
#from sequence_models.losses import MaskedCrossEntropyLoss
from losses import MaskedCrossEntropyLoss
from sequence_models.metrics import MaskedAccuracy
from sequence_models.utils import warmup, transformer_lr


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
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--final_norm', action='store_true')


    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    if args.aml:
        pass
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    _ = torch.manual_seed(0)
    if args.aml:
        args.nr = int(os.environ['OMPI_COMM_WORLD_RANK'])
    rank = args.nr * args.gpus + gpu
    #print(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)
    #print("Here")
    torch.cuda.set_device(gpu + args.offset)
    device = torch.device('cuda:' + str(gpu + args.offset))
    #print(device)
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
        data_dir = home + '/scratch/github/DMs/data/'
        ptjob = False
    data_dir += config['dataset'] + '/'
    #print(data_dir)
    # ----------------------------------------------------------
    # Build datasets, samplers, and loaders
    # ----------------------------------------------------------
    #
    # ----------------------------------------------------------
    ### DEFINE COLLATOR ###
    # ----------------------------------------------------------
    #if config['task'] == 'lm':
    #    collater = LMCollater(PROTEIN_ALPHABET)
    #    causal = True
    #elif config['task'] == 'rlm':
    #    collater = LMCollater(PROTEIN_ALPHABET, backwards=True)
    #    causal = True
    #else:
    #    collater = MLMCollater(PROTEIN_ALPHABET)
    #    causal = False
    print("Only using sarah collaters")
    simple_collater = SimpleCollater(norm=True)
    collater = OAMaskCollater(simple_collater, inputs_padded=True)
    causal = False
    #metadata = np.load(data_dir + 'lengths_and_offsets.npz')
    #ds_train = UniRefDataset(data_dir, 'train', structure=False)
    # ----------------------------------------------------------
    ### DATALOADER ###
    # ----------------------------------------------------------
    #print(data_dir)
    index_file = data_dir + 'INDEX.txt'
    seq_file = data_dir + 'SEQ.txt'
    data = UNIREF50(index_file, seq_file)

    # TODO: make test/train split more efficient
    #train_size = int(0.8 * len(data))
    #val_size = len(data) - train_size
    #ds_train, ds_valid = torch.utils.data.random_split(data, [train_size, val_size])

    #train_idx = ds_train.indices
    #len_train = metadata['ells'][train_idx]
    # TODO: implement samplers for more efficient iterating
    #train_sortish_sampler = SortishSampler(len_train, bucket_size, num_replicas=args.world_size, rank=rank)
    #train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, len_train)
    #dl_train = DataLoader(dataset=ds_train,
    #                      batch_sampler=train_sampler,
    #                      num_workers=16,
    #                      collate_fn=collater)
    # no of batches = num_samples/batch_size when using a subset sampler
    #sample_idx = np.random.randint(0, train_size, 1000)

    ## TODO: implement samplers
    # USING 100 batches = 1000 samples/100 batch_size for testing
    train_size = 10000
    val_size = 10000
    train_samples = 1
    sample_idx = np.random.randint(0,train_size,train_samples)
    train_sampler = SubsetRandomSampler(sample_idx)

    print("Using simple sampler")
    dl_train = DataLoader(data,
                          sampler=train_sampler,
                          batch_size=max_batch_size, # batches = samples/batch_size
                          num_workers=16, # CPU
                          collate_fn=collater)

    # TODO: what to use for validation? using random split for now
    if rank == 0:
        #ds_valid = UNIREF50(index_file, seq_file)
        #valid_idx = ds_valid.indices
        #len_valid = metadata['ells'][valid_idx]
        #valid_sortish_sampler = SortishSampler(len_valid, 1000, num_replicas=1, rank=0)
        #valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens // 2, max_batch_size, len_valid)
        #val_idx = np.random.randint(0, test_size, 2000)

        ## TODO: implement samplers
        # USING 2 batches = 200 samples/100 batch_size for testing
        val_samples = 1000
        val_idx = np.random.randint(0,val_size,val_samples)
        valid_sampler = SubsetRandomSampler(val_idx)
        dl_valid = DataLoader(dataset=data,
                              sampler=valid_sampler,
                              batch_size = max_batch_size,
                              num_workers=8,
                              collate_fn=collater)
    else:
        print("this commented right now")
    #     valid_sampler = DistributedSampler(ds_valid, num_replicas=args.world_size, rank=rank, shuffle=False)
    #     dl_valid = DataLoader(dataset=ds_valid, batch_size=64, num_workers=8, collate_fn=collater,
    #                           drop_last=True, sampler=valid_sampler)

    # ----------------------------------------------------------
    # Initiate model
    # ----------------------------------------------------------
    if args.zero_mask:
        padding_idx = PROTEIN_ALPHABET.index(PAD)
    else:
        #padding_idx = None
        padding_idx = PROTEIN_ALPHABET.index(PAD)
    print('Using {} as padding index'.format(padding_idx))
    model = ByteNetLM(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                      causal=causal, padding_idx=padding_idx, rank=weight_rank, dropout=args.dropout,
                      tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation)
    #optimizer = FusedAdam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    outputs = os.listdir(args.out_fpath)
    #if len(outputs) > 0:
    #    last_epoch = 0
    #    for output in outputs:
    #        if 'checkpoint' in output:
    #            epoch = int(output.split('checkpoint')[-1][:-4])
    #            if epoch > last_epoch:
    #                args.state_dict = args.out_fpath + output
    #                last_epoch = epoch
    #if args.state_dict is not None:
    #    print('Loading weights from ' + args.state_dict + '...')
    #    sd = torch.load(args.state_dict, map_location=torch.device('cpu'))
    #    msd = sd['model_state_dict']
    #    msd = {k.split('module.')[1]: v for k, v in msd.items()}
    #    model.load_state_dict(msd)
    #    optimizer.load_state_dict(sd['optimizer_state_dict'])
    #    initial_epoch = sd['epoch'] + 1
    #    total_steps = sd['step']
    #    total_tokens = sd['tokens']
    #else:
    #    initial_epoch = 0
    #    total_steps = 0
    #    total_tokens = 0
    initial_epoch=0
    total_steps=0
    total_tokens=0
    model = model.to(device)
    #optimizer.state = {}
    #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    #model = DDP(model)
    if args.decay:
        scheduler = LambdaLR(optimizer, transformer_lr(warmup_steps))
    else:
        scheduler = LambdaLR(optimizer, warmup(0.1))
    #if args.state_dict is not None:
    #    if 'amp_state_dict' in sd:
    #        amp.load_state_dict(sd['amp_state_dict'])
    #    else:
    #        amp.load_state_dict({'loss_scaler0': {'loss_scale': 512., 'unskipped': 0}})
    loss_func = MaskedCrossEntropyLoss()
    accu_func = MaskedAccuracy()

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
            n_total = train_samples
        else:
            n_total = val_samples
        for i, batch in enumerate(loader):
            print("Batch", i)
            # This is for restarting from a checkpoint
            #if train and i == 1 and e == initial_epoch and args.state_dict is not None:
            #    optimizer.load_state_dict(sd['optimizer_state_dict'])
            #    scheduler.load_state_dict(sd['scheduler_state_dict'])
            new_loss, new_accu, new_n, new_seqs, new_processed = step(model, batch, train)
            # Don't need reduce
            #if train:
            #    dist.reduce(new_loss, 0, op=dist.ReduceOp.SUM)
            #    dist.reduce(new_accu, 0, op=dist.ReduceOp.SUM)
            #    dist.reduce(new_n, 0, op=dist.ReduceOp.SUM)
            #    dist.reduce(new_seqs, 0, op=dist.ReduceOp.SUM)
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
                if datetime.now() - chunk_time > timedelta(hours=4):
                    if rank == 0:
                        if not ptjob:
                            mlflow.log_metrics({'train_loss': rloss,
                                                'train_accu': raccu,
                                                'n_tokens': total_n},
                                               step=nsteps)
                        if not ptjob:
                            print()
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
                                    'epoch': e,
                                    'amp_state_dict': amp.state_dict()
                                }, ckpt_fpath)
                                _ = epoch(model, False, current_step=nsteps, current_tokens=tokens_trained)
                        chunk_time = datetime.now()
        if not train:
            if rank == 0:
                if not ptjob:
                    print()
                    mlflow.log_metrics({'valid_loss': rloss,
                                        'valid_accu': raccu,
                                        'n_tokens': current_tokens},
                                       step=current_step)
                with open(args.out_fpath + 'metrics.csv', 'a') as f:
                    f.write(','.join([str(rloss), str(raccu), str(int(current_tokens)), str(current_step)]))
                    f.write('\n')

                print('Validation complete in ' + str(datetime.now() - start_time))
        elif rank == 0:
            if not ptjob:
                print()
            print('Epoch complete in ' + str(datetime.now() - start_time))
        return i, tokens_trained

    def step(model, batch, train):
        src, timestep, tgt, mask = batch
        src = src.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)
        #timestep = timestep.to(device)
        #input_mask = (src != PROTEIN_ALPHABET.index(PAD)).float()
        input_mask = (src != PROTEIN_ALPHABET.index(MASK)).float()
        outputs = model(src, input_mask=input_mask.unsqueeze(-1))
        n_tokens = mask.sum()
        print("n_tokens", n_tokens)
        n_processed = input_mask.sum()
        loss = loss_func(outputs, tgt, mask, timestep) * n_tokens
        accu = accu_func(outputs, tgt, mask) * n_tokens # TODO: check that this works for your problem
        if train:
            optimizer.zero_grad() # reset gradients of model parameters
            loss.backward() # backprop prediction loss
            #with amp.scale_loss(loss / max_tokens / 0.15, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            print("Clip grad norm", clip_grad_norm_(model.parameters(optimizer), max_norm=np.inf)) # clip gradients (norm computed over all gradients)
            #troubleshoot
            #for name, param in model.named_parameters():
            #    if param.requires_grad:
            #        print(name, param.data)
            # issue embedder.embedder.weight tensor
            #
            optimizer.step() # adjust parameters by the gradients collected in backward pass
            scheduler.step()

        n_seqs = torch.tensor(len(src), device=device)
        return loss, accu, n_tokens, n_seqs, n_processed

    if rank == 0:
        if not ptjob:
            mlflow.set_experiment(config['experiment'])
            mlflow.log_params(config)
    n_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print('%d model parameters' %n_parameters)
        print('%d training sequences' %train_samples)
        print('%d validation sequences' %val_samples)
    for e in range(initial_epoch, epochs):
        print("epoch ", e)
        #train_sortish_sampler.set_epoch(e + 1) # NEEDED for DDP - ignoring for now?
        s, t = epoch(model, True, current_step=total_steps, current_tokens=total_tokens)
        total_steps += s
        total_tokens += t
        print(total_steps, total_tokens)


if __name__ == '__main__':
    main()
