import argparse
import json
import os
from datetime import datetime, timedelta
import pathlib

import numpy as np
# import mlflow
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
# from apex.optimizers import FusedAdam
# from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
from dms.collaters import D3PMCollaterMSA
from dms.utils import Tokenizer
from dms.constants import MSA_ALPHABET_NEW, MSA_ALL_AAS
from losses import  D3PMCELossMSA,  D3PMLVBLossMSA
from sequence_models.esm import MSATransformer
from model import MSATransformerTime
from sequence_models.constants import MSA_PAD, MASK
from sequence_models.datasets import TRRMSADataset, A3MMSADataset
from sequence_models.collaters import MSAAbsorbingCollater
from sequence_models.losses import MaskedCrossEntropyLossMSA
from sequence_models.metrics import MaskedAccuracy
from torch.utils.data import Subset, SubsetRandomSampler
from sequence_models.utils import warmup, transformer_lr
from sequence_models.samplers import SortishSampler, ApproxBatchSampler

# from torch.utils.tensorboard import SummaryWriter

home = str(pathlib.Path.home())


# writer = SummaryWriter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fpath')
    parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/')
    # parser.add_argument('out_fpath', type=str, nargs='?', default=home + '/model_output/openfold_checkpoints/')
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
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--dummy', required=False)
    parser.add_argument('--mask', default='blosum')
    parser.add_argument('--reweighting_term', type=float, default=1.0)


    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    if args.aml:
        pass
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8881'
    mp.spawn(train, nprocs=args.gpus, args=(args,))  # calls train gpu number of times, passes in args


def train(gpu, args):
    _ = torch.manual_seed(0)
    np.random.seed(0)
    # if args.aml:
    #     args.nr = int(os.environ['OMPI_COMM_WORLD_RANK'])  # TODO: is this correct?
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

    selection_type = config['selection_type']
    d_embed = config['d_embed']
    d_hidden = config['d_hidden']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    bucket_size = config['bucket_size']
    max_tokens = config['max_tokens']
    max_batch_size = config['max_batch_size']
    epochs = config['epochs']
    lr = config['lr']
    warmup_steps = config['warmup']
    max_square_tokens = config['max_square_tokens']
    n_sequences = config['n_sequences']
    max_seq_len = config['max_seq_len']

    if args.dataset is not None:
        config['dataset'] = args.dataset

    try:
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/'
        data_dir = os.getenv('AMLT_DATA_DIR') + '/'
        data_dir += config['dataset'] + '/'
        ptjob = True
    except:
        data_top_dir = 'data/'
        #print(data_top_dir)
        data_dir = data_top_dir
        data_dir += config['dataset'] + '/'
        #print(data_dir)
        ptjob = False

    # build datasets, samplers, and loaders
    if args.mask == 'autoreg':
        tokenizer = Tokenizer()
        collater = MSAAbsorbingCollater(MSA_ALPHABET)
        diffusion_timesteps = None # Not input to model
    elif args.mask == 'blosum' or args.mask == 'random':
        diffusion_timesteps = config['diffusion_timesteps']
        tokenizer = Tokenizer(path_to_blosum=data_top_dir+"blosum62-special-MSA.mat", protein_alphabet=MSA_ALPHABET_NEW,
                              all_aas=MSA_ALL_AAS)
        if args.mask == 'random':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        if args.mask == 'blosum':
            Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps, max=6)
        collater = D3PMCollaterMSA(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)
        Q_prod = Q_prod.to(device)
    else:
        print("mask must be: 'autoreg', 'blosum', or 'random'")


    if config['dataset'] == 'trrosetta':
        dataset = TRRMSADataset(data_dir=data_dir, selection_type=selection_type, n_sequences=n_sequences,
                                max_seq_len=max_seq_len)
        train_size = len(dataset)
        random_ind = np.random.choice(train_size, size=int(train_size * 0.8), replace=False)
    elif config['dataset'] == 'openfold':
        dataset = A3MMSADataset(data_dir=data_dir, selection_type=selection_type, n_sequences=n_sequences,
                                max_seq_len=max_seq_len)
        train_size = len(dataset)
        print("TRAIN SIZE:", train_size, rank)
        random_ind = np.random.choice(train_size, size=(train_size - 10000), replace=False)
    else:
        print("Dataset options: trrosetta or openfold")

    ds_train = Subset(dataset, random_ind)

    #metadata = np.load(data_dir + config['dataset'] + '_lengths.npz')['ells']
    train_idx = ds_train.indices
    #len_train = metadata[train_idx]

    #len_train = np.minimum(len_train, max_seq_len)

    #train_sortish_sampler = SortishSampler(len_train, bucket_size, num_replicas=args.world_size, rank=rank)
    #train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, len_train,
    #                                   max_square_tokens=max_square_tokens, msa_depth=n_sequences)

    dl_train = DataLoader(dataset=ds_train,
                          batch_size=4,
                          #batch_sampler=train_sampler,
                          collate_fn=collater,
                          num_workers=8)

    if rank == 0:
        val_ind = np.delete(np.arange(train_size), random_ind)
        ds_valid = Subset(dataset, val_ind)
        valid_idx = ds_valid.indices
        #len_valid = metadata[valid_idx]
        #len_valid = np.minimum(len_valid, max_seq_len)

        #valid_sortish_sampler = SortishSampler(len_valid, bucket_size, num_replicas=1, rank=0)
        #valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens, max_batch_size, len_valid,
        #                                   max_square_tokens=max_square_tokens, msa_depth=n_sequences)

        dl_valid = DataLoader(dataset=ds_valid,
                              batch_size=1,
                              #batch_sampler=valid_sampler,
                              collate_fn=collater,
                              num_workers=8)

    # Initiate model
    model = MSATransformerTime(d_embed, d_hidden, n_layers, n_heads, timesteps=diffusion_timesteps, use_ckpt=True).cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, warmup(warmup_steps))
    scaler = torch.cuda.amp.GradScaler()

    outputs = os.listdir(args.out_fpath)

    if len(outputs) > 0:
        last_epoch = -1
        for output in outputs:
            if 'checkpoint' in output:
                epoch = int(output.split('checkpoint')[-1][:-4])
                if epoch > last_epoch:
                    args.state_dict = args.out_fpath + output
                    last_epoch = epoch
    if args.state_dict is not None:
        print('Loading weights from ' + args.state_dict + '...')
        sd = torch.load(args.state_dict, map_location=torch.device('cpu'))
        msd = sd['model_state_dict']
        msd = {k.split('module.')[1]: v for k, v in msd.items()}
        model.load_state_dict(msd)
        optimizer.load_state_dict(sd['optimizer_state_dict'])
        scheduler.load_state_dict(sd['scheduler_state_dict'])
        scaler.load_state_dict(sd['scaler_state_dict']),
        initial_epoch = sd['epoch'] + 1
        total_steps = sd['step']
        total_tokens = sd['tokens']
    else:
        initial_epoch = 0
        total_steps = 0
        total_tokens = 0

    # optimizer.state = {}
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = model.to(device)
    model = DDP(model, device_ids=[gpu + args.offset], output_device=args.offset)

    # if args.state_dict is not None:
    #     if 'amp_state_dict' in sd:
    #         amp.load_state_dict(sd['amp_state_dict'])
    #     else:
    #         amp.load_state_dict({'loss_scaler0': {'loss_scale': 512., 'unskipped': 0}})

    if args.mask == 'autoreg':
        loss_func = MaskedCrossEntropyLossMSA(ignore_index=MSA_ALPHABET.index(MSA_PAD))
    elif args.mask == 'blosum' or args.mask == 'random':
        # Austin = LVB + lambda * CE
        loss_func1 = D3PMLVBLossMSA(tmax=diffusion_timesteps, tokenizer=tokenizer)
        loss_func2 = D3PMCELossMSA(tokenizer=tokenizer)
        _lambda = args.reweighting_term


    accu_func = MaskedAccuracy()

    with open(args.config_fpath, 'r') as f_from, open(args.out_fpath + "config.json", "w") as f_to:
        f_to.write(f_from.read())

    def epoch(model, e, split, current_step=0, current_tokens=0):
        start_time = datetime.now()
        if split == 'train':
            loader = dl_train
            t = 'Training:'
        elif split == 'valid':
            loader = dl_valid
            t = 'Validating:'
        else:
            # loader = dl_test
            t = "Testing"
        ardm_losses = []
        nll_losses = []
        accus = []
        ns = []
        num_seqs = []
        chunk_time = datetime.now()
        n_seen = 0
        tokens_trained = current_tokens
        if split == 'train':
            n_total = len(ds_train)
        elif split == 'valid':
            n_total = len(ds_valid)
        # else:
        #     n_total = len(ds_test)
        for i, batch in enumerate(loader):
            if split == 'train' and i == 1 and e == initial_epoch and args.state_dict is not None:
                optimizer.load_state_dict(sd['optimizer_state_dict'])
                scheduler.load_state_dict(sd['scheduler_state_dict'])
                scaler.load_state_dict(sd['scaler_state_dict'])
            ardm_loss, nll_loss, new_accu, new_n, new_seqs, new_processed = step(model, batch, split)

            if split == 'train':
                dist.reduce(ardm_loss, 0, op=dist.ReduceOp.SUM)
                dist.reduce(nll_loss, 0, op=dist.ReduceOp.SUM)
                dist.reduce(new_accu, 0, op=dist.ReduceOp.SUM)
                dist.reduce(new_n, 0, op=dist.ReduceOp.SUM)
                dist.reduce(new_seqs, 0, op=dist.ReduceOp.SUM)
            ardm_losses.append(ardm_loss.item())
            nll_losses.append(nll_loss.item())
            accus.append(new_accu.item())
            ns.append(new_n.item())
            num_seqs.append(new_seqs.item())
            n_seen += new_seqs.item()
            total_n = sum(ns)
            total_s = sum(num_seqs)
            rloss_ardm = sum(ardm_losses) / len(ardm_losses)
            #r_ce_loss = sum(ce_losses) / len(ce_losses)
            rloss_nll = sum(nll_losses) / len(nll_losses)
            #rloss_ardm = sum(ardm_losses) / total_s  # or len(ce_losses)
            #print(rloss_ardm)
            #rloss_nll = sum(nll_losses) / total_n
            raccu = sum(accus) / total_n

            if split == 'train':
                # writer.add_scalar("Loss/train", rloss, e)
                # writer.add_scalar("Acc/train", raccu, e)
                nsteps = current_step + i + 1
                tokens_trained += new_processed.item()
            else:
                # writer.add_scalar("Loss/valid", rloss, e)
                # writer.add_scalar("Acc/valid", raccu, e)
                nsteps = i
            if rank == 0:
                if ptjob:
                    end = '\n'
                    start = ''
                else:
                    start = '\r'
                    end = ''
                print(
                    start + '%s Epoch %d of %d Step %d Example %d of %d ardm_loss = %.4f nll_loss = %.4f accu = %.4f'
                    % (t, e + 1, epochs, nsteps, n_seen, n_total, rloss_ardm, rloss_nll, raccu),
                    end=end)
                print('\n')

            if split == 'train':
                ardm_losses = ardm_losses[-999:]
                nll_losses = nll_losses[-999:]
                accus = accus[-999:]
                ns = ns[-999:]
                num_seqs = num_seqs[-999:]
                if datetime.now() - chunk_time > timedelta(hours=6):
                    # if rank == 0:
                    #     if not ptjob:
                    #         mlflow.log_metrics({'train_loss': rloss,
                    #                             'train_accu': raccu,
                    #                             'n_tokens': total_n},
                    #                            step=nsteps)
                    # if not ptjob:
                    #     print()

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
                                'scaler_state_dict': scaler.state_dict(),
                                'epoch': e,
                                # 'amp_state_dict': amp.state_dict()
                            }, ckpt_fpath)
                            _ = epoch(model, e, split='valid', current_step=nsteps, current_tokens=tokens_trained)
                    chunk_time = datetime.now()
        if split == 'valid':
            with open(args.out_fpath + 'metrics.csv', 'a') as f:
                f.write(','.join(
                    [str(rloss_ardm), str(rloss_nll), str(raccu), str(int(current_tokens)), str(current_step)]))
                f.write('\n')  # Can add for train too

            print('Validation complete in ' + str(datetime.now() - start_time))

        if split == 'test':
            with open(args.out_fpath + 'metrics_test.csv', 'a') as f:
                f.write(','.join(
                    [str(rloss_ardm), str(rloss_nll), str(raccu), str(int(current_tokens)),
                     str(current_step)]))  # TODO: is this correct?
                f.write('\n')

            print('Testing complete in ' + str(datetime.now() - start_time))

        # elif rank == 0:
        #     if not ptjob:
        #         print()
        print('Epoch complete in ' + str(datetime.now() - start_time))
        return i, tokens_trained

    def step(model, batch, split):
        if args.mask == 'blosum' or args.mask == 'random':
            src, src_one_hot, timestep, tgt, Q, q, q_minus1 = batch
            src_one_hot = src_one_hot.to(device)
            q = q.to(device)
            q_minus1 = q_minus1.to(device)
            Q = Q.to(device)
            timestep = timestep.to(device)
        else:
            src, tgt, mask = batch
            mask = mask.to(device)
        # print('z', rank, device)
        src = src.to(device)
        # print('y', rank)
        tgt = tgt.to(device)
        input_mask = (src != MSA_ALPHABET_NEW.index(MASK)).float()
        nonpad_mask = (src != MSA_ALPHABET_NEW.index(MSA_PAD)).float()
        if args.mask == 'blosum' or args.mask == 'random':
            n_tokens = nonpad_mask.sum()
        else:
            n_tokens = mask.sum()
        if n_tokens == 0:
            raise ValueError("N TOKENS IN STEP IS 0!!")
        n_processed = input_mask.sum()

        if split == 'train':
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(src, timestep)
                if args.mask == 'blosum' or args.mask == 'random':
                    lvb_loss = loss_func1(src, src_one_hot, q, q_minus1, outputs, tgt, nonpad_mask, timestep, Q, Q_prod) #* n_tokens
                    ce_loss = loss_func2(outputs, tgt, nonpad_mask)
                    nll_loss = ce_loss
                    accu = accu_func(outputs, tgt, nonpad_mask) * n_tokens
                    loss = lvb_loss + _lambda * ce_loss
                elif args.mask == 'autoreg':
                    ce_loss, nll_loss = loss_func(outputs, tgt, mask, nonpad_mask)
                    loss = ce_loss
                    accu = accu_func(outputs, tgt, mask) * n_tokens
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_scheduler = (scale > scaler.get_scale())
            if not skip_scheduler:
                scheduler.step()
        elif split == 'valid' or split == 'test':

            with torch.cuda.amp.autocast():
                outputs = model(src)
                if args.mask == 'blosum' or args.mask == 'random':
                    lvb_loss = loss_func1(src, q, q_minus1, outputs, tgt, nonpad_mask, timestep, Q, Q_prod)  # * n_tokens
                    ce_loss = loss_func2(outputs, tgt, nonpad_mask)
                    nll_loss = ce_loss
                    loss = lvb_loss + _lambda * ce_loss
                    accu = accu_func(outputs, tgt, nonpad_mask) * n_tokens
                elif args.mask == 'autoreg':
                    ce_loss, nll_loss = loss_func(outputs, tgt, mask, nonpad_mask)
                    loss = ce_loss
                    accu = accu_func(outputs, tgt, mask) * n_tokens
        n_seqs = torch.tensor(len(src), device=device)
        return loss, nll_loss, accu, n_tokens, n_seqs, n_processed

    n_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print('%d model parameters' % n_parameters)
        #print('%d training sequences' % len(len_train))
        #print('%d validation sequences' % len(len_valid))
    for e in range(initial_epoch, epochs):
        print("epoch: ", e + 1, rank)
        #train_sortish_sampler.set_epoch(e + 1)
        s, t = epoch(model, e, split='train', current_step=total_steps, current_tokens=total_tokens)
        total_steps += s
        total_tokens += t

        # writer.flush()
        # writer.close()

    # _, _ = epoch(model, e, split='test', current_step=total_steps, current_tokens=total_tokens)


if __name__ == '__main__':
    main()