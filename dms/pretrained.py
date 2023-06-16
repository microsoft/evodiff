import torch
import json
from dms.model import ByteNetLMTime
from sequence_models.constants import MSA_ALPHABET, PROTEIN_ALPHABET, ALL_AAS, PAD
from dms.utils import Tokenizer
from dms.collaters import D3PMCollater, OAMaskCollater
from sequence_models.collaters import MLMCollater

def download_model(model_name):
    #url = f"https://.. {model_name} .. " # TODO add links when uploaded to Zenodo
    #state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
    state_dict = "zenodo/checkpoints/"+model_name+".tar"
    return state_dict

def load_d3pm_checkpoint(model_name, config_path, diffusion_timesteps, tokenizer=Tokenizer(), causal=False):
    with open(config_path, 'r') as f:
        config = json.load(f)
    n_tokens = len(MSA_ALPHABET)
    d_embed = config['d_embed']
    d_model = config['d_model']
    n_layers = config['n_layers']
    kernel_size = config['kernel_size']
    r = config['r']
    lr = config['lr']
    masking_idx = tokenizer.mask_id
    if 'rank' in config:
        weight_rank = config['rank']
    else:
        weight_rank = None
    if 'slim' in config:
        slim = config['slim']
    else:
        slim = True
    if 'activation' in config:
        activation = config['activation']
    else:
        activation = 'relu'
    dropout=0.0
    tie_weights=False
    final_norm=True
    weight_decay=0.0
    model = ByteNetLMTime(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                          causal=causal, padding_idx=masking_idx, rank=weight_rank, dropout=dropout,
                          tie_weights=tie_weights, final_ln=final_norm, slim=slim, activation=activation,
                          timesteps=diffusion_timesteps)
    #optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    state_dict = download_model(model_name)
    sd = torch.load(state_dict, map_location=torch.device('cpu'))
    msd = sd['model_state_dict']
    msd = {k.split('module.')[1]: v for k, v in msd.items()}
    model.load_state_dict(msd)
    #optimizer.load_state_dict(sd['optimizer_state_dict'])

    return model, tokenizer

def D3PM_BLOSUM_640M():
    dt=500
    tokenizer = Tokenizer(path_to_blosum="data/blosum62-special-MSA.mat", sequences=True)
    Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=dt)
    collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=dt, Q=Q_t, Q_bar=Q_prod)
    model, tokenizer = load_d3pm_checkpoint("d3pm-blosum-640M", "config/config640M.json",
                                                      diffusion_timesteps=dt,
                                                      tokenizer=tokenizer)
    scheme = 'd3pm'
    return model, collater, tokenizer, scheme

def D3PM_UNIFORM_640M():
    dt = 500
    tokenizer = Tokenizer(path_to_blosum="data/blosum62-special-MSA.mat", sequences=True)
    Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=dt)
    collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=dt, Q=Q_t, Q_bar=Q_prod)
    model, tokenizer = load_d3pm_checkpoint("d3pm-uniform-640M", "config/config640M.json", diffusion_timesteps=dt,
                                            tokenizer=tokenizer)
    scheme = 'd3pm'
    return model, collater, tokenizer, scheme

def OA_AR_640M():
    tokenizer = Tokenizer()
    collater = OAMaskCollater(tokenizer=tokenizer)
    model, tokenizer = load_d3pm_checkpoint("oaar-640M", "config/config640M.json", diffusion_timesteps=None, \
                         tokenizer=tokenizer)
    scheme = 'mask'
    return model, collater, tokenizer, scheme

def LR_AR_640M():
    tokenizer = Tokenizer(protein_alphabet=PROTEIN_ALPHABET, all_aas=ALL_AAS, pad=PAD)
    collater = MLMCollater(tokenizer=tokenizer) # TODO???
    model, tokenizer = load_d3pm_checkpoint("lrar-640M", "config/config640M.json", diffusion_timesteps=None, \
                                tokenizer=tokenizer, \
                                causal=True)
    scheme='mask'
    return model, collater, tokenizer, scheme
