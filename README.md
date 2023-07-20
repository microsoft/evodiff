# EvoDiff: Generation of protein sequences and evolutionary alignments via discrete diffusion models

In this work, we train and evaluate a series of discrete diffusion models for both unconditional and conditional generation of single protein sequences as well as multiple sequence alignments (MSAs). We test both order-agnostic autoregressive diffusion and discrete denoising diffusion probabilistic models for protein sequence generation; formulate unique, bio-inspired corruption schemes for both classes of models; and evaluate the quality of generated samples for fidelity, diversity, and structural plausibility.

### Installation
```
cd evodiff
conda env create -f environment.yml
conda activate evodiff
pip install -e .
```
We obtain sequences from the [Uniref50 dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4375400/), which contains approximately 45 million protein sequences. The Multiple Sequence Alignments (MSAs) are from the [OpenFold dataset](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v2), containing MSAs for 132,000 unique Protein Data Bank (PDB) chains.

### Loading pretrained models
To load a model:
```
from evodiff.pretrained import OA_AR_640M

model, collater, tokenizer, scheme = OA_AR_640M()
```
Available models are:
* ``` D3PM_BLOSUM_640M() ```
* ``` D3PM_BLOSUM_38M() ```
* ``` D3PM_UNIFORM_640M() ```
* ``` D3PM_UNIFORM_38M() ```
* ``` OA_AR_640M() ```
* ``` OA_AR_38M() ```
* ``` LR_AR_640M() ```
* ``` LR_AR_38M() ```
* ``` MSA_D3PM_BLOSUM() ```
* ``` MSA_D3PM_UNIFORM() ```
* ``` MSA_D3PM_OA_AR_RANDSUB() ```
* ``` MSA_D3PM_OA_AR_MAXSUB() ```

### Unconditional sequence generation
For sequence generation run:
``` python generate.py --model-type oa_ar_640m --final_norm --num-seqs 250 ```

For MSA generation run:
``` python generate-msa.py TODO: ADD MODEL TYPE --subsampling random --batch-size 1 ```

### Conditional sequence generation from MSA
There are two ways to conditionally generate an MSA. 

The first is to generate the alignment from the query. To do so run:

``` python generate-msa.py TODO: ADD MODEL TYPE --subsampling random --batch-size 1 --start-query ```

The second is to generate the query from the alignment. To do so run:

``` python generate-msa.py TODO: ADD MODEL TYPE --subsampling random --batch-size 1 --start-msa ```

Note that you can only start-query or start-msa, not both. To generate unconditionally, omit the flags (see the example in the above section).

To create the Potts model, which serves as a baseline, we use [CCMpredPy and CCMgen](https://github.com/soedinglab/CCMgen/wiki/Getting-Started-with-CCMgen-and-CCMpredPy).

### Analysis of generations
To access the test sequences:
```
test_data = UniRefDataset('data/uniref50/', 'rtest', structure=False)
```
To access the generated sequences: 
```
TODO: function to download gen seqs from Zenodo
```
To analyze the quality of the generations, we look at the amino acid KL divergence ([aa_reconstruction_parity_plot](https://github.com/microsoft/evodiff/blob/main/analysis/plot.py), the secondary structure KL divergence ([evodiff/analysis/calc_kl_ss.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_kl_ss.py)), the model perplexity ([evodiff/analysis/model_perp.py](https://github.com/microsoft/evodiff/blob/main/analysis/model_perp.py)), the Fréchet inception distance ([evodiff/analysis/calc_fid.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_fid.py)), and the hamming distance ([evodiff/analysis/calc_nearestseq_hamming.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_nearestseq_hamming.py)).

We also compute the self-consistency perplexity to evaluate the foldability of generated sequences. To do so, we make use of various tools:
* [TM score](https://zhanggroup.org/TM-score/)
* [Omegafold](https://github.com/HeliXonProtein/OmegaFold)
* [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
* [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/esm/inverse_folding); see this [Jupyter notebook](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb) for setup details.
* [PGP](https://github.com/hefeda/PGP)

Our analysis scripts for iterating over these tools are in the [evodiff/analysis/downstream_scripts](https://github.com/microsoft/evodiff/tree/main/analysis/downstream_bash_scripts) folder. Once we run the scripts in this folder, we analyze the results in [self_consistency_analysis.py](https://github.com/microsoft/evodiff/blob/main/analysis/self_consistency_analysis.py).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos are subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third party trademarks or logos is subject to those third-party's policies.
