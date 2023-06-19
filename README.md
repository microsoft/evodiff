# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README


### Installation
```
cd dms
conda env create -f environment.yml
conda activate dms
pip install -e .
```
We obtain sequences from the [Uniref50 dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4375400/), which contains approximately 45 million protein sequences. The Multiple Sequence Alignments (MSAs) are from the [OpenFold dataset](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v2), containing MSAs for 132,000 unique Protein Data Bank (PDB) chains.

### Loading pretrained models
To load a model:
```
from dms.pretrained import OA_AR_640M

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
* ``` CARP_38M() ```
* ``` CARP_640M() ```
* ``` ESM1b_640M() ```

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

Note that you can only start-query or start-msa, not both. To generate unconditionally, omit the flags (see example in above section).

### Analysis of generations
To access the test sequences:
```
test_data = UniRefDataset('data/uniref50/', 'rtest', structure=False)
```
To access the generated sequences: 
```
TODO
```
To analyze the quality of the generations, we look at the amino acid KL divergence ([aa_reconstruction_parity_plot](https://github.com/microsoft/DMs/blob/main/analysis/plot.py), the secondary structre KL divergence ([DMs/analysis/calc_kl_ss.py](https://github.com/microsoft/DMs/blob/main/analysis/calc_kl_ss.py)), the Fréchet inception distance ([DMs/analysis/calc_fid.py](https://github.com/microsoft/DMs/blob/main/analysis/calc_fid.py)), and the hamming distance ([DMs/analysis/calc_nearestseq_hamming.py](https://github.com/microsoft/DMs/blob/main/analysis/calc_nearestseq_hamming.py)).

We also compute the self-consistency perplexity to evaluate the foldability of generated sequences (TODO: file).

TODO: FIX CODE ABOVE + check file paths for analysis functions! @sarah

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
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
