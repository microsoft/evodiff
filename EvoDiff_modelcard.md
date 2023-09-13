
# Model Card for EvoDiff

<!-- Provide a quick summary of what the model is/does. -->

Generation of protein sequences and evolutionary alignments via discrete diffusion models.

#  Table of Contents

1. [Model Details](#model-details)
2. [Uses](#uses)
3. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
4. [How To Get Started With the Model](#how-to-get-started-with-the-model)
5. [Training Details](#training-details)
6. [Evaluation](#evaluation)
7. [Environmental Impact](#environmental-impact)
<!-- 8. [Citation](#citation) -->
<!-- 9. [Technical Specifications](#technical-specifications-optional) -->
<!-- 7. [Model Examination](#model-examination) -->
<!-- 11. [Glossary](#glossary-optional) -->
<!-- 12. [More Information](#more-information-optional) -->
<!-- 13. [Model Card Authors](#model-card-authors-optional) -->
<!-- 14. [Model Card Contact](#model-card-contact) -->

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

In this work, we introduce a general-purpose diffusion framework, EvoDiff, that combines evolutionary-scale data with 
the distinct conditioning capabilities of diffusion models for controllable protein generation in sequence space. 
EvoDiff generates high-fidelity, diverse, and structurally-plausible proteins that cover natural sequence and functional
space. Critically, EvoDiff can generate proteins inaccessible to structure-based models, such as those with disordered 
regions, while maintaining the ability to design scaffolds for functional structural motifs, demonstrating the 
universality of our sequence-based formulation. We envision that EvoDiff will expand capabilities in protein engineering
beyond the structure-function paradigm toward programmable, sequence-first design.

- **Developed by:** Sarah Alamdari, Nitya Thakkar, Rianne van den Berg, Alex X. Lu, Nicolo Fusi, Ava P. Amini, Kevin K. Yang
- **Shared by:** Microsoft Research New England
- **Model type:** Diffusion-based protein sequence generation
- **License:** MIT License

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [https://github.com/microsoft/evodiff](https://github.com/microsoft/evodiff)
- **Preprint:** [https://www.biorxiv.org/content/10.1101/2023.09.11.556673v1](https://www.biorxiv.org/content/10.1101/2023.09.11.556673v1)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model is intended for research use. It can be used directly to generate proteins sequences and alignments. We provide checkpoints for all our models so users can run our unconditional and conditional generation scripts. 

We provide a notebook with installation guidance that can be found in [examples/evodiff.ipynb](https://github.com/microsoft/evodiff/tree/main/examples/evodiff.ipynb). It also includes examples on how to generate a smaller number of sequences and MSAs using our models. We recommend following this notebook if you would like to use our models to generate proteins.

To load a model:
```
from evodiff.pretrained import OA_DM_38M

model, collater, tokenizer, scheme = OA_DM_38M()
```
Available models are:
* ``` D3PM_BLOSUM_640M() ```
* ``` D3PM_BLOSUM_38M() ```
* ``` D3PM_UNIFORM_640M() ```
* ``` D3PM_UNIFORM_38M() ```
* ``` OA_DM_640M() ```
* ``` OA_DM_38M() ```
* ``` LR_AR_640M() ```
* ``` LR_AR_38M() ```
* ``` MSA_D3PM_BLOSUM_RANDSUB() ```
* ``` MSA_D3PM_BLOSUM_MAXSUB() ```
* ``` MSA_D3PM_UNIFORM_RANDSUB() ```
* ``` MSA_D3PM_UNIFORM_MAXSUB() ```
* ``` MSA_OA_DM_RANDSUB() ```
* ``` MSA_OA_DM_MAXSUB() ```

Note: if you want to download a `BLOSUM` model, you will first need to download [data/blosum62-special-MSA.mat](https://github.com/microsoft/evodiff/blob/main/data/blosum62-special-MSA.mat).

Please view our [README.md](https://github.com/microsoft/evodiff/blob/main/README.md) for detailed instructions on how to generate sequences and multiple sequence alignments (MSAs) both unconditionally and conditionally.

<!-- ### Downstream Use [optional] -->

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is intended for use on protein sequences. It is not meant for other biological sequences, such as DNA sequences, or natural language.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model will not perform well when trying to generate things that aren't proteins. This includes cases such as trying to generate other biological sequences, such as DNA sequences, or natural language. In other words, the model will perform best on data within the data distribution, which includes protein sequences and multiple sequence alignments (MSAs).

<!-- ### Recommendations -->

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

<!-- {{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}} -->

## How to Get Started with the Model

To download our code, we recommend creating a clean conda environment with python ```v3.8.5```.
```
conda create --name evodiff python=3.8.5
```
In that new environment, install EvoDiff: 
```
pip install evodiff
pip install git+https://github.com/microsoft/evodiff.git # bleeding edge, current repo main branch
```
You will also need to install PyTorch (we tested our models on ` v2.0.1 `), PyTorch Geometric, and PyTorch Scatter.

Our downstream analysis scripts make use of a variety of tools we do not include in our package installation. To run the
scripts, please download the following packages in addition to EvoDiff:
* [TM score](https://zhanggroup.org/TM-score/)
* [Omegafold](https://github.com/HeliXonProtein/OmegaFold)
* [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
* [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/esm/inverse_folding); see this [Jupyter notebook](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb) for setup details.
* [PGP](https://github.com/hefeda/PGP)
* [DISOPRED3](https://github.com/psipred/disopred)
* [DR-BERT](https://github.com/maslov-group/DR-BERT)

We refer to the setup instructions outlined by the authors of those tools.

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

We obtain sequences from the [Uniref50 dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4375400/), which contains 
approximately 42 million protein sequences. 
The Multiple Sequence Alignments (MSAs) are from the [OpenFold dataset](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v2), 
which contains 401,381 MSAs for 140,000 unique Protein Data Bank (PDB) chains and 16,000,000 UniClust30 clusters.
The intrinsically disordered regions (IDR) data was obtained from the [Reverse Homology GitHub](https://github.com/alexxijielu/reverse_homology/).

For the scaffolding structural motifs task, we provide pdb and fasta files used for conditionally generating sequences in the [examples/scaffolding-pdbs](https://github.com/microsoft/evodiff/tree/main/examples/scaffolding-pdbs) folder. We also provide
We provide pdb files used for conditionally generating MSAs in the [examples/scaffolding-msas](https://github.com/microsoft/evodiff/tree/main/examples/scaffolding-msas) folder.

<!-- ### Training Procedure  -->

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

<!-- #### Preprocessing [optional] -->

<!-- {{ preprocessing | default("[More Information Needed]", true)}} -->


<!-- #### Training Hyperparameters -->

<!-- - **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

<!-- #### Speeds, Sizes, Times [optional] -->

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

<!-- {{ speeds_sizes_times | default("[More Information Needed]", true)}} -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

<!-- ### Testing Data, Factors & Metrics -->

### Testing Data

<!-- This should link to a Data Card if possible. -->

To access the UniRef50 test sequences, use the following code:
```
test_data = UniRefDataset('data/uniref50/', 'rtest', structure=False) # To access the test sequences
```

We provide all generated sequences on the [EvoDiff Zenodo](https://zenodo.org/record/8329165).

To download our unconditional generated sequences from `unconditional_generations.csv` file:

```
curl -O https://zenodo.org/record/8329165/files/unconditional_generations.csv?download=1
```

To extract all unconditionally generated sequences created using the EvoDiff-seq `oa_dm_640M` model, run the following code:
```
import pandas as pd
df = pd.read_csv('unconditional_generations.csv', index_col = 0)
subset = df.loc[df['model'] == 'evodiff_oa_dm_640M']
```

Please view our [README.md](https://github.com/microsoft/evodiff/blob/main/README.md#downloading-generated-sequences) for more information about the CSV files containing generated data.

<!-- #### Factors -->

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

### Metrics

To analyze the quality of the generations, we look at:
* amino acid KL divergence ([aa_reconstruction_parity_plot](https://github.com/microsoft/evodiff/blob/main/evodiff/plot.py))
* secondary structure KL divergence ([evodiff/analysis/calc_kl_ss.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_kl_ss.py))
* model perplexity for sequences ([evodiff/analysis/sequence_perp.py](https://github.com/microsoft/evodiff/blob/main/analysis/sequence_perp.py))
* model perplexity for MSAs ([evodiff/analysis/msa_perp.py](https://github.com/microsoft/evodiff/blob/main/analysis/msa_perp.py))
* Fréchet inception distance ([evodiff/analysis/calc_fid.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_fid.py))
* Hamming distance ([evodiff/analysis/calc_nearestseq_hamming.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_nearestseq_hamming.py))
* RMSD score ([analysis/rmsd_analysis.py](https://github.com/microsoft/evodiff/blob/main/analysis/rmsd_analysis.py))

We also compute the self-consistency perplexity to evaluate the foldability of generated sequences. To do so, we make use of various tools:
* [TM score](https://zhanggroup.org/TM-score/)
* [Omegafold](https://github.com/HeliXonProtein/OmegaFold)
* [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
* [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/esm/inverse_folding); see this [Jupyter notebook](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb) for setup details.
* [PGP](https://github.com/hefeda/PGP)
* [DISOPRED3](https://github.com/psipred/disopred)
* [DR-BERT](https://github.com/maslov-group/DR-BERT)

We refer to the setup instructions outlined by the authors of those tools.

Our analysis scripts for iterating over these tools are in the [evodiff/analysis/downstream_bash_scripts](https://github.com/microsoft/evodiff/tree/main/analysis/downstream_bash_scripts) folder. Once we run the scripts in this folder, we analyze the results in [self_consistency_analysis.py](https://github.com/microsoft/evodiff/blob/main/analysis/self_consistency_analysis.py).

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

<!-- ### Results -->

<!-- {{ results | default("[More Information Needed]", true)}} -->

### Summary

We present EvoDiff, a diffusion modeling framework capable of generating high-fidelity, diverse, and novel proteins with the option of conditioning according to sequence constraints. Because it operates in the universal protein design space, EvoDiff can unconditionally sample diverse structurally-plausible proteins, generate intrinsically disordered regions, and scaffold structural motifs using only sequence information, challenging a paradigm in structure-based protein design.

<!-- ## Model Examination [optional] -->

<!-- Relevant interpretability work for the model goes here -->

<!-- {{ model_examination | default("[More Information Needed]", true)}} -->

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

<!-- Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). -->

- **Hardware Type:** `32GB NVIDIA V100` GPUs
- **Hours used:** 4,128 (14 days per sequence model, 10 days per MSA model)
- **Cloud Provider:** Azure
- **Compute Region:** East US
- **Carbon Emitted:** 485.21 kg

<!-- ## Technical Specifications [optional] -->

<!-- ### Model Architecture and Objective -->

<!-- {{ model_specs | default("[More Information Needed]", true)}} -->

<!-- ### Compute Infrastructure -->

<!-- {{ compute_infrastructure | default("[More Information Needed]", true)}} -->

<!-- #### Hardware -->

<!-- {{ hardware | default("[More Information Needed]", true)}} -->

<!-- #### Software -->

<!-- {{ software | default("[More Information Needed]", true)}} -->

<!-- ## Citation -->

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

<!-- **BibTeX:**

TODO

**APA:**

TODO -->

<!-- ## Glossary [optional] -->

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

<!-- {{ glossary | default("[More Information Needed]", true)}} -->

<!-- ## More Information [optional] -->

<!-- {{ more_information | default("[More Information Needed]", true)}} -->

<!-- ## Model Card Authors [optional] -->

<!-- {{ model_card_authors | default("[More Information Needed]", true)}} -->

<!-- ## Model Card Contact -->

<!-- {{ model_card_contact | default("[More Information Needed]", true)}} -->



