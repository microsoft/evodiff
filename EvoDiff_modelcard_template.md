
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
8. [Citation](#citation)
<!-- 9. [Technical Specifications](#technical-specifications-optional) -->
<!-- 7. [Model Examination](#model-examination) -->
<!-- 11. [Glossary](#glossary-optional) -->
<!-- 12. [More Information](#more-information-optional) -->
<!-- 13. [Model Card Authors](#model-card-authors-optional) -->
<!-- 14. [Model Card Contact](#model-card-contact) -->

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

In this work, we train and evaluate a series of discrete diffusion models for both unconditional and conditional generation of single protein sequences as well as multiple sequence alignments (MSAs). We test both order-agnostic autoregressive diffusion and discrete denoising diffusion probabilistic models for protein sequence generation; formulate unique, bio-inspired corruption schemes for both classes of models; and evaluate the quality of generated samples for fidelity, diversity, and structural plausibility. 


- **Developed by:** Sarah Alamdari, Nitya Thakkar, Rianne van den Berg, Nicolo Fusi, Ava P. Amini, Kevin K. Yang
- **Shared by:** Microsoft Research
- **Model type:** Diffusion-based protein sequence generation
- **License:** MIT License

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [https://github.com/microsoft/evodiff](https://github.com/microsoft/evodiff)
- **Paper:** TODO

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model can be used directly to generate proteins sequences and alignments. We provide checkpoints for all our models so users can run our unconditional and conditional generation scripts. We also provide a [notebook](https://github.com/microsoft/evodiff/blob/main/evodiff.ipynb) for easy access to run our code.

<!-- ### Downstream Use [optional] -->

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is intended for use on protein sequences. It is not meant for other biological sequences, such as DNA sequences, or regular language.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

No forseeable issues

<!-- ### Recommendations -->

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

<!-- {{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}} -->

## How to Get Started with the Model

To download our model, run:
```
pip install evodiff
pip install git+https://github.com/microsoft/protein-sequence-models.git # bleeding edge, current repo main branch
```

To set up a working environment, run:
```
cd evodiff
conda env create -f environment.yml
conda activate evodiff
pip install -e .
```
You will also need to install PyTorch. We tested our models on `v2.0.1`.

To load a model, run:
```
from evodiff.pretrained import OA_AR_38M

model, collater, tokenizer, scheme = OA_AR_38M()
```

Available models are:

- ` D3PM_BLOSUM_640M()`
- ` D3PM_BLOSUM_38M()`
- ` D3PM_UNIFORM_640M()`
- ` D3PM_UNIFORM_38M()`
- ` OA_AR_640M()`
- ` OA_AR_38M()`
- ` LR_AR_640M()`
- ` LR_AR_38M()`
- ` MSA_D3PM_BLOSUM()`
- ` MSA_D3PM_UNIFORM()`
- ` MSA_D3PM_OA_AR_RANDSUB()`
- ` MSA_D3PM_OA_AR_MAXSUB()`

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

We obtain sequences from the [Uniref50 dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4375400/), which contains approximately 45 million protein sequences. The Multiple Sequence Alignments (MSAs) are from the [OpenFold dataset](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v2), containing MSAs for 132,000 unique Protein Data Bank (PDB) chains.

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

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

To access the test data:

`test_data = UniRefDataset('data/uniref50/', 'rtest', structure=False)`

and to access the generated sequences:

`TODO`

<!-- #### Factors -->

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

#### Metrics

To analyze the quality of the generations, we look at the amino acid KL divergence ([aa_reconstruction_parity_plot](https://github.com/microsoft/evodiff/blob/main/analysis/plot.py)), the secondary structure KL divergence ([evodiff/analysis/calc_kl_ss.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_kl_ss.py)), the model perplexity ([evodiff/analysis/model_perp.py](https://github.com/microsoft/evodiff/blob/main/analysis/model_perp.py)), the Fréchet inception distance ([evodiff/analysis/calc_fid.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_fid.py)), and the hamming distance ([evodiff/analysis/calc_nearestseq_hamming.py](https://github.com/microsoft/evodiff/blob/main/analysis/calc_nearestseq_hamming.py)).

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

<!-- ### Results -->

<!-- {{ results | default("[More Information Needed]", true)}} -->

#### Summary: UPDATE LATER

Our discrete diffusion models generate high-fidelity, diverse, and structurally-plausible sequences and outperform existing methods when controlling for datasets and architectures. We find that OA-ARDM generally outperforms D3PM, and that D3PM corruption with a BLOSUM transition matrix does not consistently outperform a uniform transition matrix.

<!-- ## Model Examination [optional] -->

<!-- Relevant interpretability work for the model goes here -->

<!-- {{ model_examination | default("[More Information Needed]", true)}} -->

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

<!-- Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). -->

- **Hardware Type:** `38-V100s(32GB)`
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

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

TODO

**APA:**

TODO

<!-- ## Glossary [optional] -->

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

<!-- {{ glossary | default("[More Information Needed]", true)}} -->

<!-- ## More Information [optional] -->

<!-- {{ more_information | default("[More Information Needed]", true)}} -->

<!-- ## Model Card Authors [optional] -->

<!-- {{ model_card_authors | default("[More Information Needed]", true)}} -->

<!-- ## Model Card Contact -->

<!-- {{ model_card_contact | default("[More Information Needed]", true)}} -->



