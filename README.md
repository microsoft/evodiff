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
TODO: add what data we use

### Loading pretrained models
To load a model:
```
from dms.pretrained import D3PM_BLOSUM_640M, OA_AR_640M

checkpoint = OA_AR_640M()
model, collater, tokenizer, scheme = checkpoint
```
An example of loading a model from a checkpoint is in the DMs/analysis/model_perp.py file

### Unconditional sequence generation
TODO: how to use the generate.py and generate-msa.py scripts (and where/how to access checkpoints)

### Conditional sequence generation
TODO

### Downstream analysis tasks
TODO: for every analysis mentioned in table 1, point to each function in analysis files; self-consistency

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
