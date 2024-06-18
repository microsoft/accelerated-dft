# Project

Accelerated DFT is a cloud-native, GPU-accelerated DFT program for molecular systems engineered by Microsoft offered through [Azure Quantum Elements](https://quantum.microsoft.com/en-us/quantum-elements/product-overview). 

This repository contains benchmarking data and sample inputs for Accelerated DFT as detailed in this [preprint](https://arxiv.org/abs/2406.11185).

- [benchmarks](./benchmarks)
  - Geometries of 329 molecules comprising the test set (geometries.tar.gz)
  - Accelerated-DFT input files and settings used to run calculations on the test set [input_spe-m062x.ipynb](./benchmarks/input_spe-m062x.ipynb) , [input_spe-wb97x.ipynb](./benchmarks/input_spe-wb97x.ipynb)
  - Accelerated-DFT output files for the test set using both M06-2X and wB97x functionals  (m062x_results.tar.gz , wb97x_results.tar.gz)

- [samples](./samples)
  - single point energy example input [spe.ipynb](./samples/bomd.ipynb)
  - single point force example input [spf.ipynb](./samples/bomd.ipynb)
  - geometry optimization example input [go.ipynb](./samples/bomd.ipynb)
  - full Hessian example input [fh.ipynb](./samples/bomd.ipynb)
  - Born-Oppenheimer molecular dynamics example input [bomd.ipynb](./samples/bomd.ipynb)

The sameples included in this project are for reference only. You need an active Accelerated DFT service (which supplies you the access key) to run the samples. 

If you are interested in trying out Accelerated DFT, please [sign up](https://smt.microsoft.com/en-US/AQEPrivatePreviewSignup/) for private preview of Azure Quantum Elements.

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
