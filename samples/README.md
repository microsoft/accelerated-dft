# Examples  

Accelerated DFT jobs can be submitted directly from the Azure Portal, by uploading submission notebooks to 'Notebooks' section. 
Jobs can also be submitted from a local environment, such as your own laptop. To do this, a local conda environment must first be set up using the [env.yaml](./env.yaml) file provided. On the command line enter `conda env create --file env.yaml` to create the conda environment 'madft', this will take a few minutes but only needs to be done once. When submitting jobs using the notebooks below, make sure to select kernel 'madft' to allow the notebook to use this environment. If submitting via a Python script rather than a notebook (e.g. [dft_job.py](./dft_job.py)), activate the environment before running the script `conda activate madft`.  

## Simple First Jobs  
Simple first jobs can be submitted using the Python script [dft_job.py](./dft_job.py) or the notebook [Submit_spe.ipynb](./Submit_spe.ipynb).

[dft_job.py](./dft_job.py) - Python script for single point energy (spe)

[Submit_spe.ipynb](./Submit_spe.ipynb) - simple energy (spe) calculation.

[Submit_spe_with_xyz.ipynb](./Submit_spe_with_xyz.ipynb) - same simple energy calculation but geometry input is in the form of an xyz file.

[Submit_spf.ipynb](./Submit_spf.ipynb) - forces (spf) calculation.

[Submit_go.ipynb](./Submit_go.ipynb) - geometry optimization (go).

[Submit_fh.ipynb](./Submit_fh.ipynb) - full hessian (fh) calculation.

[Submit_bomd.ipynb](./Submit_bomd.ipynb) - BOMD simulation.

[Submit_Solvent_PCM.ipynb](./Submit_Solvent_PCM.ipynb) - Energy calculation with solvent (via PCM solvation).

[Submit_go_constrained.ipynb](./Submit_go_constrained.ipynb) - a constrained geometry optimization.

[Submit_go_transition_state.ipynb](./Submit_go_transition_state.ipynb) - a transition state optimization.

These examples show the general structure of the input and can be easily adjusted using the documented parameters (see [docs](../docs)). 
 
## More Advanced Workflow  

[Submit_Heat_of_Formation.ipynb](./Submit_Heat_of_Formation.ipynb) - shows a more involved workflow to compute the heat of formation, that also uses a complete basis set extrapolation. 

## Properties 
If one uses 'requireWaveFunction': True' in the Accelerated DFT input, then the orbitals, energies and MO coefficients will be written into the QCSchema formatted output.
These, as well as the molecule information (and the hessian, if computed) can be read into PySCF, in order to use additional functionality and to compute properties, such as Mulliken population analysis, frequency calculations and thermochemistry.    

In order to use the scripts below, a conda environment must first be created.  `conda env create --file env_properties.yaml` will create the environment 'madft_properties', which can be activated using `conda activate madft_properties`.

This directory contains example scripts for loading information into PySCF and computing a range of properties. For example:
 

[load_qcschema_result_compute_Dipole_Moment_uks.py](./load_qcschema_result_compute_Dipole_Moment_uks.py) - computes the molecular dipole moment.

[load_qcschema_result_compute_Dipole_Moment_rks.py](./load_qcschema_result_compute_Dipole_Moment_rks.py) - computes the molecular dipole moment.

[load_qcschema_result_compute_RESP.py](./load_qcschema_result_compute_RESP.py) - generates RESP atomic charges from the molecular electrostatic potential.

[load_qcschema_result_compute_CHELPG.py](./load_qcschema_result_compute_CHELPG.py) - generates CHELPG atomic charges from the molecular electrostatic potential.

[load_qcschema_result_compute_Mulliken_Population.py](./load_qcschema_result_compute_Mulliken_Population.py) - performs Mulliken population analysis. Provides partitioning of charge to atoms (and spin for open-shell systems).

[load_qcschema_result_compute_Spin.py](./load_qcschema_result_compute_Spin.py) - computes the spin values and spin density (via Mulliken population analysis).  

[load_qcschema_result_compute_LocalizedOrbitals.py](./load_qcschema_result_compute_LocalizedOrbitals.py) - loads MOs and generates localized orbitals using several approaches.

[load_qcschema_result_compute_Frequencies_and_Thermochemistry.py](./load_qcschema_result_compute_Frequencies_and_Thermochemistry.py) - computes frequencies and normal modes as well as thermochemistry info.

[load_qcschema_result_compute_IR.py](./load_qcschema_result_compute_IR.py) - computes the InfraRed spectrum.

[load_qcschema_result_compute_NMR.py](./load_qcschema_result_compute_NMR.py) - computes the NMR chemical shifts.

[load_qcschema_result_compute_MEP.py](./load_qcschema_result_compute_MEP.py) - computes the Molecular Electrostatic Potential (MEP/MESP)

[load_qcschema_result_compute_Polarizability.py](./load_qcschema_result_compute_Polarizability.py) - computes the  polarizability of the molecule.

[load_qcschema_result_compute_gTensor.py](./load_qcschema_result_compute_gTensor.py) - computes the g-tensor.

[load_qcschema_result_compute_HFC.py](./load_qcschema_result_compute_HFC.py) - computes the hyper-fine coupling (HFC).

[load_qcschema_result_compute_cubegen.py](load_qcschema_result_compute_cubegen.py) - generates cube files for density and an MO, for external viewing.







