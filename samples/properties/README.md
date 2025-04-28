# Properties 
Accelerated DFT produces QCSchema formatted output. Energies (total energies, repulsions energies etc) and useful system information is output into the QCSchema json file, while orbitals (MO coefficients and basis information) are output to a QCSchema compliant hdf5 file.
These, as well as the molecule information (and the hessian, if computed) can be read into PySCF, in order to use additional functionality and to compute properties, such as Mulliken population analysis, frequency calculations and thermochemistry.    

In order to use the scripts below, a conda environment must first be created.  `conda env create --file env_properties.yaml` will create the environment 'adft_properties', which can be activated using `conda activate adft_properties`.

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







