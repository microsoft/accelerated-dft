"""
Reads a wavefunction from a HDF5 file and converts it to a QC-Schema JSON.
"""

import h5py
import numpy as np
import argparse
from typing import Any, Dict
import os
from qcelemental import periodictable
from qcelemental.models.basis import BasisCenter, BasisSet, ECPPotential, ElectronShell


def read_basis(h5f, qc_wavefunction):
    """Reads a basis set from a HDF5 file and returns it as a QC-Schema dictionary."""
    basis_dict = {}
    basis_dict["schema_name"] = "qcschema_basis"
    basis_dict["schema_version"] = 1
    basis_dict["name"] = h5f["/wavefunction/basis/name"][()].decode()
    basis_dict["nbf"] = h5f['wavefunction/basis/nbf'][()]

    symbols = [ x.decode() for x in h5f["/molecule/symbols"][()] ]
    numbers = [periodictable.to_Z(x) for x in symbols]
    atoms = list(range(len(numbers)))

    basis_dict["atom_map"] = [ str(x) for x in atoms ]

    #assert h5f['wavefunction/basis/schema_name'][()].decode() == "madft_basis"
    #assert h5f['wavefunction/basis/schema_version'][()] == 1

    # read the raw shells data from file
    electron_shells = []
    pure = bool(h5f['/wavefunction/basis/pure'][()])
    if "/wavefunction/basis/electron_shells" in h5f:
        bf_idx = 0
        for sh in h5f["/wavefunction/basis/electron_shells"][()]:

            L = sh[1]
            size = 2 * L + 1 if pure else (L + 1) * (L + 2) // 2

            electron_shells.append(
                (
                    sh[0],
                    ElectronShell(
                        angular_momentum=[sh[1]],
                        harmonic_type="spherical" if pure else "cartesian",
                        exponents=tuple(sh[3]),
                        coefficients=[tuple(sh[4])],
                    ),
                    sh[2],
                    np.arange(bf_idx, bf_idx + size),
                )
            )
            bf_idx += size

        restricted = qc_wavefunction['restricted'] = bool(h5f['/wavefunction/restricted'][()])
        # hessian job doesn't require reorder 
        if h5f['driver'][()].decode() == "hessian":
            pass
        else:
            electron_shells = sorted(electron_shells, key=lambda x: (
                x[0],
                x[1].angular_momentum[0],
                -x[2],
                -x[1].exponents[0], -x[1].coefficients[0][0]
                ))
            ao_order = np.hstack([sh[-1] for sh in electron_shells])
            qc_wavefunction['scf_fock_a'] = qc_wavefunction['scf_fock_a'][ao_order[:,None], ao_order]
            qc_wavefunction['scf_density_a'] = qc_wavefunction['scf_density_a'][ao_order[:,None], ao_order]
            qc_wavefunction['scf_orbitals_a'] = qc_wavefunction['scf_orbitals_a'][ao_order[:,None]]
            if not restricted:
                qc_wavefunction['scf_fock_b'] = qc_wavefunction['scf_fock_b'][ao_order[:,None], ao_order]
                qc_wavefunction['scf_density_b'] = qc_wavefunction['scf_density_b'][ao_order[:,None], ao_order]
                qc_wavefunction['scf_orbitals_b'] = qc_wavefunction['scf_orbitals_b'][ao_order[:,None]]
        electron_shells = [ (x[0],x[1]) for x in electron_shells ]

    ecp_potentials = None
    if "/wavefunction/basis/ecp_shells" in h5f:
        ecp_potentials = [
            (
                sh[0],
                ECPPotential(
                    ecp_type="scalar",
                    angular_momentum=tuple(sh[1]),
                    gaussian_exponents=tuple(sh[3]),
                    coefficients=[tuple(sh[4])],
                    r_exponents=tuple(sh[5]),
                )
            )
            for sh in h5f["/wavefunction/basis/ecp_shells"][()]
        ]

    ecp_cores = {}
    if "/wavefunction/basis/ecp_cores" in h5f:
        ecp_cores = {
            atom: ncores
            for atom, ncores in h5f["/wavefunction/basis/ecp_cores"][()]
        }

    center_data = {}
    for idx in atoms:
        electron_shells_idx = None
        if electron_shells is not None:
            electron_shells_idx = [
                sh for idx_,sh in electron_shells if idx_ == idx
            ]

        ecp_potentials_idx = None
        if ecp_potentials is not None:
            ecp_potentials_idx = [
                p for idx_,p in ecp_potentials if idx_ == idx
            ]
            
        center_data[str(idx)] = BasisCenter(
            electron_shells=electron_shells_idx,
            ecp_potentials=ecp_potentials_idx,
            ecp_electrons=ecp_cores.get(idx, 0),
        )

    basis_dict["center_data"] = center_data
    qc_wavefunction["basis"] = BasisSet(**basis_dict)


def read_hdf5_wavefunction(file_name: str) -> Dict[str, Any]:
    qc_wavefunction = {}
    with h5py.File(file_name, "r") as h5f:
        # check that wavefunction field is present
        if "/wavefunction" not in h5f:
            raise ValueError("No wavefunction found in HDF5 file")
        # check if the wavefunction is restricted or not
        restricted = qc_wavefunction["restricted"] = bool(
            h5f["/wavefunction/restricted"][()]
        )
        for key in [
            "fock_a",
            "fock_b",
            "density_a",
            "density_b",
            "orbitals_a",
            "orbitals_b",
            "eigenvalues_a",
            "eigenvalues_b",
            "occupations_a",
            "occupations_b",
            "scf_fock_a",
            "scf_fock_b",
            "scf_density_a",
            "scf_density_b",
            "scf_orbitals_a",
            "scf_orbitals_b",
            "scf_eigenvalues_a",
            "scf_eigenvalues_b",
            "scf_occupations_a",
            "scf_occupations_b",
        ]:
            if restricted and key.endswith("_b"):
                # if restricted, we reference the alpha values for beta
                # without storing them twice
                key_a = key.replace("_b", "_a")
                qc_wavefunction[key] = key_a
            else:  # alpha values and unrestricted values
                qc_wavefunction[key] = h5f[f"/wavefunction/{key}"][()]

        # reshape matrices:
        nmo = h5f["/properties/calcinfo_nmo"][()]
        if len(qc_wavefunction["scf_eigenvalues_a"]) != nmo:
            raise ValueError("Inconsistent number of eigenvalues and nmo")
        if not restricted:
            if len(qc_wavefunction["scf_eigenvalues_a"]) != len(
                qc_wavefunction["scf_eigenvalues_b"]
            ):
                raise ValueError(
                    "Number of alpha and beta SCF eigenvalues do not match"
                )
        nbf = h5f["/wavefunction/basis/nbf"][()]
        qc_wavefunction["scf_fock_a"] = qc_wavefunction["scf_fock_a"].reshape(nbf, nbf)
        qc_wavefunction["scf_density_a"] = qc_wavefunction["scf_density_a"].reshape(nbf, nbf)
        qc_wavefunction["scf_orbitals_a"] = qc_wavefunction["scf_orbitals_a"].reshape(nbf, nmo)
        if not restricted:
            qc_wavefunction["scf_fock_b"] = qc_wavefunction["scf_fock_b"].reshape(nbf, nbf)
            qc_wavefunction["scf_density_b"] = qc_wavefunction["scf_density_b"].reshape(nbf, nbf)
            qc_wavefunction["scf_orbitals_b"] = qc_wavefunction["scf_orbitals_b"].reshape(nbf, nmo)

        read_basis(h5f, qc_wavefunction)
        return qc_wavefunction


if __name__ == "__main__":
    # parse filename from command line
    parser = argparse.ArgumentParser(
        description="Reads a wavefunction from a HDF5 file and converts it to a QC-Schema JSON."
    )
    parser.add_argument("filename", type=str, help="HDF5 file to read")
    args = parser.parse_args()
    filename = args.filename
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    qc_wavefunction = read_hdf5_wavefunction(filename)
    print(qc_wavefunction.keys())
    print(qc_wavefunction['basis'].dict().keys())
    print(qc_wavefunction)
