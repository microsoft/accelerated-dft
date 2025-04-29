from pyscf.hessian.thermo import harmonic_analysis, thermo
from pyscf import dft
from .libqcschema import *
from pint import UnitRegistry
import numpy as np


# multiplicity of the atom

# experimental value of heat of formation, kcal/mol
# Experimental enthalpy values taken from Curtiss, et. al., J. Chem. Phys. 106, 1063 (1997). 

# the correction to enthalpy at 298K, kcal/mol
# Calculated enthalpy values taken from J. Am. Chem. Soc. 117, 11299 (1995).

# the entropy of the atom at 298K, cal/mol*k
# Entropy values taken from JANAF Thermochemical Tables: M. W. Chase, Jr., C. A. Davies, J. R. Downey, Jr., D. J. Frurip, R. A. McDonald, and A. N. Syverud, J. Phys. Ref. Data 14 Suppl. No. 1 (1985).

atom_info = {'H' :{'multiplicity':2, 'enthalpy_0K':51.63 , 'correction_298K': 1.01, 'entropy_298K': 27.418},
             'Li':{'multiplicity':2, 'enthalpy_0K':37.69 , 'correction_298K': 1.10, 'entropy_298K': 33.169},
             'Be':{'multiplicity':1, 'enthalpy_0K':76.48 , 'correction_298K': 0.46, 'entropy_298K': 32.570},
             'B' :{'multiplicity':2, 'enthalpy_0K':136.20, 'correction_298K': 0.29, 'entropy_298K': 36.672},
             'C' :{'multiplicity':3, 'enthalpy_0K':169.98, 'correction_298K': 0.25, 'entropy_298K': 37.787},
             'N' :{'multiplicity':4, 'enthalpy_0K':112.53, 'correction_298K': 1.04, 'entropy_298K': 36.640},
             'O' :{'multiplicity':3, 'enthalpy_0K':58.99 , 'correction_298K': 1.04, 'entropy_298K': 38.494},
             'F' :{'multiplicity':2, 'enthalpy_0K':18.47 , 'correction_298K': 1.05, 'entropy_298K': 37.942},
             'Na':{'multiplicity':2, 'enthalpy_0K':25.69 , 'correction_298K': 1.54, 'entropy_298K': 36.727},
             'Mg':{'multiplicity':1, 'enthalpy_0K':34.87 , 'correction_298K': 1.19, 'entropy_298K': 8.237 },
             'Al':{'multiplicity':2, 'enthalpy_0K':78.23 , 'correction_298K': 1.08, 'entropy_298K': 39.329},
             'Si':{'multiplicity':3, 'enthalpy_0K':106.6 , 'correction_298K': 0.76, 'entropy_298K': 40.148},
             'P' :{'multiplicity':4, 'enthalpy_0K':75.42 , 'correction_298K': 1.28, 'entropy_298K': 39.005},
             'S' :{'multiplicity':3, 'enthalpy_0K':65.66 , 'correction_298K': 1.05, 'entropy_298K': 40.112},
             'Cl':{'multiplicity':2, 'enthalpy_0K':28.59 , 'correction_298K': 1.10, 'entropy_298K': 39.481},
            }


# define N as Avogadro constant
u = UnitRegistry()
N = u.Quantity(1, u.avogadro_constant)


def convert_energy_au_to_kJmol(energy_au: float):
    
    energy_au = energy_au * u.hartree 
    energy_kJmol = energy_au.to('kilojoule') *  N.to_base_units()
    
    return energy_kJmol.magnitude


def convert_energy_kcalmol_to_au(energy_kcalmol: float):
    
    energy_kcalmol = energy_kcalmol * u.kilocalorie
    energy_au = energy_kcalmol.to('hartree') /  N.to_base_units()
    
    return energy_au.magnitude


def get_optmized_xyz(json):
    
    BtoA = 0.5291772109

    syms = np.array(json["final_molecule"]["symbols"])
    geo = np.array(json["final_molecule"]["geometry"])*BtoA
    NAtoms = len(syms)
    geo = np.reshape(geo, (NAtoms,3))
    
    # Concatenate the symbols and coordinates along the second axis
    combined = np.concatenate([syms[:, np.newaxis], geo], axis=1)
    
    # Convert the combined array to a string with spaces as separators
    output = np.array2string(combined, separator=' ', max_line_width=np.inf)
    
    # Remove the brackets and quotes from the output string
    coords = output.replace('[', '').replace(']', '').replace("'", '')

    # Remove leading whitespace from each line
    coords = '\n'.join([line.strip() for line in coords.split('\n')])

    xyz = f'{NAtoms}\n\n'

    return xyz+coords


def get_composition(xyz):
    
    composition = {}
    for line in xyz.split('\n')[2:]:
        element = line.split()[0]
        if element in composition:
            composition[element] += 1
        else:
            composition[element] = 1
            
    return composition


def get_cbs_result(e_0, e_1, e_2):
    return (e_0*e_2-e_1**2)/(e_0-2*e_1+e_2)


def get_34cbs_result(e_3, e_4):
    # using def2 basis set, a=7.88, ref orca manual 5.0.3 eq8.1
    from math import exp, sqrt
    return (e_4*exp(-7.88*sqrt(3))-e_3*exp(-7.88*sqrt(4)))/(exp(-7.88*sqrt(3))-exp(-7.88*sqrt(4)))


def get_thermo_dict(hessian_results, functional):
    
    # Create Pyscf Molecule
    scf_dict, mol = load_qcschema_mol_scf(hessian_results)
    mol.unit = 'B' # QCSchema outputs in Bohr AU
    mol.build()

    # Create DFT object
    ks = dft.RKS(mol)
    ks.xc = functional

    # optional
    ks.grids.level = 4
    ks.init_guess = 'minao'
    ks.conv_tol=1e-8
    ks.direct_scf = True
    ks.direct_scf_tol = 1e-12
    # REQUIRED:Load 4 key pieces of info we got from json into DFT object 
    ks.mo_coeff = scf_dict["mo_coeff"]
    ks.mo_energy = scf_dict["mo_energy"]
    ks.mo_occ = scf_dict["mo_occ"]
    ks.e_tot = scf_dict["e_tot"]

    # Compute Hessian
    h = load_qcschema_hessian(hessian_results)

    # Compute Vibrational Frequencies
    freq = harmonic_analysis(mol,h)

    # Compute Thermochemistry
    thermochem = thermo(ks,freq['freq_au'], 298.15)

    return thermochem


def get_heat_of_formation(energies, thermo, composition):

    heat_of_formation = {'unit': 'kJ/mol'}
    
    # atomization energy
    sum_atom_energy = 0
    for element in composition:
        atom_energy = energies[element]
        sum_atom_energy += composition[element] * atom_energy

    # internal energy is E_0K = E_elec + ZPE
    E_internal = thermo['E_0K']
    atomization_energy = sum_atom_energy - E_internal

    # Heat of formation at 0K of the molecule
    sum_atom_hof_0K = 0
    for atom in composition:
        atom_hof_0K = atom_info[atom]['enthalpy_0K']
        sum_atom_hof_0K += composition[atom] * atom_hof_0K

    sum_atom_hof_0K_au = convert_energy_kcalmol_to_au(sum_atom_hof_0K)
    hof_0K = sum_atom_hof_0K_au - atomization_energy
    
    hof_0K_kJmol = convert_energy_au_to_kJmol(hof_0K)

    heat_of_formation['0K'] = hof_0K_kJmol

    # sum of ethalpy correction at 298K for constituent atoms
    sum_atom_hof_298K_correction = 0
    for atom in composition:
        atom_hof_298K_correction = atom_info[atom]['correction_298K']
        sum_atom_hof_298K_correction += composition[atom] * atom_hof_298K_correction
        
    sum_atom_hof_298K_correction_au = convert_energy_kcalmol_to_au(sum_atom_hof_298K_correction)
    
    hof_298K_correction = thermo['H_tot'] - E_internal
    hof_298K = hof_0K + hof_298K_correction - sum_atom_hof_298K_correction_au
    
    hof_298K_kJmol = convert_energy_au_to_kJmol(hof_298K)
    heat_of_formation['298K'] = hof_298K_kJmol

    return heat_of_formation


