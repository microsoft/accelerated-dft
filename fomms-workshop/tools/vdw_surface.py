# Adapted and Extended from Psi4NumPy by Microsoft under BSD-3.
# Original credit:
#__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
#__license__   = "BSD-3-Clause"
#
import numpy as np

"""
A sript to generate van der Waals surface of molecules.
"""

def surface(n):
    """Computes approximately n points on unit sphere. Code adapted from GAMESS.

    Parameters
    ----------
    n : int
        approximate number of requested surface points

    Returns
    -------
    ndarray
        numpy array of xyz coordinates of surface points
    """

    u = []
    eps = 1e-10
    nequat = int(np.sqrt(np.pi*n))
    nvert = int(nequat/2)
    nu = 0
    for i in range(nvert+1):
        fi = np.pi*i/nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = int(nequat*xy+eps)
        if nhor < 1:
            nhor = 1
        for j in range(nhor):
            fj = 2*np.pi*j/nhor
            x = np.cos(fj)*xy
            y = np.sin(fj)*xy
            if nu >= n:
                return np.array(u)
            nu += 1
            u.append([x, y, z])
    return np.array(u)

def vdw_surface(coordinates, elements, scale_factor, density, input_radii):
    """Computes points outside the van der Waals surface of molecules.

    Parameters
    ----------
    coordinates : ndarray
        cartesian coordinates of the nuclei, in units of angstrom
    elements : list
        The symbols (e.g. C, H) for the atoms
    scale_factor : float
        The points on the molecular surface are set at a distance of
        scale_factor * vdw_radius away from each of the atoms.
    density : float
        The (approximate) number of points to generate per square angstrom
        of surface area. 1.0 is the default recommended by Kollman & Singh.
    input_radii : dict
        dictionary of PySCF (or user's) defined VDW radii

    Returns
    -------
    radii : dict
        A dictionary of scaled VDW radii
    surface_points : ndarray
        array of the coordinates of the points on the surface

    """
    radii = {}
    surface_points = []
    # scale radii
    for i in elements:
        if i in input_radii.keys():
            radii[i] = input_radii[i] * scale_factor
        else:
            raise KeyError('%s is not a supported element; ' %i
                         + 'use the "VDW_RADII" option to add '
                         + 'its van der Waals radius.')
    # loop over atomic coordinates
    for i in range(len(coordinates)):
        # calculate approximate number of ESP grid points
        n_points = int(density * 4.0 * np.pi* np.power(radii[elements[i]], 2))
        # generate an array of n_points in a unit sphere around the atom
        dots = surface(n_points)
        # scale the unit sphere by the VDW radius and translate
        dots = coordinates[i] + radii[elements[i]] * dots
        for j in range(len(dots)):
            save = True
            for k in range(len(coordinates)):
                if i == k:
                    continue
                # exclude points within the scaled VDW radius of other atoms
                d = np.linalg.norm(dots[j] - coordinates[k])
                if d < radii[elements[k]]:
                    save = False
                    break
            if save:
                surface_points.append(dots[j])
    return np.array(surface_points), radii
