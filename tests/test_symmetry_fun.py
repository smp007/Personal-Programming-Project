import pytest
import os
import numpy as np
from src.nnpp.symmetry import(
    cutoff_function,d_cutoff_function,
    radial_distribution,angular_distribution,
    psi,xhi,phi,
    symmetry_fun_r,symmetry_fun_a,symmetry_function,
    d_symmetry_fun_r,d_symmetry_fun_a,
    Atom,

)

#---------------------------------------------------------------------------------------------------
#cutoff function and its derivative
#---------------------------------------------------------------------------------------------------

def test_cutoff_function_inside_sphere():
    R1 = 2
    R_c = 6
    assert cutoff_function(R1,R_c) == 0.75

def test_cutoff_function_outside_sphere():
    R2 = 10
    R_c = 6
    assert cutoff_function(R2,R_c) == 0

def test_d_cutoff_function_inside_sphere():
    R1 = 2
    R_c = 6
    assert round(d_cutoff_function(R1,R_c),5) == -0.22672

def test_d_cutoff_function_outside_sphere():
    R2 = 10
    R_c = 6
    assert d_cutoff_function(R2,R_c) == 0

#---------------------------------------------------------------------------------------------------

def test_radial_distribution():
    R_array = np.asarray([1,3,5,7,9])
    R_s = 0
    R_c = 6
    eeta = 1
    assert round(radial_distribution(R_array,R_s,R_c,eeta),6)==0.343298


def test_angular_distribution():
    theeta = 10
    R1,R2,R3 = 4,4,4
    eeta,lamda,zeta = 0,-1,0
    assert round(angular_distribution(theeta,R1,R2,R3,eeta,lamda,zeta),6) == 0.033604


#components in the derivative of symmetry function--------------------------------------------------
def test_psi():
    R1,R2,theeta,eeta,lamda,zeta = 1,2,5,0,-1,1
    assert round(psi(theeta,R1,R2,eeta,lamda,zeta),6) == 0.697995

def test_phi():
    R1,R2,theeta,eeta,lamda,zeta = 1,2,5,0,1,1
    assert round(phi(theeta,R1,R2,eeta,lamda,zeta),6) == -0.389511

def test_xhi():
    R1,R_c = 4,6
    assert round(xhi(R1,R_c),5)== -0.01417

#shape check of symmetry function-------------------------------------------------------------------

atom_data = [['Ti', 1.76228668, 1.84079395, 2.39690881],
             ['Ti', 0.0, 0.0, 0.0],
             ['O', 1.76228668, 0.71748727, 0.93424445],
             ['O', 1.76228668, 2.96410063, 3.85957312],
             ['O', 0.0, 1.12328094, 3.33115431], 
             ['O', 0.0, 2.55830696, 1.46266332]]

atoms = (([Atom(atomtype,x,y,z) for (atomtype,x,y,z) in atom_data])) 

def test_radial_part_shape():
    assert symmetry_fun_r(atoms).shape == (6,16)

def test_angular_part_shape():
    assert symmetry_fun_a(atoms,'Ti','O').shape == (6,18) and \
           symmetry_fun_a(atoms,'Ti','Ti').shape == (6,18) and \
           symmetry_fun_a(atoms,'O','O').shape == (6,18)

def test_symmetry_fun_shape():
    assert symmetry_function(atom_data)[0].shape == (6,70)

def test_d_radial_part_shape():
    assert d_symmetry_fun_r(atoms).shape == (6,16,3)

def test_d_angular_part_shape():
    assert d_symmetry_fun_a(atoms,'Ti','O').shape == (6,18,3) and \
           d_symmetry_fun_a(atoms,'Ti','Ti').shape == (6,18,3) and \
           d_symmetry_fun_a(atoms,'O','O').shape == (6,18,3)

def test_d_symmetry_fun_shape():
    assert symmetry_function(atom_data)[1].shape == (6,70,3)

    
