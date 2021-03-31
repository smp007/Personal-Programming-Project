"""
====================================================================================================
Reader function unit tests
----------------------------------------------------------------------------------------------------
Here we consider the xsf file named structure0001.xsf to check whether the reader function works in
the correct way.
====================================================================================================

"""
import pytest
import numpy as np
import os 

from nnpp.reader import *
path = './dataset_TiO2'
file_list = sorted(os.listdir(path))
file_1 = file_list[0]


def test_reader_outputs_correct_energy():
    E,_,_ = xsf_reader(file_1)
    assert E == -19960.66173260

def test_reader_outputs_correct_n_value():
    _,n,_ = xsf_reader(file_1)
    assert n == 24

def test_reader_outputs_correct_data():
    _,_,atoms = xsf_reader(file_1)
    assert len(atoms) == 24 and len(atoms[0]) == 4