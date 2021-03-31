"""
====================================================================================================
Activations unit tests
----------------------------------------------------------------------------------------------------
Uses a particular input to the activations and then,compares the output of the activations functions 
and their derivatives with preknown output value.
====================================================================================================
"""

import pytest
import numpy as np
from src.nnpp.neural_network import(
    sigmoid,d_sigmoid,
    ReLU,d_ReLU,
    tanh,d_tanh,
    linear,d_linear,
)
#-------------------------------------------------------------------------------
#Sigmoid and its derivative
#-------------------------------------------------------------------------------
def test_sigmoid_integer():
    assert sigmoid(0) == 0.5

def test_sigmoid_array_1():
    x = np.array([0,0])
    z = np.array([0.5,0.5])
    assert (sigmoid(x) == z).all()

def test_sigmoid_array_2():
    x = np.array([-5,-1,0,1,5])
    z = np.array([0.00669285,0.26894142,0.5,0.73105858,0.99330715])
    assert np.isclose(sigmoid(x),z).all()

def test_d_sigmoid_integer():
    assert d_sigmoid(0) == 0

def test_d_sigmoid_array_1():
    x = np.array([0,0])
    z = np.array([0,0])
    assert (d_sigmoid(x) == z).all()

def test_d_sigmoid_array_2():
    x = np.array([-5,-1,0,1,5])
    z = np.array([-30,-2,0,0,-20])
    assert np.isclose(d_sigmoid(x),z).all()

#-------------------------------------------------------------------------------
#Re-LU and its derivative
#-------------------------------------------------------------------------------

def test_ReLU_positive():
    assert ReLU(1) == 1

def test_ReLU_negative():
    assert ReLU(-1) == 0

def test_ReLU_array():
    x = np.array([5,6,-5,-2])
    z = np.array([5,6,0,0])
    assert (ReLU(x) == z).all()


def test_d_ReLU_positive():
    assert d_ReLU(1) == 1

def test_d_ReLU_negative():
    assert ReLU(-1) == 0

def test_d_ReLU_array():
    x = np.array([5,6,-5,-2])
    z = np.array([5,6,0,0])
    assert (ReLU(x) == z).all()

#-------------------------------------------------------------------------------
#tanh and its derivative
#-------------------------------------------------------------------------------

def test_tanh_array_1():
    x = np.array([0,0])
    z = np.array([0,0])
    assert (tanh(x) == z).all()

def test_tanh_array_2():
    x = np.array([-10,-5,-1,0,1,5,10])
    z = np.array([-1.,-0.9999092,-0.76159416,0.,0.76159416,0.9999092,1.])
    assert np.isclose(tanh(x),z).all()

def test_d_tanh_array_1():
    x = np.array([0,0])
    z = np.array([1,1])
    assert (d_tanh(x) == z).all()

def test_d_tanh_array_2():
    x = np.array([-10,-5,-1,0,1,5,10])
    z = np.array([-99,-24,0,1,0,-24,-99])
    assert np.isclose(d_tanh(x),z).all()

#-------------------------------------------------------------------------------
#linear function and its derivative
#-------------------------------------------------------------------------------
def test_linear_1():
    x = np.array([1,4,7,0,3.5,4.5,-0.8,-25,5])
    z = np.array([1,4,7,0,3.5,4.5,-0.8,-25,5])
    assert (linear(x) == z).all()

def test_d_linear_1():
    x = np.array([1,4,7,0,3.5,4.5,-0.8,-25,5])
    z = np.array([1,1,1,1,1,1,1,1,1])
    assert (d_linear(x) == z).all()