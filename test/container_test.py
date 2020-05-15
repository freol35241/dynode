import pytest
import mock
import numpy as np

from dynode.containers import (ParameterContainer, TypedParameterContainer,
    ResultContainer)

@pytest.mark.parametrize('cont', [ParameterContainer, 
    TypedParameterContainer, ResultContainer])
def test_attribute_access(cont):
    c = cont()
    value = np.array([1,2,3])
    c['test'] = value
    np.array_equal(value, c.test)

@pytest.mark.parametrize('cont', [ParameterContainer, 
    TypedParameterContainer])
def test_attribute_creation(cont):
    c = cont()
    value = np.array([1,2,3])
    c.test = value
    np.array_equal(value, c['test'])

@pytest.mark.parametrize('cont', [ParameterContainer, 
    TypedParameterContainer, ResultContainer])
def test_attribute_access_error(cont):
    c = cont()
    with pytest.raises(KeyError):
        c.test

def test_type_enforcement_scalar():
    c = TypedParameterContainer()

    test_scalar = 10
    c['test'] = test_scalar
    assert(isinstance(c.test, np.ndarray))
    assert(c.test[0] == test_scalar)

def test_type_enforcement_list():
    c = TypedParameterContainer()

    test_list = [1,2,3]
    c['test'] = test_list
    assert(isinstance(c.test, np.ndarray))
    assert(np.array_equal(c.test, test_list))

def test_type_enforcement_tuple():
    c = TypedParameterContainer()

    test_tuple = (1,2,3)
    c['test'] = test_tuple
    assert(isinstance(c.test, np.ndarray))
    assert(np.array_equal(c.test, test_tuple))

def test_type_enforcement_array():
    c = TypedParameterContainer()

    test_array = np.array([1,2,3])
    c['test'] = test_array
    assert(np.array_equal(c.test, test_array))

def test_type_enforcement_dict():
    c = TypedParameterContainer()

    with pytest.raises(TypeError):
        test_dict = {'test': 10}
        c['test'] = test_dict

def test_result_storing():
    c = ResultContainer()
    N = 10

    for i in range(N):
        c.store('test', [i,i,i])

    assert(len(c.test) == N)
    assert([N-1,N-1,N-1] == c.test[-1])