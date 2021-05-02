import pytest
import mock
import numpy as np

from dynode.containers import ParameterContainer, VariableContainer, ResultContainer


@pytest.mark.parametrize("cont", [ParameterContainer, VariableContainer])
def test_attribute_access(cont):
    c = cont()
    value = np.array([1, 2, 3])
    c["test"] = value
    np.array_equal(value, c.test)


@pytest.mark.parametrize("cont", [ParameterContainer, VariableContainer])
def test_attribute_creation(cont):
    c = cont()
    value = np.array([1, 2, 3])
    c.test = value
    np.array_equal(value, c["test"])


@pytest.mark.parametrize("cont", [ParameterContainer, VariableContainer])
def test_attribute_access_error(cont):
    c = cont()
    with pytest.raises(KeyError):
        c.test


def test_type_enforcement_scalar():
    c = VariableContainer()

    test_scalar = 10
    c["test"] = test_scalar
    assert isinstance(c.test, float)
    assert c.test == test_scalar


def test_type_enforcement_list():
    c = VariableContainer()

    test_list = [1, 2, 3]
    c["test"] = test_list
    assert isinstance(c.test, np.ndarray)
    assert np.array_equal(c.test, test_list)


def test_type_enforcement_tuple():
    c = VariableContainer()

    test_tuple = (1, 2, 3)
    c["test"] = test_tuple
    assert isinstance(c.test, np.ndarray)
    assert np.array_equal(c.test, test_tuple)


def test_type_enforcement_array():
    c = VariableContainer()

    test_array = np.array([1, 2, 3])
    c["test"] = test_array
    assert np.array_equal(c.test, test_array)


def test_type_enforcement_dict():
    c = VariableContainer()

    with pytest.raises(TypeError):
        test_dict = {"test": 10}
        c["test"] = test_dict


def test_result_storing():
    c = ResultContainer()
    N = 10

    for i in range(N):
        c.store("test", [i, i, i])

    assert len(c["test"]) == N
    assert [N - 1, N - 1, N - 1] == c["test"][-1]
