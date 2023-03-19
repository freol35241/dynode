"""Tests for system.py
"""
from unittest import mock

import pytest
import numpy as np

import dynode.system as ds
from dynode.containers import ParameterContainer, VariableContainer

from test_systems import (
    VanDerPol,
    SingleDegreeMass,
    EmptyTestSystem,
    CompositeTestSystem,
)


def test_ABC_contract():
    # pylint: disable=abstract-class-instantiated
    with pytest.raises(TypeError):
        ds.SystemInterface()


def test_system_attributes():
    s = EmptyTestSystem()
    assert isinstance(s.inputs, ParameterContainer)
    assert isinstance(s.outputs, ParameterContainer)
    assert isinstance(s.states, VariableContainer)
    assert isinstance(s.ders, VariableContainer)


def test_try_add_self_as_subsystem():
    s = EmptyTestSystem()
    with pytest.raises(ValueError):
        s.add_subsystem(s)


@pytest.mark.parametrize(
    "sys", [EmptyTestSystem, VanDerPol, SingleDegreeMass, CompositeTestSystem]
)
def test_states_ders_equal_length(sys):
    ts = sys()
    states = ts.get_states()
    ders = np.zeros_like(states)
    ts.get_ders(0, ders)
    assert states.shape == ders.shape


@pytest.mark.parametrize(
    "sys", [EmptyTestSystem, VanDerPol, SingleDegreeMass, CompositeTestSystem]
)
def test_dispatch_states(sys):
    s = sys()
    states = s.get_states()
    new_states = np.random.randn(*states.shape)
    s.dispatch_states(0, new_states)

    assert np.array_equal(s.get_states(), new_states)


def test_pre_connection_callback_signature():
    s = EmptyTestSystem()
    cb = mock.Mock(dependees=[])
    t = 10.5

    s.add_pre_connection(cb)

    s._step(t)

    cb.assert_called_with(s, t)


def test_post_connection_callback_signature():
    s = EmptyTestSystem()
    cb = mock.Mock(dependees=[])
    t = 10.5

    s.add_post_connection(cb)
    s._step(t)

    cb.assert_called_with(s, t)


def test_pre_connection_callback_remover():
    s = EmptyTestSystem()
    cb = mock.Mock()
    t = 10.5

    remover = s.add_pre_connection(cb)
    remover()

    s._step(t)

    cb.assert_not_called()


def test_post_connection_callback_remover():
    s = EmptyTestSystem()
    cb = mock.Mock()
    t = 10.5

    remover = s.add_post_connection(cb)
    remover()

    s._step(t)

    cb.assert_not_called()


def test_pre_connection_add_twice():
    s = EmptyTestSystem()
    cb = mock.Mock()

    s.add_pre_connection(cb)

    with pytest.raises(ValueError):
        s.add_pre_connection(cb)


def test_post_connection_add_twice():
    s = EmptyTestSystem()
    cb = mock.Mock()

    s.add_post_connection(cb)

    with pytest.raises(ValueError):
        s.add_post_connection(cb)


def test_pre_connection_ordering():
    s = EmptyTestSystem()

    expected = [3, 1, 2]
    result = []

    def first(s, t):
        result.append(3)

    first.dependees = []

    def second(s, t):
        result.append(1)

    second.dependees = [first]

    def third(s, t):
        result.append(2)

    third.dependees = [second, first]

    s.add_pre_connection(second)
    s.add_pre_connection(third)
    s.add_pre_connection(first)

    s._step(10.5)

    assert expected == result
    assert s.connections.index(first) == 0
    assert s.connections.index(second) == 1
    assert s.connections.index(third) == 2


def test_post_connection_ordering():
    s = EmptyTestSystem()

    expected = [3, 1, 2]
    result = []

    def first(s, t):
        result.append(3)

    first.dependees = []

    def second(s, t):
        result.append(1)

    second.dependees = [first]

    def third(s, t):
        result.append(2)

    third.dependees = [second, first]

    s.add_post_connection(second)
    s.add_post_connection(third)
    s.add_post_connection(first)

    s._step(10.5)

    assert expected == result
    assert s.connections.index(first) == 0
    assert s.connections.index(second) == 1
    assert s.connections.index(third) == 2
