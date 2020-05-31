"""Tests for system.py
"""

import pytest
import mock
import numpy as np

import dynode.system as ds
from dynode.containers import (ParameterContainer, VariableContainer,
    ResultContainer)

from test_systems import VanDerPol, SingleDegreeMass, EmptyTestSystem, CompositeTestSystem

def test_ABC_contract():
    #pylint: disable=abstract-class-instantiated
    with pytest.raises(TypeError):
        ds.SystemInterface()

def test_system_attributes():
    s = EmptyTestSystem()
    assert(isinstance(s.inputs, ParameterContainer))
    assert(isinstance(s.outputs, ParameterContainer))
    assert(isinstance(s.states, VariableContainer))
    assert(isinstance(s.ders, VariableContainer))
    assert(isinstance(s.res, ResultContainer))

@pytest.mark.parametrize('sys', [EmptyTestSystem, VanDerPol, SingleDegreeMass, CompositeTestSystem])
def test_states_ders_equal_length(sys):
    ts = sys()
    states = ts.get_states()
    ders = np.zeros_like(states)
    ts.get_ders(0, ders)
    assert(states.shape == ders.shape)

@pytest.mark.parametrize('sys', [EmptyTestSystem, VanDerPol, SingleDegreeMass, CompositeTestSystem])
def test_dispatch_states(sys):
    s = sys()
    states = s.get_states()
    new_states = np.random.randn(*states.shape)
    s.dispatch_states(0, new_states)

    assert(np.array_equal(s.get_states(), new_states))

def test_pre_connection_callback_signature():

    s = EmptyTestSystem()
    cb = mock.Mock()
    t = 10.5

    s.add_pre_connection(cb)

    s._step(t)

    cb.assert_called_with(s, t)

def test_post_connection_callback_signature():

    s = EmptyTestSystem()
    cb = mock.Mock()
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
        
def test_storing_non_existing():
    
    s = VanDerPol()
    
    with pytest.raises(AttributeError):
        s.add_store('mupp.muppet')

def test_storing_existing(pinned):
    
    s = VanDerPol()
    
    s.add_store('inputs.mu')
    
    s.store(0)
    s.inputs.mu = 8
    s.store(1)
    
    assert(s.res['inputs.mu'] == pinned)
    
def test_storing_existing_with_other_name(pinned):
    
    s = VanDerPol()
    
    s.add_store('inputs.mu', 'mu')
    
    s.store(0)
    s.inputs.mu = 8
    s.store(1)
    
    assert(s.res['mu'] == pinned)

    