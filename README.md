# dynode

[![PyPI version](https://badge.fury.io/py/dynode.svg)](https://badge.fury.io/py/dynode)
![](https://github.com/freol35241/dynode/workflows/dynode/badge.svg)
[![codecov](https://codecov.io/gh/freol35241/dynode/branch/master/graph/badge.svg)](https://codecov.io/gh/freol35241/dynode)
![docs](https://github.com/freol35241/dynode/workflows/docs/badge.svg)

A framework for modelling and simulation of dynamical systems in the form of ordinary differential equations.

[**--> Docs <--**](https://freol35241.github.io/dynode/)

**Requires python >= 3.6**

## General

Dynode solves equations of the form ```y' = f(t, y)``` using SciPy's [ode solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html) but allows ```f``` to be modelled in a modular, object-oriented fashion using the notions of separate ```Systems``` that expose ```states``` and their corresponding derivatives, ```ders```. ```f``` may then be composed of an arbitraily complex collection of, connected or unconnected, ```Systems```.


#### Example: Single Van der Pol oscillator
A well-known dynamical system is the [Van der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator), which is described by a second-order differential equation:

![Van der Pol 2nd order differential equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/99e33aa1bcd07cd6ce8cf2cf5bd9d630c3b0d21e)

Rewriting it to a system of ordinary differential equations yields:

![Van der Pol ODE1](https://wikimedia.org/api/rest_v1/media/math/render/svg/2e9748620372632fc912d764f4589a32f0626658)

![Van der Pol ODE2](https://wikimedia.org/api/rest_v1/media/math/render/svg/82fff2145f98d0281f9c22c97fe6c625386d2b8e)

In dynode, a Van der Pol ```system``` may be modelled as:

```
from dynode import SystemInterface

class VanDerPol(SystemInterface):

    def __init__(self):
        super().__init__()

        self.inputs.mu = 1.0

        self.states.x = 0.0
        self.ders.dx = 0.0

        self.states.y = 0.0
        self.ders.dy = 0.0

    def do_step(self, time):
        mu = self.inputs.mu
        x = self.states.x
        y = self.states.y

        self.ders.dx = y
        self.ders.dy = mu*(1-x**2)*y - x
```

And may be simulated like this:

```
from dynode.simulation import Simulation

sys = VanDerPol()
sys.add_store('states.x', alias='x')
sys.add_store('states.y', alias='y')

sim = Simulation()
sim.add_system(sys)

sys.states.x = 1
sim.simulate(100, 0.1)

import matplotlib.pyplot as plt
plt.plot(sys.res.time, sys.res['x'])
plt.plot(sys.res.time, sys.res['y'])
plt.show()
```

## Connected systems
Dynode systems accepts ```connections``` as callbacks registered to a system. The callback signature looks like:

```
def connection_callback(system, time):
    pass
```
where ```system``` is a reference to the system this callback is registered to and ```time``` is the current time in the simulation.

Connections can be either ```pre_step_connections``` or ```post_step_connections``` depending on if the callback should be called prior to or after the ```do_step```-method of the system


#### Example: Two connected Van der Pol oscillators
Imagine the situation where you have two oscillators interacting as follows:

* The damping paramater (```mu```) of oscillator 1 is forced by a sinus wave according to ```0.1*sin(0.1*t)```
* The damping parmeter (```mu```) of oscillator 2 is forced to follow the the state ```y``` of oscillator 1

In dynode, the above scenario can be described and simulated as:
```
from dynode import connect_signals

sys1 = VanDerPol()
sys1.states.x = 1

sys2 = VanDerPol()

# Connecting state y of sys1 to input mu of sys2
sys1.add_post_connection(connect_signals(sys1.states, 'y', sys2.inputs, 'mu'))

# Forcing input mu of sys1 to follow a sinus function
def sinus_forcer(sys, t):
    sys.inputs.mu = 0.1*np.sin(0.1*t)

sys1.add_pre_connection(sinus_forcer)

sim = Sim()
sim.add_system(sys1)
sim.add_system(sys2)

sim.simulate(100, 0.1)
```

## License

Distributed under the terms of the MIT license, `dynode` is free and open source software


