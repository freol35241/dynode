.. dynode-single-vanderpol

Single Van der Pol oscillator
==========================================

A well-known dynamical system is the `Van der Pol
oscillator <https://en.wikipedia.org/wiki/Van_der_Pol_oscillator>`__,
which is described by a second-order differential equation:

.. figure:: https://wikimedia.org/api/rest_v1/media/math/render/svg/99e33aa1bcd07cd6ce8cf2cf5bd9d630c3b0d21e
   :alt: Van der Pol 2nd order differential equation

Rewriting it to a system of ordinary differential equations yields:

.. figure:: https://wikimedia.org/api/rest_v1/media/math/render/svg/2e9748620372632fc912d764f4589a32f0626658
   :alt: Van der Pol ODE1

   Van der Pol ODE1
.. figure:: https://wikimedia.org/api/rest_v1/media/math/render/svg/82fff2145f98d0281f9c22c97fe6c625386d2b8e
   :alt: Van der Pol ODE2

In dynode, a Van der Pol ``system`` may be modelled as:

.. code:: python

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

And may be simulated like this:

.. code:: python

    from dynode.simulation import Simulation, Recorder

    sys = VanDerPol()

    rec = Recorder()
    rec.store(sys, 'states.x', alias='x')
    rec.store(sys, 'states.y', alias='y')

    sim = Simulation()
    sim.add_system(sys)
    sim.add_observer(rec)

    sys.states.x = 1
    sim.simulate(100, 0.1)

    import matplotlib.pyplot as plt
    plt.plot(rec[sys]['time'], rec[sys]['x'])
    plt.plot(rec[sys]['time'], rec[sys]['y'])
    plt.show()