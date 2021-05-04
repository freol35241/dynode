.. dynode-two-connected_vanderpols

Two connected Van der Pol oscillators
==========================================

Imagine the situation where you have two oscillators interacting as
follows:

-  The damping paramater (``mu``) of oscillator 1 is forced by a sinus
   wave according to ``0.1*sin(0.1*t)``
-  The damping parmeter (``mu``) of oscillator 2 is forced to follow the
   the state ``y`` of oscillator 1

In dynode, the above scenario can be described and simulated as:

.. code:: python

    from dynode import connect_signals, Simulation

    sys1 = VanDerPol()
    sys1.states.x = 1

    sys2 = VanDerPol()

    # Connecting state y of sys1 to input mu of sys2
    sys1.add_post_connection(connect_signals(sys1.states, 'y', sys2.inputs, 'mu'))

    # Forcing input mu of sys1 to follow a sinus function
    def sinus_forcer(sys, t):
        sys.inputs.mu = 0.1*np.sin(0.1*t)

    sys1.add_pre_connection(sinus_forcer)

    sim = Simulation()
    sim.add_system(sys1)
    sim.add_system(sys2)

    sim.simulate(100, 0.1)