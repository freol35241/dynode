.. dynode documentation master file, created by
   sphinx-quickstart on Mon May  3 18:13:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dynode's documentation!
==================================

Dynode solves equations of the form ``y' = f(t, y)`` using SciPy's `ode
solver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`__
but allows ``f`` to be modelled in a modular, object-oriented fashion
using the notions of separate ``Systems`` that expose ``states`` and
their corresponding derivatives, ``ders``. ``f`` may then be composed of
an arbitraily complex collection of, connected or unconnected,
``Systems``. 

.. toctree::
   :maxdepth: 2
   :caption: General

   content/overview
   content/installation

.. toctree::
   :maxdepth: 2
   :caption: Examples

   content/examples/single_vanderpol
   content/examples/two_connected_vanderpols

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   content/api_reference



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
