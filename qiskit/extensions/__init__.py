# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================================
Quantum Circuit Extensions (:mod:`qiskit.extensions`)
=====================================================

.. currentmodule:: qiskit.extensions

Unitary Extensions
==================

.. autosummary::
   :toctree: ../stubs/

   UnitaryGate
   HamiltonianGate
   SingleQubitUnitary

Simulator Extensions
====================

.. autosummary::
   :toctree: ../stubs/

   Snapshot

Initialization
==============

.. autosummary::
   :toctree: ../stubs/

   Initialize

Uniformly Controlled Rotations
==============================

.. autosummary::
   :toctree: ../stubs

   UCPauliRotGate
   UCRXGate
   UCRYGate
   UCRZGate
"""

# import all standard gates
from qiskit.circuit.library.standard_gates import *  # noqa: F403
from qiskit.circuit.barrier import Barrier

from .quantum_initializer import (
    Initialize,
    SingleQubitUnitary,
    UCPauliRotGate,
    UCRXGate,
    UCRYGate,
    UCRZGate,
)
from .unitary import UnitaryGate
from .hamiltonian_gate import HamiltonianGate
from .simulator import Snapshot
