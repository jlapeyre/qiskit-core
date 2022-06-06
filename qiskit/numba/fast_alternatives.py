# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Provide two implementations of some functions, a fast one in numba, and a fallback in Python/numpy.
"""

import numpy

from .maybe_numba import HAVE_NUMBA, vectorize, njit

if HAVE_NUMBA:

    @vectorize
    def abs2(z):
        return (z * numpy.conj(z)).real

else:

    def abs2(z):
        return abs(z) ** 2
