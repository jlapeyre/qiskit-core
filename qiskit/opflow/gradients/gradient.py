# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The base interface for Opflow's gradient."""

from typing import Union, List


from multipledispatch import dispatch
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit import ParameterExpression, ParameterVector
from ..expectations.pauli_expectation import PauliExpectation
from .gradient_base import GradientBase
from ..list_ops.composed_op import ComposedOp
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.tensored_op import TensoredOp
from ..operator_base import OperatorBase
# from ..operator_globals import Zero, One
from ..state_fns.circuit_state_fn import CircuitStateFn
from ..exceptions import OpflowError

try:
    from jax import grad, jit
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


from .derivative_utils import ZERO_EXPR
from .derivative_utils import is_coeff_c

class Gradient(GradientBase):
    """Convert an operator expression to the first-order gradient."""

    # Call _convert because of some multipledispatch bug, or s.t.
    # I'm unable to use MD with convert. It seems PauliExpectation.convert is confusing the MD?
    def convert(self, operator, params):
        r"""
        Args:
            operator: The operator we are taking the gradient of.
            params: params: The parameters we are taking the gradient with respect to.

        Returns:
            An operator whose evaluation yields the Gradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        return self._convert(operator, params)

    @dispatch(object, (ParameterVector, list))
    def _convert(self, operator, params):
        param_grads = [self._convert(operator, param) for param in params]
        absent_params = [params[i]
                         for i, grad_ops in enumerate(param_grads) if grad_ops is None]
        if len(absent_params) > 0:
            raise ValueError(
                "The following parameters do not appear in the provided operator: ",
                absent_params
            )
        return ListOp(param_grads)

    @dispatch(object, object)
    def _convert(self, operator, param):
        # Preprocessing
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
        return self.get_gradient(cleaned_op, param)

    @dispatch((object, ComposedOp, SummedOp, TensoredOp, CircuitStateFn), (ParameterVector, list))
    def get_gradient(self, operator, params):
        param_grads = [self.get_gradient(operator, param) for param in params]
        # If get_gradient returns None, then the corresponding parameter was probably not
        # present in the operator. This needs to be looked at more carefully as other things can
        # probably trigger a return of None.
        absent_params = [params[i]
                         for i, grad_ops in enumerate(param_grads) if grad_ops is None]
        if len(absent_params) > 0:
            raise ValueError(
                'The following parameters do not appear in the provided operator: ',
                absent_params
            )
        return ListOp(param_grads)


    def handle_coeff_not_one(self, operator, param):
        # Separate the operator from the coefficient
        coeff = operator._coeff
        op = operator / coeff
        # Get derivative of the operator (recursively)
        d_op = self.get_gradient(op, param)
        # ..get derivative of the coeff
        d_coeff = self.parameter_expression_grad(coeff, param)

        grad_op = 0
        if d_op != ZERO_EXPR and not is_coeff_c(coeff, 0.0):
            grad_op += coeff * d_op
        if op != ZERO_EXPR and not is_coeff_c(d_coeff, 0.0):
            grad_op += d_coeff * op
        if grad_op == 0:
            grad_op = ZERO_EXPR
        return grad_op

    # Base Case, you've hit a ComposedOp!
    # Prior to execution, the composite operator was standardized and coefficients were
    # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
    # circuits were applied. Additionally, all coefficients within ComposedOps were collected
    # and moved out front.
    @dispatch(ComposedOp, object)
    def get_gradient(self, operator, param):
        if not is_coeff_c(operator._coeff, 1.0):
            return self.handle_coeff_not_one(operator, param)
        # Do some checks to make sure operator is sensible
        # TODO add compatibility with sum of circuit state fns
        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError(
                'The gradient framework is compatible with states that are given as '
                'CircuitStateFn')

        return self.grad_method.convert(operator, param)

    @dispatch(CircuitStateFn, object)
    def get_gradient(self, operator, param):
        # Gradient of an a state's sampling probabilities
        if not is_coeff_c(operator._coeff, 1.0):
            return self.handle_coeff_not_one(operator, param)
        return self.grad_method.convert(operator, param)

    @dispatch(SummedOp, object)
    def get_gradient(self, operator, param):
        if not is_coeff_c(operator._coeff, 1.0):
            return self.handle_coeff_not_one(operator, param)
        grad_ops = [self.get_gradient(op, param) for op in operator.oplist]
        return SummedOp(oplist=[grad for grad in grad_ops if grad != ZERO_EXPR]).reduce()

    @dispatch(TensoredOp, object)
    def get_gradient(self, operator, param):
        if not is_coeff_c(operator._coeff, 1.0):
            return self.handle_coeff_not_one(operator, param)
        grad_ops = [self.get_gradient(op, param) for op in operator.oplist]
        return TensoredOp(oplist=grad_ops)

    @dispatch(ListOp, object)
    def get_gradient(self, operator, param) -> OperatorBase:
        """Get the gradient for the given operator w.r.t. the given parameters

        Args:
            operator: Operator w.r.t. which we take the gradient.
            params: Parameters w.r.t. which we compute the gradient.

        Returns:
            Operator which represents the gradient w.r.t. the given params.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            OpflowError: If the coefficient of the operator could not be reduced to 1.
            OpflowError: If the differentiation of a combo_fn requires JAX but the package is not
                       installed.
            TypeError: If the operator does not include a StateFn given by a quantum circuit
            Exception: Unintended code is reached
            MissingOptionalLibraryError: jax not installed
        """

        # Handle Product Rules
        if not is_coeff_c(operator._coeff, 1.0):
            return self.handle_coeff_not_one(operator, param)

        # Handle the chain rule
        grad_ops = [self.get_gradient(op, param) for op in operator.oplist]

        # Note: this check to see if the ListOp has a default combo_fn
        # will fail if the user manually specifies the default combo_fn.
        # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
        # later on jax will try to differentiate it and raise an error.
        # An alternative is to check the byte code of the operator's combo_fn against the
        # default one.
        if operator._combo_fn == ListOp([])._combo_fn:
            return ListOp(oplist=grad_ops)

        if operator.grad_combo_fn:
            grad_combo_fn = operator.grad_combo_fn
        else:
            if _HAS_JAX:
                grad_combo_fn = jit(grad(operator._combo_fn, holomorphic=True))
            else:
                raise MissingOptionalLibraryError(
                    libname='jax',
                    name='get_gradient',
                    msg='This automatic differentiation function is based on JAX. '
                    'Please install jax and use `import jax.numpy as jnp` instead '
                    'of `import numpy as np` when defining a combo_fn.')

        def chain_rule_combo_fn(x):
            result = np.dot(x[1], x[0])
            if isinstance(result, np.ndarray):
                result = list(result)
            return result

        return ListOp([ListOp(operator.oplist, combo_fn=grad_combo_fn), ListOp(grad_ops)],
                      combo_fn=chain_rule_combo_fn)
