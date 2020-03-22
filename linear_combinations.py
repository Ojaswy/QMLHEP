import numbers
import typing

import cirq
import numpy as np
import sympy

from paulicirq.utils import resolve_scalar

T = typing.TypeVar("T")
TypeOfSelf = typing.TypeVar("TypeOfSelf", bound="LinearSymbolicDict")
Scalar = typing.Union[numbers.Number, sympy.Basic]


class LinearSymbolicDict(cirq.value.LinearDict[T]):
    def clean(self, *, atol: float = 1e-9):
        """
        Remove terms with coefficients of absolute value atol or less.

        Modified from `cirq.value.linear_dict.LinearDict.clean` to support symbolic
        coefficients.

        """
        negligible = []
        for v, c in self._terms.items():
            try:
                if abs(c) <= atol:
                    negligible.append(v)
            except TypeError:  # "cannot determine truth value of Relational"
                pass

        for v in negligible:
            del self._terms[v]
        return self

    @staticmethod
    def _is_term_parameterized(term: T) -> typing.Union[
        bool, type(NotImplemented)
    ]:
        return cirq.is_parameterized(term)

    @staticmethod
    def _resolve_parameters_of_term(
            term: T,
            param_resolver: cirq.ParamResolverOrSimilarType
    ) -> T:
        return cirq.resolve_parameters(term, param_resolver)

    def _is_parameterized_(self) -> typing.Union[bool, type(NotImplemented)]:
        for v, c in self._terms.items():
            _is = self._is_term_parameterized(v)
            if _is in (True, NotImplemented):
                return _is

            _is = cirq.is_parameterized(c)
            if _is in (True, NotImplemented):
                return _is

        return False

    def _resolve_parameters_(
            self: TypeOfSelf,
            param_resolver: cirq.ParamResolverOrSimilarType
    ) -> TypeOfSelf:
        _resolved = type(self)({})

        for v, c in self._terms.items():
            _c_resolved = resolve_scalar(c, param_resolver)

            _resolved += type(self)({
                self._resolve_parameters_of_term(v, param_resolver):
                    _c_resolved
            })

        return _resolved

    def __str__(self):
        return str(self._terms)


class LinearCombinationOfMoments(LinearSymbolicDict[cirq.Moment]):
    def __init__(self, terms):
        super(LinearCombinationOfMoments, self).__init__(terms)

    @staticmethod
    def _is_term_parameterized(term: cirq.Moment) -> typing.Union[
        bool, type(NotImplemented)
    ]:
        for op in term.operations:
            _is = cirq.is_parameterized(op)

            if _is in (True, NotImplemented):
                return _is

        return False

    @staticmethod
    def _resolve_parameters_of_term(term: cirq.Moment, param_resolver) \
            -> cirq.Moment:
        _moment = cirq.Moment(
            cirq.resolve_parameters(op, param_resolver)
            for op in term.operations
        )

        return _moment


class LinearCombinationOfOperations(
    LinearSymbolicDict[typing.Tuple[cirq.Operation]]
):
    def __init__(self, terms):
        super(LinearCombinationOfOperations, self).__init__(terms)

    @staticmethod
    def _is_term_parameterized(term: typing.Tuple[cirq.Operation]) \
            -> typing.Union[bool, type(NotImplemented)]:
        for op in term:
            _is = cirq.is_parameterized(op)

            if _is in (True, NotImplemented):
                return _is

        return False

    @staticmethod
    def _resolve_parameters_of_term(
            term: typing.Tuple[cirq.Operation], param_resolver
    ) -> typing.Tuple[cirq.Operation]:
        _op_tuple = tuple(
            cirq.resolve_parameters(op, param_resolver)
            for op in term
        )

        return _op_tuple

    @property
    def qubits(self) -> typing.Tuple[cirq.Qid, ...]:
        _qubits = set()
        for op_tuple in self.keys():
            for op in op_tuple:
                _qubits.update(op.qubits)

        return tuple(_qubits)


def simulate_linear_combination_of_operations(
        lco: LinearCombinationOfOperations,
        initial_state
) -> np.ndarray:
    state_vector = 0  # type: np.ndarray

    if cirq.is_parameterized(lco):
        raise ValueError("Operations containing parameters are not supported!")

    for op_tuple, coeff in lco.items():
        circuit = cirq.Circuit()
        circuit.append(op_tuple)

        simulator = cirq.Simulator()
        _state = (simulator.simulate(circuit, initial_state=initial_state)
                  .final_simulator_state.state_vector)  # type: np.ndarray

        state_vector += coeff * _state
        # print(id(lco), coeff, _state)

    return state_vector
