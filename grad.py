import copy
import typing

import cirq
import numpy as np
import sympy
from cirq import flatten_op_tree

from paulicirq.gates import GlobalPhaseGate
from paulicirq.gates.gate_block import GateBlock
from paulicirq.gates.universal_gate_set import (
    is_a_basic_operation, is_an_indecomposable_operation
)
from paulicirq.linear_combinations import (
    LinearCombinationOfOperations,
    LinearSymbolicDict
)
from paulicirq.op_tree import OpTreeGenerator
from paulicirq.utils import ToBeTested

ZERO_OP = LinearCombinationOfOperations({})  # constant


class GradNotImplemented:
    __slots__ = ("operation",)

    def __init__(self, operation):
        self.operation = operation


def is_zero_op_or_grad_not_implemented(grad):
    return grad == ZERO_OP or isinstance(grad, GradNotImplemented)


@ToBeTested
def op_grad(
        operation: cirq.GateOperation, parameter: sympy.Symbol
) -> typing.Union[
    LinearCombinationOfOperations,
    GradNotImplemented
]:
    if not cirq.is_parameterized(operation):
        return ZERO_OP.copy()

    gate = operation.gate
    qubits = operation.qubits

    if isinstance(gate, cirq.XPowGate):
        # dXPow(e=f(t), s) / dt = i π (s + 1 / 2 - X / 2) * (df(t)/dt) XPow(e=f(t), s)
        # Note that: Rx(θ) = XPowGate(exponent=θ/pi, global_shift=-0.5)
        partial = sympy.diff(gate.exponent, parameter)
        coeff_i = 1.0j * sympy.pi * (gate._global_shift + 0.5)
        coeff_x = -1.0j / 2 * sympy.pi
        if partial == 0:
            return ZERO_OP.copy()
        return LinearCombinationOfOperations({
            (cirq.X.on(*qubits), operation): coeff_x * partial,
            (cirq.I.on(*qubits), operation): coeff_i * partial
        })

    elif isinstance(gate, cirq.YPowGate):
        # dYPow(e=f(t), s) / dt = i π (s + 1 / 2 - Y / 2) * (df(t)/dt) YPow(e=f(t), s)
        # Note that: Ry(θ) = YPowGate(exponent=θ/pi, global_shift=-0.5)
        partial = sympy.diff(gate.exponent, parameter)
        coeff_i = 1.0j * sympy.pi * (gate._global_shift + 0.5)
        coeff_y = -1.0j / 2 * sympy.pi
        if partial == 0:
            return ZERO_OP.copy()
        return LinearCombinationOfOperations({
            (cirq.Y.on(*qubits), operation): coeff_y * partial,
            (cirq.I.on(*qubits), operation): coeff_i * partial
        })

    elif isinstance(gate, cirq.ZPowGate):
        # dZPow(e=f(t), s) / dt = i π (s + 1 / 2 - Z / 2) * (df(t)/dt) ZPow(e=f(t), s)
        # Note that: Ry(θ) = ZPowGate(exponent=θ/pi, global_shift=-0.5)
        partial = sympy.diff(gate.exponent, parameter)
        coeff_i = 1.0j * sympy.pi * (gate._global_shift + 0.5)
        coeff_z = -1.0j / 2 * sympy.pi
        if partial == 0:
            return ZERO_OP.copy()
        return LinearCombinationOfOperations({
            (cirq.Z.on(*qubits), operation): coeff_z * partial,
            (cirq.I.on(*qubits), operation): coeff_i * partial
        })

    elif isinstance(gate, GlobalPhaseGate):
        gate = typing.cast(GlobalPhaseGate, gate)
        # Ph(θ) = exp(i pi θ)
        # dPh(f(θ)) / dθ = [i pi df(θ) / dθ] exp(i pi f(θ))
        #                = [i pi df(θ) / dθ] Ph(f(θ))
        coeff = 1.0j * sympy.diff(gate.rad * sympy.pi, parameter)
        if coeff == 0:
            return ZERO_OP.copy()
        return LinearCombinationOfOperations({
            (operation,): coeff
        })

    elif isinstance(gate, cirq.EigenGate):
        gate = typing.cast(cirq.EigenGate, gate)
        eigenvalues = {v for v, p in gate._eigen_components()}
        if eigenvalues == {0, 1}:
            e = gate.exponent
            s = gate._global_shift
            partial = sympy.diff(e, parameter)

            if partial == 0:
                return ZERO_OP.copy()

            num_qubits = gate.num_qubits()
            gate_e1_s0 = copy.deepcopy(gate)
            gate_e1_s0._exponent = 1.0
            gate_e1_s0._global_shift = 0.0  # Any better solutions?
            coeff = 0.5 * sympy.exp(1.0j * sympy.pi * (1 + s) * e)

            return 1.0j * sympy.pi * LinearCombinationOfOperations({
                (operation,): s * partial,
                (gate_e1_s0.on(*qubits),): -coeff * partial,
                (cirq.IdentityGate(num_qubits).on(*qubits),): coeff * partial
            })

        else:
            return GradNotImplemented(operation)

    elif isinstance(gate, GateBlock):
        gate = typing.cast(GateBlock, gate)

        generator_grad = op_tree_generator_grad(
            gate._op_generator,
            parameter
        )

        if isinstance(generator_grad, GradNotImplemented):
            return generator_grad

        if len(generator_grad) == 0:
            return ZERO_OP.copy()

        _grad = LinearCombinationOfOperations({})
        for generator, coeff in generator_grad.items():
            grad_gate = GateBlock(generator)

            _grad += LinearCombinationOfOperations({
                (grad_gate.on(*qubits),): coeff
            })

            # print(grad_gate.diagram())

        return _grad

    elif isinstance(gate, cirq.ControlledGate):
        gate = typing.cast(cirq.ControlledGate, gate)

        sub_gate_qubits = qubits[gate.num_controls():]
        sub_op_grad = op_grad(
            gate.sub_gate.on(*sub_gate_qubits), parameter
        )

        if is_zero_op_or_grad_not_implemented(sub_op_grad):
            return sub_op_grad

        _gate_grad = LinearCombinationOfOperations({})
        op: cirq.GateOperation
        for op_series, coeff in sub_op_grad.items():
            _controlled_op_series = [
                cirq.ControlledGate(
                    op.gate, control_qubits=qubits[:gate.num_controls()]
                ).on(*sub_gate_qubits)
                for op in op_series
            ]
            _controlled_negative_op_series = copy.deepcopy(_controlled_op_series)
            _controlled_negative_op_series.insert(
                0,
                cirq.ControlledGate(
                    GlobalPhaseGate(rad=1),
                    control_qubits=qubits[:gate.num_controls()]
                ).on(*sub_gate_qubits)
            )
            _gate_grad += LinearCombinationOfOperations({
                tuple(_controlled_op_series): coeff,
                tuple(_controlled_negative_op_series): -coeff,
            }) / 2.0

        return _gate_grad

    # if `operation` is a basic and indecomposable operation whose grad is
    # not implemented
    elif is_an_indecomposable_operation(operation):
        return GradNotImplemented(operation)

    else:
        op_series = cirq.decompose(
            operation,
            keep=(lambda op:
                  is_a_basic_operation(op)
                  or is_an_indecomposable_operation(op)),
            on_stuck_raise=None
        )  # type: typing.List[cirq.Operation]

        _grad = op_series_grad(op_series, parameter)
        return _grad


def op_series_grad(
        op_series: typing.Sequence[cirq.Operation],
        parameter: sympy.Symbol
) -> typing.Union[LinearCombinationOfOperations, GradNotImplemented]:
    grad_dict = LinearCombinationOfOperations({})
    for i, _op in enumerate(op_series):
        _grad = op_grad(typing.cast(cirq.GateOperation, _op), parameter)
        if isinstance(_grad, GradNotImplemented):
            return _grad
        elif _grad == ZERO_OP:
            continue
        else:
            for _grad_op, coeff in _grad.items():
                _grad_op_series = copy.deepcopy(op_series)
                _grad_op_series = (
                        _grad_op_series[:i]
                        + list(_grad_op)
                        + _grad_op_series[i + 1:]
                )
                grad_dict += LinearCombinationOfOperations({
                    tuple(_grad_op_series): coeff
                })

    return grad_dict


def op_tree_generator_grad(
        op_tree_generator: OpTreeGenerator,
        parameter: sympy.Symbol,
        **generator_kwargs
) -> typing.Union[LinearSymbolicDict[OpTreeGenerator], GradNotImplemented]:
    num_qubits = op_tree_generator.num_qubits

    def _op_generator_grad(_qubits):
        op_list = flatten_op_tree(
            op_tree_generator(_qubits, **generator_kwargs),
            preserve_moments=False
        )  # type: typing.Iterable[cirq.Operation]
        return op_series_grad(list(op_list), parameter)

    qubits = cirq.LineQubit.range(num_qubits)
    _op_grad = _op_generator_grad(qubits)  # [*]

    if _op_grad == ZERO_OP:
        return LinearSymbolicDict({})
    if isinstance(_op_grad, GradNotImplemented):
        return _op_grad

    # _op_grad is a non-empty LinearCombinationOfOperations
    _grad = LinearSymbolicDict({})
    for i in range(len(_op_grad)):
        def _deriv_op_generator_wrapper(_i):  # enclosure wrapper

            class _DerivOpTreeGenerator(type(op_tree_generator)):
                def __call__(self, _qubits) -> typing.Tuple[cirq.Operation]:
                    op = list(_op_generator_grad(_qubits))[_i]
                    return op

            return _DerivOpTreeGenerator(
                **op_tree_generator._kwargs
            )

        _grad += LinearSymbolicDict({
            _deriv_op_generator_wrapper(i):
                _op_grad[list(_op_grad)[i]]  # consistent with [*]
        })

    return _grad


def generate_random_circuit(
        num_qubits=5,
        circuit_len=10,
):
    qubits = cirq.LineQubit.range(num_qubits)

    circuit = cirq.Circuit()

    param_index = 0

    for _ in range(circuit_len):
        indices = np.arange(num_qubits)
        np.random.shuffle(indices)
        ops = []

        i = 0
        while i < num_qubits:
            param = sympy.Symbol("θ{}".format(param_index))
            gate_set = [cirq.Rx(param), cirq.Ry(param), cirq.Rz(param), cirq.CNOT]
            gate = np.random.choice(
                [g for g in gate_set
                 if i - 1 + g.num_qubits() <= num_qubits - 1],
                size=1
            )[0]  # type: cirq.Gate

            if gate.num_qubits() == 2:
                op = gate.on(
                    qubits[indices[i]], qubits[indices[i + 1]]
                )
                i += 1
            else:
                op = gate.on(
                    qubits[indices[i]]
                )
                param_index += 1

            ops.append(op)
            i += 1

        circuit.append(ops)

    return circuit
