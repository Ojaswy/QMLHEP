import copy
from abc import abstractmethod, ABCMeta

import cirq
import typing

import sympy


class OpTreeGenerator(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @property
    @abstractmethod
    def num_qubits(self):
        pass

    @abstractmethod
    def __call__(
            self,
            qubits: typing.Iterable[cirq.Qid],
            **kwargs
    ) -> cirq.OP_TREE:
        pass

    def check_num_of_given_qubits(self, qubits):
        if self.num_qubits != len(qubits):
            raise ValueError(
                "The number of qubits ({}) != num_qubits of generator ({})"
                .format(len(qubits), self.num_qubits)
            )

    @abstractmethod
    def params(self) -> typing.Iterable[sympy.Symbol]:
        pass

    def _resolve_parameters_(self, param_resolver: cirq.ParamResolver):
        class _ParamResolvedGenerator(type(self)):
            def __call__(_self, qubits, **kwargs) -> cirq.OP_TREE:
                _op_tree = self.__call__(qubits, **kwargs)
                _resolved_op_tree = cirq.transform_op_tree(
                    _op_tree,
                    op_transformation=(
                        lambda op: cirq.resolve_parameters(op, param_resolver)
                    )
                )

                return _resolved_op_tree

        _resolved_generator = copy.deepcopy(self)
        _resolved_generator.__class__ = _ParamResolvedGenerator

        return _resolved_generator


class VariableNQubitsGenerator(OpTreeGenerator, metaclass=ABCMeta):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits=num_qubits)
        self._num_qubits = num_qubits

    @property
    def num_qubits(self):
        return self._num_qubits
