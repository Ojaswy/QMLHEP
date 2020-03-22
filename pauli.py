import math
import numbers
import re
import typing

import numpy as np
from scipy.sparse import spmatrix, coo_matrix, kron

from .utils import lazy_load_instance_property


class Pauli(object):
    X = np.array([
        [0, 1],
        [1, 0]
    ])

    Y = np.array([
        [0, -1j],
        [1j, 0]
    ])

    Z = np.array([
        [1, 0],
        [0, -1]
    ])

    I = np.identity(2)

    PAULI_LABELS = list("XYZI")

    def __init__(
            self,
            op: typing.Union[
                str, numbers.Integral, np.ndarray
            ]
    ):
        self.op_label = None

        if isinstance(op, str):
            op = op.upper()
            if op in self.PAULI_LABELS:
                self.op_label = op
                return
            else:
                raise ValueError("invalid operator label {}".format(op))

        elif isinstance(op, numbers.Integral):
            try:
                self.op_label = self.PAULI_LABELS[typing.cast(int, op)]
            except IndexError:
                raise ValueError("invalid operator label {}".format(op))

        elif isinstance(op, np.ndarray):
            for op_label in self.PAULI_LABELS:
                if np.allclose(op, getattr(self, op_label)):
                    self.op_label = op_label
                    break

            if self.op_label is None:
                raise ValueError("invalid operator label {}".format(op))

        else:
            raise ValueError("invalid operator label {}".format(op))

    @property
    def index(self) -> int:
        return self.PAULI_LABELS.index(self.op_label)

    @property
    def string(self) -> str:
        return self.op_label

    @property
    def fullstr(self) -> str:
        """ Just for compatibility with PauliWord. """
        return self.op_label

    @lazy_load_instance_property
    def array(self) -> np.ndarray:
        return getattr(self, self.op_label)

    def get_the_other_two_paulis(self) -> typing.Tuple["Pauli", "Pauli"]:
        if self.op_label == "I":
            return Pauli("I"), Pauli("I")

        return Pauli((self.index + 1) % 3), Pauli((self.index + 2) % 3)

    def __str__(self):
        return self.string

    def __eq__(self, other: "Pauli"):
        if not isinstance(other, Pauli):
            return False

        return self.op_label == other.op_label


class PauliWord(object):
    def __init__(
            self,
            word: typing.Union[
                str,
                typing.Iterable[numbers.Integral],
                "PauliWord"
            ],
            length: typing.Union[numbers.Integral, int, None] = None
    ):
        self.word = None

        if isinstance(word, str):
            word = re.sub(r"\s+", "", word)

            # e.g. "XXZYXI"
            if set(word).issubset(set(Pauli.PAULI_LABELS)):
                self.word = word

            # e.g. "X0X1Z2Y3X4I5X10"
            # note: if there are duplicated Pauli operators on same qubit,
            # only one of them will be kept.
            else:
                pattern = (
                    r"((?P<op_label>[{0}])(?P<qubit_index>\d+))"
                    .format("".join(Pauli.PAULI_LABELS))
                )
                if not re.fullmatch(pattern + r"+", word):
                    raise ValueError("invalid Pauli word {}".format(word))

                ops = {
                    int(match.group("qubit_index")): match.group("op_label")
                    for match in re.finditer(pattern, word)
                }

                self.word = ""
                for i in range(max(ops.keys()) + 1):
                    self.word += ops.get(i, "I")

        elif isinstance(word, typing.Iterable):
            self.word = "".join([
                Pauli(index).string for index in word
            ])

        elif isinstance(word, PauliWord):
            self.word = word.word

        else:
            raise ValueError("invalid Pauli word {}".format(word))

        if length is not None:
            original_len = len(self.word)
            if original_len < length:
                self.word += "I" * (length - original_len)
            else:
                self.word = self.word[:length]

    @property
    def indices(self) -> typing.Iterable[int]:
        return [Pauli(op).index for op in self.word]

    @property
    def fullstr(self) -> str:
        return self.word

    @property
    def qubit_operator_str(self) -> str:
        return " ".join([
            "{0}{1}".format(op_label, qubit_index)
            for qubit_index, op_label in enumerate(self.word)
            if op_label != "I"
        ])

    @property
    def dict_form(self) -> typing.Dict[int, str]:
        return {
            int(qubit_index): op_label
            for qubit_index, op_label in enumerate(self.word)
            if op_label != "I"
        }

    @property
    def effective_len(self) -> int:
        return len(self.word.replace("I", ""))

    # Don't use `@lazy_load_instance_property`, which may cause error
    # for `multiprocessing.pool.map` in `qcc.study.QCC.pauli_word_ranking`.
    @property
    def sparray(self) -> spmatrix:
        return kron_all_sparse([
            coo_matrix(Pauli(op).array) for op in self.word
        ])

    @classmethod
    def join(cls, *pauli_words: typing.Union["PauliWord", Pauli]) -> "PauliWord":
        return PauliWord("".join([word.fullstr for word in pauli_words]))

    def __str__(self):
        return self.qubit_operator_str

    def __getitem__(self, item: typing.Union[int, slice]):
        return PauliWord(self.word[item])

    def __len__(self):
        return len(self.word)

    def __eq__(self, other: "PauliWord"):
        return self.word == other.word

    def __repr__(self):
        return "PauliWord(\"{}\")".format(self.fullstr)

    def strip(self):
        return PauliWord(
            self.word.strip("I")
        )


def kron_all_sparse(arrays: typing.Iterable[spmatrix]) -> spmatrix:
    result = None
    for array in arrays:
        result = kron(result, array) if result is not None else array
    return result


def kron_all(arrays: typing.Iterable[np.ndarray]) -> np.ndarray:
    result = None
    for array in arrays:
        result = np.kron(result, array) if result is not None else array
    return result


def commutator(a, b):
    return a @ b - b @ a


def is_close_to_zero(a, atol=0.0, rtol=1e-09):
    return math.isclose(abs(a), 0, rel_tol=rtol, abs_tol=atol)
