import typing

import openfermion


def reduce_inactive_qubits(
        qubit_operator: openfermion.QubitOperator,
        inactive_qubits: typing.Optional[typing.Iterable[int]] = None
) -> typing.Tuple[openfermion.QubitOperator, int]:
    """
    Reduce the inactive qubits in a qubit operator.

    :param qubit_operator:
        The qubit operator to be simplified (i.e. reduced).
    :param inactive_qubits:
        The qubits to be reduced from the qubit operator.

        If given out explicitly, the qubits in `inactive_qubits` will all be
        removed from the qubit operator (no matter whether it is really
        "inactive").

        If left to be default, the inactive qubits to be removed will be
        automatically determined. Here, "inactive" means that there are not
        any effective Pauli operators ("X", "Y", "Z") acting on the qubits in
        `qubit_operator`.

    :return:
        The reduced qubit operator, and the number of reduced qubits.

    Example:

    >>> op = (openfermion.QubitOperator("X0 Y1", 0.1) +
    ...       openfermion.QubitOperator("Z0 Y1 X3", 0.2))
    >>> reduce_inactive_qubits(op)
    (0.1 [X0 Y1] +
    0.2 [Z0 Y1 X2], 1)
    >>> reduce_inactive_qubits(op, (0, 2))
    (0.1 [Y0] +
    0.2 [Y0 X1], 2)
    >>> reduce_inactive_qubits(op, (0,))
    (0.1 [Y0] +
    0.2 [Y0 X2], 1)

    """
    if inactive_qubits is None:
        active_qubits = set()
        for term in qubit_operator.terms.keys():
            for qubit_index, pauli in term:
                active_qubits.add(qubit_index)
        active_qubits = sorted(active_qubits)
    else:
        active_qubits = sorted(
            q for q in range(0, openfermion.count_qubits(qubit_operator))
            if q not in inactive_qubits
        )

    reduced_qubit_op = openfermion.QubitOperator()
    for term, coefficient in qubit_operator.terms.items():
        reduced_term = tuple(
            (active_qubits.index(qubit_index), pauli) for qubit_index, pauli in term
            if qubit_index in active_qubits
        )
        _op = openfermion.QubitOperator()
        _op.terms[reduced_term] = coefficient
        reduced_qubit_op += _op

    if inactive_qubits is None:  # if in the default case:
        assert len(active_qubits) == openfermion.count_qubits(reduced_qubit_op)
    else:
        # For cases like:
        #
        #     reduce_inactive_qubits(
        #         openfermion.QubitOperator("Z0 Y1 X3", 0.2),
        #         (3,)
        #     )
        #
        # active_qubits == (0, 1, 2), but the reduced operator is
        #
        #     openfermion.QubitOperator("Z0 Y1", 0.2)
        #
        # counting whose qubits will give out 2 instead of 3.
        pass

    num_reduced_qubits = \
        openfermion.count_qubits(qubit_operator) - len(active_qubits)

    return reduced_qubit_op, num_reduced_qubits


def inactivate_stationary_qubits(
        qubit_operator: openfermion.QubitOperator
) -> typing.Tuple[openfermion.QubitOperator, int]:
    """
    Set the stationary qubits in a qubit operator to be "inactive".

    Here, "stationary" means the qubit is ONLY acted by ONE kind of operators
    in {"X", "Y", "Z", "I"}, which differs from the meaning of "generally
    stationary" in `block_reduce`.

    "Inactive" means the qubit is not acted by any effective Pauli operators
    (i.e. only acted on by the identity operator). The definition here is same
    as in `reduce_inactive_qubits`.

    See also:
        `block_reduce`
        `reduce_inactive_qubits`

    :param qubit_operator:
        The qubit operator to be processed.

    :return:
        The processed qubit operator, and the number of inactivated stationary
        qubits.

    Example:

    >>> op = (openfermion.QubitOperator("X0 Y1", 0.1) +
    ...       openfermion.QubitOperator("Z0 Y1 X3", 0.2))
    >>> inactivate_stationary_qubits(op)
    (0.1 [X0] +
    0.2 [Z0 X3], 1)

    """
    common_ops = typing.cast(set, None)  # type: typing.Set[typing.Tuple[int, str]]
    for term in qubit_operator.terms.keys():
        if common_ops is not None:
            common_ops.intersection_update(set(term))
        else:
            common_ops = set(term)
    num_stationary_qubits = len(common_ops)

    reduced_qubit_op = openfermion.QubitOperator()
    for term, coefficient in qubit_operator.terms.items():
        reduced_term = tuple(
            pauli_on_qubit for pauli_on_qubit in term
            if pauli_on_qubit not in common_ops
        )
        _op = openfermion.QubitOperator()
        _op.terms[reduced_term] = coefficient
        reduced_qubit_op += _op

    return reduced_qubit_op, num_stationary_qubits


def reduce(
        qubit_operator: openfermion.QubitOperator
) -> typing.Tuple[
    openfermion.QubitOperator, int, int
]:
    """
    Reduce the qubits which are stationary (not generally) or inactive in the
    qubit operator.

    See also:
        `inactivate_stationary_qubits`
        `block_reduce`

    :param qubit_operator:
        The qubit operator to be reduced.

    :return:
        The reduced qubit operator, the number of stationary qubits, and
        the number of inactive qubits (not including stationary qubits).

    Example:

    >>> op = (openfermion.QubitOperator("X0 Y1", 0.1) +
    ...       openfermion.QubitOperator("Z0 Y1 X3", 0.2))
    >>> reduce(op)
    (0.1 [X0] +
    0.2 [Z0 X1], 1, 1)

    """
    op, n_stationary = inactivate_stationary_qubits(qubit_operator)
    op, n_total_reduced = reduce_inactive_qubits(op)
    n_inactive = n_total_reduced - n_stationary
    return op, n_stationary, n_inactive


def group(
        qubit_operator: openfermion.QubitOperator,
        by_qubit_indices: typing.Iterable[int]
):
    grouped = {}
    for term, coefficient in qubit_operator.terms.items():
        paulis_at_indices = {
            qubit_index: pauli for qubit_index, pauli in term
            if qubit_index in by_qubit_indices
        }

        for i in by_qubit_indices:
            if i not in paulis_at_indices:
                paulis_at_indices[i] = "I"

        paulis_at_indices = sorted(
            ((qubit_index, pauli)
             for qubit_index, pauli in paulis_at_indices.items()),
            key=lambda item: item[0]
        )
        paulis_at_indices = tuple(paulis_at_indices)

        op = openfermion.QubitOperator()
        op.terms[term] = coefficient

        if paulis_at_indices not in grouped:
            grouped[paulis_at_indices] = op
        else:
            grouped[paulis_at_indices] += op

    return grouped


def block_reduce(
        qubit_operator: openfermion.QubitOperator
):
    """
    Reduce the qubit operator into blocks.

    If a qubit in `qubit_operator` is only acted by the identity operator or
    one kind of Pauli operator (or both of them), it is considered to be
    "generally stationary", and can be used to reduce the operator into several
    blocks.

    See also:
        `reduce`
        `inactivate_stationary_qubits`

    :param qubit_operator:
        The qubit operator to be block-reduced.

    :return:
        The reduced blocks.

    Example:

    >>> op = (openfermion.QubitOperator("X0 Y1 Z3   ", 0.1) +
    ...       openfermion.QubitOperator("Y0 Z1 Z3 Y4", 0.2) +
    ...       openfermion.QubitOperator("X0 Z1      ", 0.3) +
    ...       openfermion.QubitOperator("Y0       X4", 0.4))

    We can find that qubit 3 is "generally stationary", and thus we can use it
    to do the block reduction:
    >>> block_reduce(op)
    {((2, 'I'), (3, 'Z')): 0.1 [X0 Y1] +
    0.2 [Y0 Z1 Y2], ((2, 'I'), (3, 'I')): 0.3 [X0 Z1] +
    0.4 [Y0 X2]}

    Here, `0.1 [X0 Y1] + 0.2 [Y0 Z1]` and `0.3 [X0 Z1] + 0.4 [Y0]` are the two
    blocks corresponding to "I2 Z3" and "I2 I3".

    """
    n_qubits = openfermion.count_qubits(qubit_operator)
    appeared_paulis = [set() for _ in range(n_qubits)]

    for term, _ in qubit_operator.terms.items():
        for qubit_index, pauli in term:
            appeared_paulis[qubit_index].add(pauli)

    reducible_qubits = []
    for i, paulis in enumerate(appeared_paulis):
        if len(paulis) <= 1:
            reducible_qubits.append(i)

    grouped = group(qubit_operator, by_qubit_indices=reducible_qubits)
    reduced = {}
    for paulis_at_reducible_qubits, op in grouped.items():
        reduced[paulis_at_reducible_qubits] = reduce_inactive_qubits(
            op, reducible_qubits
        )[0]

    return reduced
