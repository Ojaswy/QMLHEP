import typing

import numpy as np
import sympy
 
from paulicirq.pauli import PauliWord, Pauli


def pauli_word_exp_factorization(
        coefficient: typing.Union[float, sympy.Basic],
        pauli_word: PauliWord
) -> typing.List[
    typing.Tuple[typing.Union[float, sympy.Basic], PauliWord]
]:
    """
    Factorize the exponentiation of Pauli word P
    $$
        e^{−i t P}
    $$
    (where t is the coefficient) with formula:
    $$
        e^{−i t P} = e^{i (π/4) wk'' P2} e^{−i t P1 wk'} e^{−i (π/4) wk'' P2},
    $$
    where wk, P1 and P2 satisfy $P = P1 wk P2$. The subscript of $wk$ denotes
    that this elementary Pauli operator corresponds to the kth qubit. The word
    $P1 (P2)$ contains all qubit indices that are strictly greater (lower)
    than k. wk' and wk'' are the other two Pauli operators which satisfy
    $$
        [ w', w'' ] = 2 i w.
    $$

    :param coefficient: float
        coefficient t in the exponential.
    :param pauli_word: PauliWord
        the Pauli word P in the exponential.
    :return:
        the factorized form of $e^{−i t P}$.
        For example, if
        $$
            e^{−i t P} = e^{−i tn Pn} ... e^{−i t1 P1} e^{−i t0 P0},
        $$
        then the returned will be like:
            [
                (tn, Pn),
                ...
                (t0, P0),
            ].

        Note that 0 <= Pi.effective_len <= 2.

    """

    if pauli_word.effective_len == 0:
        return []
    elif pauli_word.effective_len < 3:
        return [(coefficient, pauli_word)]

    # choose k
    pauli_dict_form = pauli_word.dict_form
    k = sorted(pauli_dict_form.keys())[len(pauli_dict_form) // 2]

    p1, wk, p2 = [pauli_word[:k], pauli_word[k], pauli_word[k + 1:]]
    i1, i2 = map(lambda word: PauliWord("I" * len(word)), [p1, p2])
    wk1, wk2 = Pauli(wk.word).get_the_other_two_paulis()

    _factorized = pauli_word_exp_factorization(
        coefficient=sympy.Symbol("t"),
        pauli_word=PauliWord.join(i1, wk2, p2)
    )

    returned = [
        (float(c.subs({"t": -np.pi / 4})), w)
        if isinstance(c, sympy.Basic) else (c, w)
        for c, w in _factorized
    ]

    returned.extend(pauli_word_exp_factorization(
        coefficient=coefficient,
        pauli_word=PauliWord.join(p1, wk1, i2)
    ))

    returned.extend([
        (float(c.subs({"t": np.pi / 4})), w)
        if isinstance(c, sympy.Basic) else (c, w)
        for c, w in _factorized
    ])

    # returned = pauli_word_exp_factorization(
    #     coefficient=-np.pi / 4,
    #     pauli_word=PauliWord.join(i1, wk2, p2)
    # )
    #
    # returned.extend(pauli_word_exp_factorization(
    #     coefficient=coefficient,
    #     pauli_word=PauliWord.join(p1, wk1, i2)
    # ))
    #
    # returned.extend(pauli_word_exp_factorization(
    #     coefficient=np.pi / 4,
    #     pauli_word=PauliWord.join(i1, wk2, p2)
    # ))

    return returned
