import os
import collections
import copy
import random
import numpy as np

import qiskit
from qiskit_aer import AerSimulator

from collections import Counter
from itertools import product

import input_prep as inp


class Error(Exception):
    """Base class for other exceptions"""
    pass


class VirtualQubitError(Error):
    """Exception raised if the number of virtual qubits chosen is not suitable.
    Specifically, if the number of qubits chosen is too big
    """

    def __init__(self, message="Too many virtual qubits!"):
        self.message = message
        super().__init__(self.message)


class Pauli_Operator():
    """This class represents Hermitian Pauli operators and defines several
    operations that can be performed on and with them.
    """

    def __init__(self, bin_vec):
        """Initialize the attributes that are used to describe each
        Pauli_Operator instance (i.e., each Hermitian Pauli operator).

        Args:
            bin_vec (list([0/1])): binary list that characterizes the
            Pauli_Operator instance.
        """
        # The attributes of any Pauli_Operator instance are the corresponding
        # binary list and the number of elements in that list
        self.bin_vec = bin_vec
        self.nr_columns = len(bin_vec)

    def x(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated through a Pauli X gate acting on the i-th
        qubit of the `q_stab` quantum register (i.e., the stabilizer register).

        Args:
            i (int): number of the qubit the X gate is being applied to.
        """
        if self.bin_vec[i + int((self.nr_columns - 1) / 2)] == 1:
            self.bin_vec[-1] = (self.bin_vec[-1] + 1) % 2

    def z(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated through a Pauli Z gate acting on the i-th
        qubit of the `q_stab` quantum register (i.e., the stabilizer register).

        Args:
            i (int): number of the qubit the Z gate is being applied to.
        """
        if self.bin_vec[i] == 1:
            self.bin_vec[-1] = (self.bin_vec[-1] + 1) % 2

    def h(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated through a Hadamard gate acting on the i-th
        qubit of the `q_stab` quantum register.

        Args:
            i (int): number of the qubit the Hadamard is being applied to.
        """
        self.bin_vec[-1] = (self.bin_vec[-1] +
                            self.bin_vec[i] * self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)]) % 2
        if (self.bin_vec[i]
                != self.bin_vec[i + int((self.nr_columns - 1) / 2)]):
            aux = self.bin_vec[i]
            self.bin_vec[i] = self.bin_vec[i + int((self.nr_columns - 1) / 2)]
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = aux

    def s(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated to the RIGHT of a phase gate S acting on the
        i-th qubit of the `q_stab` quantum register.
        (Observation: left and right here are with respect to the circuit model
        depiction.)

        Args:
            i (int): number of the qubit the S gate is being applied to.
        """
        self.bin_vec[-1] = (self.bin_vec[-1] +
                            self.bin_vec[i] * self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)]) % 2
        if self.bin_vec[i] == 1:
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[i + int((self.nr_columns - 1) / 2)] + 1) % 2

    def sdg(self, i):
        """This method updates the Pauli_Operator instance it is applied to when
        that operator is propagated to the LEFT of a phase gate S acting on the
        i-th qubit of the `q_stab` quantum register. This is the transformation
        to use (instead of `s(i)`) when propagating the Pauli measurements from
        the end of the circuit to the beginning. In the case of all the other
        unitaries, there is no distinction because they are Hermitian operators,
        but in the case of the phase gate we need to be very careful with what
        we are doing.

        Args:
            i (int): nr. of the qubit the S^{†} gate is being applied to.
        """
        self.bin_vec[-1] = (self.bin_vec[-1] + self.bin_vec[i] *
                            (self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)] + 1)) % 2
        if self.bin_vec[i] == 1:
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[i + int((self.nr_columns - 1) / 2)] + 1) % 2

    def cx(self, i, j):
        """This method updates the Pauli_Operator instance it is applied to when
        that instance is propagated through a controlled-NOT gate CX controlled
        on the i-th qubit of the `q_stab` quantum register and targetting the
        j-th qubit of the system.

        Args:
            i (int): number of the control qubit.
            j (int): number of the target qubit.
        """
        self.bin_vec[-1] = (self.bin_vec[-1] + self.bin_vec[i] *
                            self.bin_vec[j + int((self.nr_columns - 1) / 2)] *
                            (self.bin_vec[j] + self.bin_vec[i + int(
                                (self.nr_columns - 1) / 2)] + 1)) % 2
        if self.bin_vec[i] == 1:
            self.bin_vec[j] = (self.bin_vec[j] + 1) % 2
        if self.bin_vec[j + int((self.nr_columns - 1) / 2)] == 1:
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[i + int((self.nr_columns - 1) / 2)] + 1) % 2

    def cz(self, i, j):
        """This method updates the Pauli_Operator instance it is applied to when
        that instance is propagated through a controlled-Z gate CZ controlled
        on the i-th qubit of the quantum register and targetting the j-th qubit
        of the system.

        Args:
            i (int): number of the control qubit.
            j (int): number of the target qubit.
        """
        self.bin_vec[-1] = (self.bin_vec[-1] +
                            self.bin_vec[i] * self.bin_vec[j]) % 2
        if self.bin_vec[i] == 1:
            self.bin_vec[j + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[j + int((self.nr_columns - 1) / 2)] + 1) % 2
        if self.bin_vec[j] == 1:
            self.bin_vec[i + int((self.nr_columns - 1) / 2)] = (
                self.bin_vec[i + int((self.nr_columns - 1) / 2)] + 1) % 2

    def v(self, P, Q, sP, sQ):
        """This method updates the Pauli_Operator instance it is applied to when
        propagating it to the left of the Clifford
        V = [ (-1)^sP * P + (-1)^sQ * Q ] / sqrt(2).

        Args:
            P (Pauli_Operator): Pauli operator instance P.
            Q (Pauli_Operator): Pauli operator instance Q.
            sP (int): Measurement outcome (0 or 1) obtained for P.
            sQ (int): Measurement outcome (0 or 1) obtained for Q.
        """
        symprodP = self.symplectic_inner_product(P)
        symprodQ = self.symplectic_inner_product(Q)

        if (symprodP == 0 and symprodQ == 1):
            RPQ = self.RPQproduct(P, Q)
            self.bin_vec = RPQ.bin_vec

            self.bin_vec[-1] = (self.bin_vec[-1] + sP + sQ) % 2

        elif (symprodP == 1 and symprodQ == 0):
            RPQ = self.RPQproduct(P, Q)
            self.bin_vec = RPQ.bin_vec

            self.bin_vec[-1] = (self.bin_vec[-1] + sP + sQ + 1) % 2

        elif (symprodP == 1 and symprodQ == 1):
            self.bin_vec[-1] = (self.bin_vec[-1] + 1) % 2

    def gen_update(self, gate_sequence, q_count):
        """This method automatically updates the Pauli_Operator instance under
        consideration by BACKPROPAGATING it through the sequence of Clifford
        unitaries specified in the `gate_sequence` list.

        Args:
            gate_sequence (list[str]): list that establishes the gate sequence
            through which the Pauli needs to be propagated; this list has
            elements that are strings 'x (...)', 'z (...)', 'h (...)',
            's (...)', 'sdg (...)', 'cx (...)', 'cz (...)', or 'v (...)'; each
            line has a certain default format and the knowledge of that format
            can be used to read the information encoded on it.
            q_count (int): number of qubits in the original Clifford+T circuit.
        """
        for line in gate_sequence:
            if line.startswith('x '):
                x = line.partition('q_stab[')
                y = x[2].partition(']')
                i = int(y[0])
                self.x(i)

            elif line.startswith('z '):
                x = line.partition('q_stab[')
                y = x[2].partition(']')
                i = int(y[0])
                self.z(i)

            elif line.startswith('h '):
                x = line.partition('q_stab[')
                y = x[2].partition(']')
                i = int(y[0])
                self.h(i)

            elif line.startswith('s '):
                x = line.partition('q_stab[')
                y = x[2].partition(']')
                i = int(y[0])
                self.sdg(i)

            elif line.startswith('sdg '):
                x = line.partition('q_stab[')
                y = x[2].partition(']')
                i = int(y[0])
                self.s(i)

            elif line.startswith('cx '):
                w = line.partition('q_stab[')
                x = w[2].partition('],')
                i = int(x[0])
                y = x[2].partition('[')
                z = y[2].partition(']')
                if y[0] == 'q_stab': j = int(z[0])
                elif y[0] == 'q_magic': j = int(z[0]) + q_count
                self.cx(i, j)

            elif line.startswith('cz '):
                w = line.partition('q_stab[')
                x = w[2].partition('],')
                i = int(x[0])
                y = x[2].partition('[')
                z = y[2].partition(']')
                j = int(z[0])
                self.cz(i, j)

            elif line.startswith('v '):
                v = line.partition(' ')
                w = v[2].partition(' ')
                sP = int(w[0])
                x = w[2].partition(' ')
                sQ = int(x[0])
                y = x[2].partition(' ')
                P = Pauli_Operator([int(y[0][i]) for i in range(len(y[0]))])
                z = y[2].partition(';')
                Q = Pauli_Operator([int(z[0][i]) for i in range(len(z[0]))])
                self.v(P, Q, sP, sQ)

    def return_Pauli_str(self):
        """Returns the list of single-qubit Pauli operators that correspond to
        the Pauli_Operator instance.
        (Mostly useful for functionality testing, debugging, and output
        presentation.)

        Returns:
            str: string indicating the sequence of single-qubit Pauli operators
            (I, X, Y, Z) that constitute the Pauli_Operator instance
        """
        Pauli_str = ''
        if self.bin_vec[-1] == 0: Pauli_str += '+'
        else: Pauli_str += '-'

        for i in range(int((self.nr_columns - 1) / 2)):
            if (self.bin_vec[i] == 0
                    and self.bin_vec[i + int((self.nr_columns - 1) / 2)] == 0):
                Pauli_str += 'I'
            elif (self.bin_vec[i] == 0
                  and self.bin_vec[i + int((self.nr_columns - 1) / 2)] == 1):
                Pauli_str += 'Z'
            elif (self.bin_vec[i] == 1
                  and self.bin_vec[i + int((self.nr_columns - 1) / 2)] == 0):
                Pauli_str += 'X'
            else:
                Pauli_str += 'Y'

        return Pauli_str

    def symplectic_inner_product(self, P):
        """Method that calculates the symplectic inner product between the
        Pauli_Operator instance to which this method is applied, and the
        Pauli_Operator instance P.

        Args:
            P (Pauli_Operator): Pauli operator instance with which we want
            to perform the symplectic inner product.

        Returns:
            int (0/1): value of the symplectic inner product.
        """
        P1 = self.bin_vec.copy()
        P2 = P.bin_vec.copy()
        P1.pop(-1)
        P2.pop(-1)

        symprod = 0
        for i in range(int((self.nr_columns - 1) / 2)):
            symprod = (symprod +
                       (P1[i] * P2[int((self.nr_columns - 1) / 2) + i] +
                        P1[int((self.nr_columns - 1) / 2) + i] * P2[i])) % 2

        return symprod

    def RPQproduct(self, P, Q):
        """This method calculates the triple product between the Pauli_Operator
        instance R to which it is applied (i.e., `self`) and the Pauli instances
        P and Q, that is, it calculates the Hermitian Pauli_Operator instance
        R' = PQR. Note that R' will be Hermitian because the PQR product will
        only be applied in situations wherein P and Q anti-commute, and R anti-
        commutes with either P or Q (but not both simultaneously). In situations
        other than this specific one, this PQR product is not valid and should
        not be employed.

        Args:
            P (Pauli_Operator): Pauli operator instance P.
            Q (Pauli_Operator): Pauli operator instance Q.

        Returns:
            Pauli_Operator: Hermitian Pauli_Operator instance R'=PQR.
        """
        PQ_prod = [0] * self.nr_columns
        RPQ_prod = [0] * self.nr_columns

        # start by multiplying P and Q:
        for i in range(self.nr_columns - 1):
            PQ_prod[i] = (P.bin_vec[i] + Q.bin_vec[i]) % 2

        contribution = 0
        for i in range(int((self.nr_columns - 1) / 2)):
            if (P.bin_vec[i] == 1
                    and P.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                if (Q.bin_vec[i] == 1 and
                        Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1
                elif (Q.bin_vec[i] == 0
                      and Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

            elif (P.bin_vec[i] == 1
                  and P.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (Q.bin_vec[i] == 1 and
                        Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 3
                elif (Q.bin_vec[i] == 0
                      and Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1

            elif (P.bin_vec[i] == 0
                  and P.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (Q.bin_vec[i] == 1 and
                        Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 1
                elif (Q.bin_vec[i] == 1
                      and Q.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

        PQ_prod[-1] = (2 * P.bin_vec[-1] + 2 * Q.bin_vec[-1] +
                       contribution) % 4

        # next calculate R*(PQ):
        for i in range(self.nr_columns - 1):
            RPQ_prod[i] = (self.bin_vec[i] + PQ_prod[i]) % 2

        contribution = 0
        for i in range(int((self.nr_columns - 1) / 2)):
            if (self.bin_vec[i] == 1
                    and self.bin_vec[int((self.nr_columns - 1) / 2) + i] == 0):
                if (PQ_prod[i] == 1
                        and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1
                elif (PQ_prod[i] == 0
                      and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

            elif (self.bin_vec[i] == 1
                  and self.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (PQ_prod[i] == 1
                        and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 3
                elif (PQ_prod[i] == 0
                      and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 1

            elif (self.bin_vec[i] == 0
                  and self.bin_vec[int((self.nr_columns - 1) / 2) + i] == 1):
                if (PQ_prod[i] == 1
                        and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 0):
                    contribution += 1
                elif (PQ_prod[i] == 1
                      and PQ_prod[int((self.nr_columns - 1) / 2) + i] == 1):
                    contribution += 3

        RPQ_prod[-1] = (2 * self.bin_vec[-1] + PQ_prod[-1] + contribution) % 4

        if RPQ_prod[-1] == 2: RPQ_prod[-1] = 1

        return Pauli_Operator(RPQ_prod)

    def is_independent(self, q_count, t_count, pipc_Paulis, M):
        """This method checks whether or not the Pauli operator it is applied to
        is independent from all previous (independent) Paulis in the Pauli-based
        computation.

        Args:
            q_count (int): number of qubits in the original Clifford+T circuit.
            t_count (int): number of T gates in the original Clifford+T circuit.
            pipc_Paulis (list([Pauli_Operator])): list with all of the
            previously measured (independent and pairwise commuting)
            Pauli_Operator instances.
            M: matrix we use to compute the nullspace(), and which is
            changed and updated as the PBC procedure progresses.

        Returns:
            tuple (diverse, list): The first element of the tuple has 3
            different possible types depending on the situation:
            (a) int (0/1): outcome of the Pauli if it is trivial in the magic
            register;
            (b) 'True': if the Pauli is independent from all previous Paulis;
            (c) list([0/1]): list of zeros and ones which is the kernel of
            the matrix made up by the previous (independent) Paulis and the
            current one. This list identifies which are the previous Pauli
            operators that the current Pauli depends on.
            The second element of the tuple is the matrix M, which needs to be
            tracked during the whole computation.
        """
        if (self.bin_vec[q_count:q_count + t_count]
                == [0 for _ in range(t_count)]
                and self.bin_vec[2 * q_count + t_count:-1]
                == [0 for _ in range(t_count)]):
            # if the Pauli is trivial in the ancillary (magic) qubits, its mmt
            # is (immediately) trivial and deterministic
            outcome = self.bin_vec[-1]
            return (int(outcome), M)

        else:
            # if the Pauli is not trivial in the ancillas it may be dependent
            # or it may be independent
            if len(pipc_Paulis) == q_count:
                #if this is the case we get the first independent Pauli:
                P = self.bin_vec.copy()
                M.append(P[q_count:q_count + t_count] +
                         P[2 * q_count + t_count:] + [1] + [0] * t_count)
                return ('True', M)

            else:
                P = self.bin_vec.copy()
                position = len(M)
                vec = [0] * (t_count + 1)
                vec[position] = 1
                M.append(P[q_count:q_count + t_count] +
                         P[2 * q_count + t_count:] + vec)
                (kernel, M) = nullspace(M)
                if kernel == []:
                    # The operator is independent from the previous operators.
                    return ('True', M)
                else:
                    return (kernel, M)

    def is_commuting(self, q_count, t_count, pipc_Paulis):
        """This method checks out whether the Pauli_Operator instance commutes
        with all previously measured (independent and pairwise commuting) Paulis
        in the Pauli-based computation.

        Args:
            q_count (int): number of qubits in the original Clifford+T circuit.
            t_count (int): number of T gates in the original Clifford+T circuit.
            pipc_Paulis (list([Pauli_Operator])): list with all of the
            previously measured (independent and pairwise commuting)
            Pauli_Operator instances

        Returns:
            'True': if the Pauli operator commutes with all previous Paulis
            int: the index of the first previous Pauli operator with which the
            current Pauli anti-commutes
        """
        for index in range(q_count):
            if self.bin_vec[index] == 1:
                # The Pauli operator anti-commutes at least with Z_{index}
                return index

        P = self.bin_vec.copy()
        P = P[q_count:q_count + t_count] + P[2 * q_count + t_count:]

        reduced_pipc_Paulis = []
        for Pauli in pipc_Paulis[q_count:]:
            Q = Pauli.bin_vec.copy()
            reduced_pipc_Paulis.append(
                Pauli_Operator(Q[q_count:q_count + t_count] +
                               Q[2 * q_count + t_count:]))

        count = q_count
        index = q_count
        for Pauli in reduced_pipc_Paulis:
            sip = Pauli_Operator(P).symplectic_inner_product(Pauli)
            if sip == 1:
                break
            elif sip == 0:
                count += 1
            index += 1

        if count == len(pipc_Paulis):
            # The Pauli operator instance commutes with all previous Paulis
            return 'True'
        else:
            # The Pauli operator anti-commutes at least with the Pauli #{index}
            return index


def Pauli_mult(q_count, P1, P2):
    """This function multiplies the two Pauli operators P1 and P2. The way to
    do so is explained in the paper by Aaronson and Gottesman
    [https://doi.org/10.1103/PhysRevA.70.052328].

    Args:
        q_count (int): number of qubits in the Clifford+T quantum circuit.
        P1 (Pauli_Operator): one of the operators to be multiplied.
        P2 (Pauli_Operator): the other operator to be multiplied.

    Returns:
        Pauli_Operator: the Pauli operator instance that is the result of the
        product.
    """
    vec1 = P1.bin_vec
    vec2 = P2.bin_vec
    res = [0] * P1.nr_columns
    for j in range(P1.nr_columns - 1):
        res[j] = (vec1[j] + vec2[j]) % 2

    gsum = 0
    for j in range(q_count, int((P1.nr_columns - 1) / 2)):
        x1 = vec1[j]
        z1 = vec1[j + int((P1.nr_columns - 1) / 2)]
        x2 = vec2[j]
        z2 = vec2[j + int((P1.nr_columns - 1) / 2)]

        if x1 == 1 and z1 == 1:
            gsum += z2 - x2
        elif x1 == 1 and z1 == 0:
            gsum += z2 * (2 * x2 - 1)
        elif x1 == 0 and z1 == 1:
            gsum += x2 * (1 - 2 * z2)

    gsum = (gsum + 2 * vec1[-1] + 2 * vec2[-1]) % 4

    if gsum == 0:
        res[-1] = 0

    elif gsum == 2:
        res[-1] = 1

    return Pauli_Operator(res)


def nullspace(matrix):
    """This function receives a matrix input as a list of list (i.e., a nested
    list) where each line represents a different binary vector and returns a
    list indicating which of the previously independent and pairwise communting
    Paulis can be used to obtain the current Pauli under test. If all vectors
    are linearly independent (mod 2), then the function returns an empty list.

    Args:
        matrix (list): nested list representing the matrix containing one
        binary vector per line.

    Returns:
        tuple (list, list): (1) list with entries 0 or 1, or empty list. In the
        former case, the entries with value 1 indicate the Pauli operators
        which are linearly dependent on one another. (2) updated matrix used to
        assess linear (in)dependence in future rounds.
    """
    t_count = int((len(matrix[0]) - 2) / 3)
    n_rows = len(matrix)
    all_columns = [i for i in range(len(matrix[0]))]
    sel_columns = [i for i in range(2 * t_count)]
    is_used = [False] * (2 * t_count)
    used_columns = []

    track_rc = [-1] * 2 * t_count  # tracks which column is 1 only for row r
    for r in range(n_rows - 1):
        for c in sel_columns:
            if matrix[r][c] == 1:
                track_rc[c] = r
                is_used[c] = True
                used_columns.append(c)
                break

    for c1 in used_columns:
        if matrix[-1][c1] == 1:
            for c2 in all_columns:
                matrix[-1][c2] = (matrix[-1][c2] +
                                  matrix[track_rc[c1]][c2]) % 2

    if matrix[-1][:-(t_count + 2)] == [0] * (2 * t_count):
        # The current Pauli depends on previous Paulis:
        kernel = matrix[-1][2 * t_count + 1:2 * t_count + 1 + n_rows]
        matrix.pop()  # we eliminate the Pauli from the matrix
        return (kernel, matrix)

    else:
        # The current Pauli is independent from previous Paulis:
        kernel = []  # we leave the kernel empty to signal independence

        # The matrix will need to be modified so that this new Pauli is 1 in
        # a column where all other Paulis are 0:
        for c1 in sel_columns:
            if not is_used[c1] and matrix[-1][c1] == 1:
                track_rc[-1] = c1
                break

        flagged = track_rc[-1]
        for r in range(n_rows - 1):
            if matrix[r][flagged] == 1:
                for c in all_columns:
                    matrix[r][c] = (matrix[r][c] + matrix[-1][c]) % 2

        return (kernel, matrix)


def initialize_lists(q_count, t_count):
    """This function initializes the list of Pauli_Operator instances with
    which the PBC procedure is initiated. More specifically, it builds the
    following Python list: list([Z_1,Z_2,...,Z_{q_count}]).

    Args:
        q_count (int): number of qubits in the original Clifford+T circuit.
        t_count (int): number of T gates in the original Clifford+T circuit.

    Returns:
        list([Pauli_Operators]): list with which the PBC procedure is
        instantiated.
    """
    Pauli_list = []

    for row in range(q_count):
        bin_vec = [0] * (2 * (q_count + t_count) + 1)
        bin_vec[row + q_count + t_count] = 1
        Pauli_list.append(Pauli_Operator(bin_vec))

    return Pauli_list


def find_new_Pauli(circuit_list, q_count, t_count):
    """Function which reads the adaptive (gadgetized) Clifford circuit encoded
    in `circuit_list` and finds the next Pauli operator in the PBC measurement
    sequence.

    Args:
        circuit_list (list[str]): list of operations associated with the
        adaptive Clifford circuit with magic state injection.
        q_count (int): number of qubits in the original Clifford+T circuit.
        t_count (int): number of T gates in the original Clifford+T circuit.

    Returns:
        tuple (list[str], Pauli_Operator): updated circuit list and the new
        Pauli_Operator instance.
    """
    bin_vec = [0] * (2 * (q_count + t_count) + 1)
    for index, line in enumerate(circuit_list):
        if line.startswith('measure '):
            x = line.partition('[')
            y = x[2].partition(']')
            i = int(y[0])
            z = x[0].partition(' ')
            if z[2] == 'q_stab':
                bin_vec[i + q_count + t_count] = 1
            elif z[2] == 'q_magic':
                bin_vec[i + 2 * q_count + t_count] = 1
            circuit_list[index] = ''
            break

    Pauli = Pauli_Operator(bin_vec)
    gate_sequence = []
    i = 0
    for line in circuit_list[:index]:
        i += 1
        if (line.startswith('x ') or line.startswith('z ')
                or line.startswith('h ') or line.startswith('s ')
                or line.startswith('sdg ') or line.startswith('cx ')
                or line.startswith('cz ') or line.startswith('v ')):
            gate_sequence.append(line)

    gate_sequence.reverse()
    Pauli.gen_update(gate_sequence, q_count)

    return circuit_list, Pauli


def cancel(lst):
    """This function takes as input a list of integers and cancels pairs
    of repeated integers. E.g.: [1,1,2,1,3] = [2,1,3].

    Args:
        lst (list): list of integers.

    Returns:
        list: list where all pairs of repeated integers have been removed.
    """
    counts = collections.Counter(lst)
    cancelled = [k for k, v in counts.items() if v % 2 != 0]
    cancelled.sort()
    return cancelled


def full_zero_branch(circuit_list, q_count, t_count, nr_mmts):
    """This function finds the full PBC branch assuming that all gadget
    measurements yield the outcome zero.

    Args:
        circuit_list (list[str]): list of operations describing the adaptive
        Clifford circuit.
        q_count (int): number of qubits in the original Clifford+T circuit.
        t_count (int): number of T gates in the original circuit.
        nr_mmts (int): total number of measurements in the gadgetized circuit
        (equals the number of readout measurements in the orignal circuit plus
        the number of T gates).

    Returns:
        list[PauliOperators]: list with the full PBC branch obtained if all
        gadget measurements yield 0.
    """
    # We post-select all gadget measurements to 0 to find the full zero branch
    for index, line in enumerate(circuit_list):
        if line.startswith('if'):
            circuit_list[index] = ''

    full_branch = []
    for _ in range(nr_mmts):
        circuit_list, current_Pauli = find_new_Pauli(circuit_list, q_count,
                                                     t_count)
        full_branch.append(current_Pauli)

    return full_branch


def compile_branch(branch, q_count, t_count, n_gadgets=None):
    """Function that compiles the full PBC branch `branch` into a list of
    pairwise commuting Pauli measurements to be performed only on the magic
    states. The compilation assumes that all branching measurements yield
    zero and returns a compiled branch together with additional information
    concerning the compilation procedure, including sign interdependences and
    outcome correlations.

    Args:
        branch (list[PauliOperators]): full zero branch of the PBC.
        q_count (int): number of qubits in the original Clifford+T circuit.
        t_count (int): number of T gates in the original Clifford+T circuit.
        n_gadgets (int, optional): identifies how many gadget measurements
        there are. Defaults to None, in which case it means that the number
        of gadget measurements matches the number of magic qubits.

    Returns:
        tuple(list[PauliOperators], list[str], dic{}, dic{}, list[int]): tuple
        with (1) the compiled PBC branch, that is, the list of quantum
        measurements along branch 0 that can be used to generate all other
        branches; (2) a list that flags the type of each measurement along
        the full branch; (3) a dictionary that records which Pauli operators
        have signs that depend on which previously measured Paulis or
        anti-commuting readouts; (4) a dictionary that records which dependent
        Paulis depend on which quantum measurents (essential to fix the
        distribution when actually running the computation); (5) a list with
        pseudo-outcomes.
    """
    if n_gadgets is None:
        n_gadgets = t_count

    pipc_Ps = initialize_lists(q_count, t_count)
    pseudout = [0 for _ in range(len(branch))]
    matrix = []
    branchers = []
    types = []
    signs = {k: set() for k in range(len(branch))}
    correlats = {k: [] for k in range(len(branch))}

    count = 0
    while count < len(branch):
        current_Pauli = branch[count]
        index_ac_Pauli = current_Pauli.is_commuting(q_count, t_count, pipc_Ps)
        if index_ac_Pauli != 'True':
            types.append('a')  # We flag the measurement as anti-commuting
            P = pipc_Ps[index_ac_Pauli]  # The Pauli it anti-commutes with
            sP = 0
            Q = current_Pauli
            sQ = 0
            for i in range(count + 1, len(branch)):
                before = branch[i].bin_vec.copy()
                before.pop(-1)
                branch[i].v(P, Q, sP, sQ)  # updates upcoming Paulis
                after = branch[i].bin_vec.copy()
                after.pop(-1)
                # In the readouts we have to track classical dependences:
                if count >= n_gadgets:
                    if index_ac_Pauli < q_count and before != after:
                        signs[i] ^= {count}
                        signs[i] ^= signs[count]

                    elif index_ac_Pauli >= q_count and before != after:
                        gen_ind = [
                            j for j, val in enumerate(types) if val == 'q'
                        ][index_ac_Pauli - q_count]
                        signs[i] ^= {gen_ind}
                        signs[i] ^= signs[gen_ind]
                        signs[i] ^= {count}
                        signs[i] ^= signs[count]

        else:
            kernel, matrix = current_Pauli.is_independent(
                q_count, t_count, pipc_Ps, matrix)

            if count < n_gadgets:
                types.append('q')
                # In the case of gadget measurements, if they are not
                # anti-commuting then they are independent
                pipc_Ps.append(current_Pauli)
                branchers.append(count)

            else:
                if isinstance(kernel, int):
                    types.append('d1')
                    pseudout[count] = kernel

                elif isinstance(kernel, list):
                    types.append('d2')
                    nr_rows = len(kernel)
                    prod_P = Pauli_Operator([0] * current_Pauli.nr_columns)
                    for i in range(nr_rows - 1):
                        if kernel[i] == 1:
                            gen_ind = [
                                j for j, val in enumerate(types) if val == 'q'
                            ][i]
                            correlats[count].append(gen_ind)
                            prod_P = Pauli_mult(q_count, prod_P,
                                                pipc_Ps[q_count + i])
                    if prod_P.bin_vec[-1] == current_Pauli.bin_vec[-1]:
                        correlats[count].append("+")
                    else:
                        correlats[count].append("-")
                else:
                    types.append('q')
                    # If the Pauli is independent, we append it to the list
                    # of Pauli operators to be measured in quantum hardware
                    pipc_Ps.append(current_Pauli)

        count += 1

    for key, val in signs.items():
        signs[key] = sorted(val)

    return (pipc_Ps[q_count:], types, signs, correlats, pseudout)


def remove_free_qubits(branch, q_count, t_count):
    """Function that takes the full zero branch associated with a given input
    quantum circuit, does a first compilation (compressing the branch so that 
    the quantum computation acts only on the magic qubits), and then given the
    already compiled branch, assesses if there are additional (magic) qubits
    that can be removed for free. If that is the case, a second compilation is
    performed, compressing the quantum computation even further.

    Args:
        branch (list[PauliOperators]): full zero branch of the PBC.
        q_count (int): number of qubits in the original Clifford+T circuit.
        t_count (int): number of T gates in the original Clifford+T circuit.

    Returns:
        tuple(list[PauliOperators], list[str], dic{}, dic{}, list[int],
        dic{}, int): Tuple with (1) the compiled PBC branch, that is, the list
        of quantum measurements along branch 0 that can be used to generate
        all other branches; (2) a list that flags the type of each measurement
        along the full branch; (3) a dictionary that records which Pauli
        operators have signs that depend on which previously measured Paulis or
        anti-commuting readouts; (4) a dictionary that records which dependent
        Paulis depend on which quantum measurents (essential to fix the
        distribution when actually running the computation); (5) a list with
        pseudo-outcomes; (6) a dictionary recording which magic qubit the Sdg
        gate associated with each gadget measurement must be acting on;
        (7) an integer corresponding to the "effective" number of stabilizer
        qubits.
    """
    # STEP 1. We get the compiled branch:
    comp_branch1, types1, signs1, correlats1, pseudoutcomes1 = compile_branch(
        branch, q_count, t_count)

    sdg_tracker = dict()  # To track where the Sdg gates should be applied
    n_true_new_gadgets = 0
    for i, val in enumerate(types1):
        if val == 'q' and i < t_count:
            sdg_tracker[i] = i
            n_true_new_gadgets += 1

    # STEP 2. We investigate whether there exist magic qubits that can be
    # freely removed:
    free_mqubits = []
    for i in range(t_count):
        count = 0
        for Pauli in comp_branch1:
            if Pauli.bin_vec[q_count + i] == 0:
                count += 1

        if count == len(comp_branch1):
            free_mqubits.append(i)
            for Pauli in comp_branch1:
                Pauli.h(q_count + i)

    # STEP 3. If there are no free qubits, return.
    if len(free_mqubits) == 0:
        return (comp_branch1, types1, signs1, correlats1, pseudoutcomes1,
                sdg_tracker, q_count)

    # STEP 4. If there *are* free magic qubits, do a second compilation step
    # that removes those.
    else:
        # STEP 4.1. `free_mqubits` identifies which of the magic qubits will be
        # freely removed. This helps us to track down the correct position of
        # the Sdg gates and update the `sdg_tracker` dict.
        for i in free_mqubits:
            for key, val in sdg_tracker.items():
                if key > i:
                    sdg_tracker[key] -= 1

        # STEP 4.2. Reorder the qubits so that the free magic qubits are the
        # first ones in the magic register. We will then increase q_count by
        # the number of free magic qubits, making the stabilizer register
        # effectively larger and the magic register effectively smaller.
        count = q_count
        for i in free_mqubits:
            for Pauli in comp_branch1:
                Pauli.bin_vec.insert(count, Pauli.bin_vec.pop(q_count + i))
                Pauli.bin_vec.insert(
                    count + q_count + t_count,
                    Pauli.bin_vec.pop(2 * q_count + t_count + i))
            count += 1

        q_count = q_count + len(free_mqubits)
        t_count = t_count - len(free_mqubits)
        reassign = [re for re, val in enumerate(types1) if val == 'q']

        comp_branch2, types2, signs2, correlats2, pseudoutcomes2 = compile_branch(
            comp_branch1, q_count, t_count, n_gadgets=n_true_new_gadgets)

        # We now need to update types1, signs1, correlats1, pseudoutcomes1
        # based on types2, signs2, correlats2, pseudoutcomes2.
        for i in range(len(comp_branch1)):
            if types1[reassign[i]] != types2[i]:
                types1[reassign[i]] = types2[i]

            signs1[reassign[i]] = signs1[reassign[i]] + [
                reassign[j] for j in signs2[i]
            ]
            correlats1[reassign[i]] = correlats1[reassign[i]] + [
                reassign[x] if isinstance(x, int) else x for x in correlats2[i]
            ]
            pseudoutcomes1[reassign[i]] = (pseudoutcomes1[reassign[i]] +
                                           pseudoutcomes2[i]) % 2

        for key, val in signs1.items():
            signs1[key] = cancel(val)

        branchers = [
            i for i, val in enumerate(types1)
            if val == 'q' and i < (t_count + len(free_mqubits))
        ]

        track_keys = list(sdg_tracker.keys())
        for key in track_keys:
            if key not in branchers:
                del sdg_tracker[key]

        return (comp_branch2, types1, signs1, correlats1, pseudoutcomes1,
                sdg_tracker, q_count)


def remove_virtual_qubits(branch, q_count, t_count, stab_decomp):
    """Function that removes a certain number of virtual qubits by replacing
    a certain number of magic qubits by stabilizer states described by
    `stab_decomp`.

    Args:
        branch (list[PauliOperators]): full zero branch of the PBC.
        q_count (int): number of qubits in the original Clifford+T circuit.
        t_count (int): number of T gates in the original Clifford+T circuit.
        stab_decomp (tuple): tuple of integer numbers describing what is the
        stabilizer state that is replacing the magic states.

    Raises:
        VirtualQubitError: if the number of virtual qubits is larger than
        the number of qubits in the PBC, then an error is raised.

    Returns:
        tuple(list[PauliOperators], list[str], dic{}, dic{}, list[int],
        dic{}, int): Tuple with (1) the compiled PBC branch, that is, the list
        of quantum measurements along branch 0 that can be used to generate
        all other branches; (2) a list that flags the type of each measurement
        along the full branch; (3) a dictionary that records which Pauli
        operators have signs that depend on which previously measured Paulis or
        anti-commuting readouts; (4) a dictionary that records which dependent
        Paulis depend on which quantum measurents (essential to fix the
        distribution when actually running the computation); (5) a list with
        pseudo-outcomes; (6) a dictionary recording which magic qubit the Sdg
        gate associated with each gadget measurement must be acting on;
        (7) an integer corresponding to the "effective" number of stabilizer
        qubits.
    """

    original_tcount = t_count

    # STEP 1. We get the compiled branch (with free qubits already removed)
    comp_branch1, types1, signs1, correlats1, pseudoutcomes1, sdg_tracker, new_qcount = remove_free_qubits(
        branch, q_count, t_count)

    reassign = [re for re, val in enumerate(types1) if val == 'q']
    n_true_gadgets = sum(1 for r in reassign if r < original_tcount)

    for i in range(len(stab_decomp)):
        for Pauli in comp_branch1:
            if stab_decomp[i] == 0:
                Pauli.h(new_qcount + i)
            elif stab_decomp[i] == 1:
                Pauli.z(new_qcount + i)
                Pauli.h(new_qcount + i)
            elif stab_decomp[i] == 2:
                Pauli.sdg(new_qcount + i)
                Pauli.h(new_qcount + i)
            else:
                raise ValueError

    new_tcount = t_count - (new_qcount - q_count)

    for key, val in sdg_tracker.items():
        sdg_tracker[key] -= len(stab_decomp)

    if len(stab_decomp) > new_tcount:
        raise VirtualQubitError
    else:
        q_count = new_qcount + len(stab_decomp)
        t_count = new_tcount - len(stab_decomp)

    comp_branch2, types2, signs2, correlats2, pseudoutcomes2 = compile_branch(
        comp_branch1, q_count, t_count, n_gadgets=n_true_gadgets)

    # We now need to update types1, signs1, correlats1, pseudoutcomes1
    # based on types2, signs2, correlats2, pseudoutcomes2.
    for i in range(len(comp_branch1)):
        if types1[reassign[i]] != types2[i]:
            types1[reassign[i]] = types2[i]
            if types1[reassign[i]] == 'q':
                del sdg_tracker[reassign[i]]

        signs1[reassign[i]] = signs1[reassign[i]] + [
            reassign[j] for j in signs2[i]
        ]
        correlats1[reassign[i]] = correlats1[reassign[i]] + [
            reassign[x] if isinstance(x, int) else x for x in correlats2[i]
        ]
        pseudoutcomes1[reassign[i]] = (pseudoutcomes1[reassign[i]] +
                                       pseudoutcomes2[i]) % 2

    for key, val in signs1.items():
        signs1[key] = cancel(val)

    branchers = [
        reassign[i] for i, val in enumerate(types2)
        if val == 'q' and i < original_tcount
    ]

    for key in list(sdg_tracker.keys()):
        if key not in branchers:
            del sdg_tracker[key]

    return (comp_branch2, types1, signs1, correlats1, pseudoutcomes1,
            sdg_tracker, q_count)


def get_pbc_tree(file_loc, input_file_name, clifford_file_name,
                 info_file_name):
    """This function saves into a file all the information that we need
    to construct a compiled quantum circuit using PBC. 

    Args:
        file_loc (str): location of the input circuit files.
        input_file_name (str): name of the file with the input circuit.
        clifford_file_name (str): name used to save the file with the
        gadgetized adaptive Clifford circuit created from the input circuit.
        info_file_name (str): name used to save the file with all the
        information needed to construct the PBC-compiled quantum circuit.
    """
    # Creating the folder for the output files:
    if not os.path.exists(f'{file_loc}output'):
        os.mkdir(f'{file_loc}output')

    # STEP 1: Creates the gadgetized (i.e. adaptive) Clifford circuit which is
    # equivalent to the original universal quantum circuit.
    qc = inp.QuCirc(file_loc, input_file_name)
    clifford, q_count, t_count, nr_mmts = qc.msi_circuit(
        file_loc, clifford_file_name)
    # q_count -> nr. of qubits of the original quantum circuit
    # t_count -> nr. of T-gates of the original quantum circuit
    # nr_mmts -> nr. of measurements of the (new) gadgetized circuit

    full = full_zero_branch(clifford, q_count, t_count, nr_mmts)
    comp_branch0, tps, sgns, correlats, pseudoutcomes, tracker, new_q_count = remove_free_qubits(
        full, q_count, t_count)

    with open(f'{file_loc}output/{info_file_name}.txt', 'w') as file_object:
        for Pauli in comp_branch0:
            file_object.write(Pauli.return_Pauli_str()[0] +
                              Pauli.return_Pauli_str()[new_q_count + 1:] +
                              '\n')
        file_object.write('\n')
        file_object.write('Type of each of the measurements: ' + str(tps) +
                          '\n')
        file_object.write('\n')
        file_object.write(
            'Branching at gadget measurements: ' +
            str([i
                 for i, val in enumerate(tps) if val == 'q' and i < t_count]) +
            '\n')
        file_object.write('\n')
        file_object.write(
            'Reassignment of qubits for branchers (i.e., Sdg gates): ' +
            str(tracker) + '\n')
        file_object.write('\n')
        file_object.write('Identifying PIPC: ' +
                          str([i for i, val in enumerate(tps) if val == 'q']) +
                          '\n')
        file_object.write('\n')
        file_object.write('Classical tracking of sign flipping: ' + str(sgns) +
                          '\n')
        file_object.write('\n')
        file_object.write('Classical correlations of outcomes: ' +
                          str(correlats) + '\n')
        file_object.write('\n')
        file_object.write('Pseudo-outcomes: ' + str(pseudoutcomes) + '\n')


def naive_pbc_simulator(file_loc,
                        input_file_name,
                        clifford_file_name,
                        histo_file,
                        shots=1024):
    """This function simulates the input quantum circuit using the normal form
    theorem, that is, it produces samples from the correct output distribution.
    This is done in a somewhat naive way (there are ways of constructing better
    circuits), but the goal here is merely to show that the compilation is
    working correctly and that the correct output distribution is retrieved
    using our procedure.

    Args:
        file_loc (str): location of the input circuit files.
        input_file_name (str): name of the file with the input circuit.
        clifford_file_name (str): name used to save the file with the
        gadgetized adaptive Clifford circuit created from the input circuit.
        histo_file (str): name of the file that will store the histogram with
        the outcomes of the various shots.
        shots (int, optional): number of shots. Defaults to 1024.
    """
    # STEP 1: Creating the folder for the output files.
    if not os.path.exists(f'{file_loc}output'):
        os.mkdir(f'{file_loc}output')

    # STEP 2: Creating the gadgetized (i.e. adaptive) Clifford circuit which is
    # equivalent to the original universal quantum circuit.
    qc = inp.QuCirc(file_loc, input_file_name)
    clifford, q_count, t_count, nr_mmts = qc.msi_circuit(
        file_loc, clifford_file_name)
    # q_count -> nr. of qubits of the original quantum circuit
    # t_count -> nr. of T-gates of the original quantum circuit
    # nr_mmts -> nr. of measurements of the (new) gadgetized circuit

    # STEP 3: Getting the compiled branch and the (classical) information
    # needed to construct the PBC tree.
    full = full_zero_branch(clifford, q_count, t_count, nr_mmts)
    comp_branch0, tps, sgns, correlats, pseudoutcomes, tracker, new_q_count = remove_free_qubits(
        full, q_count, t_count)

    branchers = [i for i, val in enumerate(tps) if val == 'q' and i < t_count]
    qu_mmts = [i for i, val in enumerate(tps) if val == 'q']

    # STEP 4: Constructing the PBC-compiled quantum circuits. In this step
    # the most important thing is to guarantee that we go through a valid
    # branch of the tree. Note that we do NOT change the outcomes of the
    # classical registers (as we do not know what they are until we run the
    # actual computation), but we make sure that whatever values the classical
    # register takes, it is correctly used to make the correct choice of the
    # following branch.
    effective_t_count = t_count - (new_q_count - q_count)
    counter = len(comp_branch0) - 1
    nr_branchers = len(branchers)
    if len(comp_branch0) == 0:
        qu_outcomes = []
        print("This is a fully classical computation!")
        print("The outcome will be determined by fully classical means!")

    else:
        qr = qiskit.QuantumRegister(effective_t_count, 'qr')
        cr = qiskit.ClassicalRegister(len(comp_branch0), 'cr')

        if effective_t_count == 1:
            qc = qiskit.QuantumCircuit(qr, cr)
            qc.h(qr[0])
            qc.t(qr[0])

            Pauli = comp_branch0[0]
            if (Pauli.bin_vec[new_q_count] == 0 and Pauli.bin_vec[-2] == 1):
                qc.measure(qr, cr)

            elif (Pauli.bin_vec[new_q_count] == 1 and Pauli.bin_vec[-2] == 0):
                qc.h(qr[0])
                qc.measure(qr, cr)

            elif (Pauli.bin_vec[new_q_count] == 1 and Pauli.bin_vec[-2] == 1):
                qc.h(qr[0])
                qc.s(qr[0])
                qc.h(qr[0])
                qc.measure(qr, cr)

        elif effective_t_count > 1:
            qaux = qiskit.QuantumRegister(1, 'qaux')
            qc = qiskit.QuantumCircuit(qr, qaux, cr)

            for i in range(effective_t_count):
                qc.h(qr[i])
                qc.t(qr[i])

            for p in range(len(comp_branch0)):
                Pauli = comp_branch0[p]

                qc.h(qaux)
                for i in range(new_q_count, int((Pauli.nr_columns - 1) / 2)):
                    j = i - new_q_count
                    if (Pauli.bin_vec[i] == 0 and
                            Pauli.bin_vec[i + int((Pauli.nr_columns - 1) / 2)]
                            == 1):
                        qc.cz(qaux, qr[j])

                    elif (Pauli.bin_vec[i] == 1 and
                          Pauli.bin_vec[i + int((Pauli.nr_columns - 1) / 2)]
                          == 0):
                        qc.cx(qaux, qr[j])

                    elif (Pauli.bin_vec[i] == 1 and
                          Pauli.bin_vec[i + int((Pauli.nr_columns - 1) / 2)]
                          == 1):
                        qc.sdg(qr[j])
                        qc.cx(qaux, qr[j])
                        qc.s(qr[j])
                qc.h(qaux)
                qc.measure(qaux, cr[counter])

                if p < nr_branchers:
                    # We are in the branching part of the circuit. We must make
                    # sure we run through the correct branch, with the correctly
                    # placed S† gates in place. Hence, we need to read whether
                    # there exist sign interdependencies.
                    signal = Pauli.bin_vec[-1]
                    for k in sgns[p]:
                        with qc.if_test(
                            (cr[len(comp_branch0) - 1 - k],
                             (comp_branch0[qu_mmts.index(k)].bin_vec[-1] + 1) %
                             2)):
                            signal = (signal + 1) % 2

                    with qc.if_test((cr[counter], (signal + 1) % 2)):
                        qc.sdg(qr[tracker[qu_mmts[p]]])

                qc.reset(qaux)
                counter -= 1

        # STEP 5: Running the circuit in the desired backend:
        sim = AerSimulator(method='statevector')
        result = sim.run(qiskit.transpile(qc, sim), shots=shots).result()
        counts = result.get_counts()

        qu_outcomes = []
        for key, val in counts.items():
            qu_outcomes += [key] * val

    # STEP 6: Result reinterpretation for the proper probability distribution
    true_dist = [pseudoutcomes.copy() for _ in range(shots)]
    for s in range(shots):
        for i in range(nr_mmts):
            typ = tps[i]
            sgn = sgns[i]
            cor = correlats[i]
            if typ == "a":
                if i >= t_count:
                    true_dist[s][i] = random.randint(0, 1)

            elif typ == "q":
                true_dist[s][i] = (
                    int(qu_outcomes[s][qu_mmts.index(i)]) +
                    comp_branch0[qu_mmts.index(i)].bin_vec[-1]) % 2

            elif typ == "d1" or typ == "d2":
                if cor != []:
                    for j in cor[:-1]:
                        true_dist[s][i] += true_dist[s][j]
                    if cor[-1] == "-":
                        true_dist[s][i] += 1
                    true_dist[s][i] = true_dist[s][i] % 2

            for j in sgn:
                true_dist[s][i] += true_dist[s][j]
            true_dist[s][i] = true_dist[s][i] % 2

    final_result = dict()
    for res in true_dist:
        res_str = "".join(str(x) for x in res[t_count - nr_mmts:])
        final_result[res_str] = final_result.get(res_str, 0) + 1

    with open(f'{file_loc}output/{histo_file}.txt', 'w') as file_object:
        file_object.write(f"For a total of {shots} shots, we obtain:\n")
        file_object.write(str(dict(sorted(final_result.items()))))


def get_virtualization_info(file_loc,
                            input_file_name,
                            clifford_file_name,
                            info_file_name,
                            virtual_qubits=1):
    """Function that outputs the normal form for the different PBCs that need
    to be run when simulating a certain number of virtual qubits.

    Args:
        file_loc (str): location of the input circuit files.
        input_file_name (str): name of the file with the input circuit.
        clifford_file_name (str): name used to save the file with the
        gadgetized adaptive Clifford circuit created from the input circuit.
        info_file_name (str): name used to save the file with all the
        information needed to construct the PBC-compiled quantum circuit.
        virtual_qubits (int, optional): desired number of virtual qubits.
        Defaults to 1.
    """
    # Creating the folder for the output files:
    if not os.path.exists(f'{file_loc}output'):
        os.mkdir(f'{file_loc}output')

    it = 0
    for combo in product(range(3), repeat=virtual_qubits):
        # STEP 1: Creates the gadgetized (i.e. adaptive) Clifford circuit which
        # is equivalent to the original universal quantum circuit.
        qc = inp.QuCirc(file_loc, input_file_name)
        clifford, q_count, t_count, nr_mmts = qc.msi_circuit(
            file_loc, clifford_file_name)
        # q_count -> nr. of qubits of the original quantum circuit
        # t_count -> nr. of T-gates of the original quantum circuit
        # nr_mmts -> nr. of measurements of the (new) gadgetized circuit

        full = full_zero_branch(clifford, q_count, t_count, nr_mmts)
        comp_branch0, tps, sgns, correlats, pseudoutcomes, tracker, new_q_count = remove_virtual_qubits(
            full, q_count, t_count, combo)

        with open(f'{file_loc}output/{info_file_name}--{it}.txt',
                  'w') as file_object:
            for Pauli in comp_branch0:
                file_object.write(Pauli.return_Pauli_str()[0] +
                                  Pauli.return_Pauli_str()[new_q_count + 1:] +
                                  '\n')
            file_object.write('\n')
            file_object.write('Type of each of the measurements: ' + str(tps) +
                              '\n')
            file_object.write('\n')
            file_object.write('Branching at gadget measurements: ' + str(
                [i
                 for i, val in enumerate(tps) if val == 'q' and i < t_count]) +
                              '\n')
            file_object.write('\n')
            file_object.write(
                'Reassignment of qubits for branchers (i.e., Sdg gates): ' +
                str(tracker) + '\n')
            file_object.write('\n')
            file_object.write(
                'Identifying PIPC: ' +
                str([i for i, val in enumerate(tps) if val == 'q']) + '\n')
            file_object.write('\n')
            file_object.write('Classical tracking of sign flipping: ' +
                              str(sgns) + '\n')
            file_object.write('\n')
            file_object.write('Classical correlations of outcomes: ' +
                              str(correlats) + '\n')
            file_object.write('\n')
            file_object.write('Pseudo-outcomes: ' + str(pseudoutcomes) + '\n')

        it += 1


def naive_virtual_qubit_simulator(file_loc,
                                  input_file_name,
                                  clifford_file_name,
                                  result_file,
                                  virtual_qubits=1,
                                  precision=0.1,
                                  confidence_level=0.99):
    """This function estimates the probability that a given qubit of the input
    circuit yielding 1, when measured. We use the normal form theorem combined
    with a certain number of virtual qubits to produce the desired estimate to
    some `precision`.
    This is done in a somewhat naive way (there are ways of constructing better
    circuits), but the goal here is merely to demonstrate that the procedure is
    working correctly and that the correct estimate is retrieved using this
    procedure.

    Args:
        file_loc (str): location of the input circuit files.
        input_file_name (str): name of the file with the input circuit.
        clifford_file_name (str): name used to save the file with the
        gadgetized adaptive Clifford circuit created from the input circuit.
        result_file (str): name of the file that will store the desired
        estimate.
        virtual_qubits (int, optional): number of virtual qubits to be used.
        Defaults to 1.
        precision (float, optional): desired precision. Defaults to 0.1.
        confidence_level (float, optional): desired confidence level for the
        Hoeffding-bound sample count. Defaults to 0.99.
    """
    # STEP 1: Creating the folder for the output files.
    if not os.path.exists(f'{file_loc}output'):
        os.mkdir(f'{file_loc}output')

    # STEP 2:  Determine the required number of samples.
    weights_1vq = [1 / 2, (1 - np.sqrt(2)) / 2, 1 / np.sqrt(2)]
    probs_1vq = [np.sqrt(2) / 4, (2 - np.sqrt(2)) / 4, 1 / 2]
    l1_norm = np.sqrt(2)**virtual_qubits
    l1_squared = 2**virtual_qubits
    fail_prob = 1 - confidence_level
    k_const = np.log(2 / fail_prob) / 2  # from Hoeffding's inequality
    nr_samples = int(np.ceil(k_const * l1_squared / precision**2))

    combos = []
    for _ in range(nr_samples):
        aux = []
        for _ in range(virtual_qubits):
            aux.append(np.random.choice(np.arange(0, 3), p=probs_1vq))
        combos.append(tuple(aux))

    different_PBCs = Counter(combos)

    rand_variable = 0
    for stab, shots in different_PBCs.items():
        # STEP 3: Creates the gadgetized (i.e. adaptive) Clifford circuit which
        # is equivalent to the original universal quantum circuit.
        qc = inp.QuCirc(file_loc, input_file_name)
        clifford, q_count, t_count, nr_mmts = qc.msi_circuit(
            file_loc, clifford_file_name)
        # q_count -> nr. of qubits of the original quantum circuit
        # t_count -> nr. of T-gates of the original quantum circuit
        # nr_mmts -> nr. of measurements of the (new) gadgetized circuit

        full = full_zero_branch(clifford, q_count, t_count, nr_mmts)
        comp_branch0, tps, sgns, correlats, pseudoutcomes, tracker, new_q_count = remove_virtual_qubits(
            full, q_count, t_count, stab)

        branchers = [
            i for i, val in enumerate(tps) if val == 'q' and i < t_count
        ]
        qu_mmts = [i for i, val in enumerate(tps) if val == 'q']

        # STEP 4: Constructing the PBC-compiled quantum circuits. In this step
        # the most important thing is to guarantee that we go through a valid
        # branch of the tree. Note that we do NOT change the outcomes of the
        # classical registers (as we do not know what they are until we run the
        # actual computation), but we make sure that whatever values the
        # classical register takes, it is correctly used to make the correct
        # choice of the following branch.
        effective_t_count = t_count - (new_q_count - q_count)
        counter = len(comp_branch0) - 1
        nr_branchers = len(branchers)
        if len(comp_branch0) == 0:
            qu_outcomes = []
            print(f"PBC {stab} is fully classical!")
            print("Its outcome will be determined fully classically!")

        else:
            qr = qiskit.QuantumRegister(effective_t_count, 'qr')
            cr = qiskit.ClassicalRegister(len(comp_branch0), 'cr')

            if effective_t_count == 1:
                qc = qiskit.QuantumCircuit(qr, cr)
                qc.h(qr[0])
                qc.t(qr[0])

                Pauli = comp_branch0[0]
                if (Pauli.bin_vec[new_q_count] == 0
                        and Pauli.bin_vec[-2] == 1):
                    qc.measure(qr, cr)

                elif (Pauli.bin_vec[new_q_count] == 1
                      and Pauli.bin_vec[-2] == 0):
                    qc.h(qr[0])
                    qc.measure(qr, cr)

                elif (Pauli.bin_vec[new_q_count] == 1
                      and Pauli.bin_vec[-2] == 1):
                    qc.h(qr[0])
                    qc.s(qr[0])
                    qc.h(qr[0])
                    qc.measure(qr, cr)

            elif effective_t_count > 1:
                qaux = qiskit.QuantumRegister(1, 'qaux')
                qc = qiskit.QuantumCircuit(qr, qaux, cr)

                for i in range(effective_t_count):
                    qc.h(qr[i])
                    qc.t(qr[i])

                for p in range(len(comp_branch0)):
                    Pauli = comp_branch0[p]

                    qc.h(qaux)
                    for i in range(new_q_count, int(
                        (Pauli.nr_columns - 1) / 2)):
                        j = i - new_q_count
                        if (Pauli.bin_vec[i] == 0 and Pauli.bin_vec[i + int(
                            (Pauli.nr_columns - 1) / 2)] == 1):
                            qc.cz(qaux, qr[j])

                        elif (Pauli.bin_vec[i] == 1 and Pauli.bin_vec[i + int(
                            (Pauli.nr_columns - 1) / 2)] == 0):
                            qc.cx(qaux, qr[j])

                        elif (Pauli.bin_vec[i] == 1 and Pauli.bin_vec[i + int(
                            (Pauli.nr_columns - 1) / 2)] == 1):
                            qc.sdg(qr[j])
                            qc.cx(qaux, qr[j])
                            qc.s(qr[j])
                    qc.h(qaux)
                    qc.measure(qaux, cr[counter])

                    if p < nr_branchers:
                        # We are in the branching part of the circuit. We must
                        # make sure we run through the correct branch, with the
                        # correctly placed S† gates in place. Hence, we need to
                        # read whether there exist sign interdependencies.
                        signal = Pauli.bin_vec[-1]
                        for k in sgns[p]:
                            with qc.if_test(
                                (cr[len(comp_branch0) - 1 - k],
                                 (comp_branch0[qu_mmts.index(k)].bin_vec[-1] +
                                  1) % 2)):
                                signal = (signal + 1) % 2

                        with qc.if_test((cr[counter], (signal + 1) % 2)):
                            qc.sdg(qr[tracker[qu_mmts[p]]])

                    qc.reset(qaux)
                    counter -= 1

            # STEP 5: Running the circuit in the desired backend:
            #sim = AerSimulator(method='statevector', max_parallel_threads=1)
            sim = AerSimulator(method='statevector')
            result = sim.run(qiskit.transpile(qc, sim), shots=shots).result()
            counts = result.get_counts()

            qu_outcomes = []
            for key, val in counts.items():
                qu_outcomes += [key] * val

        # STEP 6: Result reinterpretation for the proper probab. distribution
        true_dist = [pseudoutcomes.copy() for _ in range(shots)]
        for s in range(shots):
            for i in range(nr_mmts):
                typ = tps[i]
                sgn = sgns[i]
                cor = correlats[i]
                if typ == "a":
                    if i >= t_count:
                        true_dist[s][i] = random.randint(0, 1)

                elif typ == "q":
                    true_dist[s][i] = (
                        int(qu_outcomes[s][qu_mmts.index(i)]) +
                        comp_branch0[qu_mmts.index(i)].bin_vec[-1]) % 2

                elif typ == "d1" or typ == "d2":
                    if cor != []:
                        for j in cor[:-1]:
                            true_dist[s][i] += true_dist[s][j]
                        if cor[-1] == "-":
                            true_dist[s][i] += 1
                        true_dist[s][i] = true_dist[s][i] % 2

                for j in sgn:
                    true_dist[s][i] += true_dist[s][j]
                true_dist[s][i] = true_dist[s][i] % 2

        weight = 1
        for t in stab:
            weight = weight * weights_1vq[t]
        signal = weight / abs(weight)

        for res in true_dist:
            outcome = signal * ((-1)**res[-1]) * l1_norm
            rand_variable += (1 - outcome) / (2 * nr_samples)

    with open(f'{file_loc}output/{result_file}.txt', 'w') as file_object:
        file_object.write(
            f"The probability is {rand_variable} +/- {precision}, with a confidence level of {confidence_level * 100}%.\n"
        )
