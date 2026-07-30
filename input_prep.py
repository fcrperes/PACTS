class Error(Exception):
    """Base class for other exceptions"""
    pass


class RegisterInputError(Error):
    """Exception raised for errors associated with the registers of the input
    quantum circuit.
    """

    def __init__(
        self,
        message="Invalid circuit form: registers are not correctly specified!"
    ):
        self.message = message
        super().__init__(self.message)


class MeasurementInputError(Error):
    """Exception raised when there are no measurements on the input quantum
    circuit.
    """

    def __init__(self,
                 message="Invalid circuit form: circuit has no measurements!"):
        self.message = message
        super().__init__(self.message)


class StabilizerInputError(Error):
    """Exception raised when the input quantum circuit is a stabilizer
    circuit.
    """

    def __init__(
            self,
            message="Compilation is unnecessary: input is a stabilizer circuit."
    ):
        self.message = message
        super().__init__(self.message)


class NonCliffordTGenInputError(Error):
    """Exception raised when the input circuit has operations other than the
    ones from the {H, S, CNOT, T} generating set.
    """

    def __init__(
            self,
            message="Invalid circuit form: illegal operations are being used!"
    ):
        self.message = message
        super().__init__(self.message)


class QuCirc():
    """ This class represents a given Clifford+T input quantum circuit"""

    def __init__(self, input_file_loc, input_file_name):
        """Initialize the attributes that are used to describe the input quantum
        circuit

        Args:
            input_file_loc (str): Location of the .qasm input file
            input_file_name (str): Name of the .qasm input file
        """
        # defining the two attributes of the input quantum circuits:
        self.input_file_loc = input_file_loc
        if input_file_name[-5:] == '.qasm':
            self.input_file_name = input_file_name
        else:
            self.input_file_name = input_file_name + '.qasm'

        # defining other gen. variables useful within the methods of this class:
        self.qasm_header = ['OPENQASM ', 'include ']
        self.qasm_registers = ['qreg ', 'creg ']
        self.qasm_allowed_operations = [
            'x ', 'z ', 'h ', 's ', 'sdg ', 'cx ', 'cz ', 't ', 'measure ',
            'barrier '
        ]
        self.qasm_all = self.qasm_header + self.qasm_registers + self.qasm_allowed_operations

    def get_circuit_data(self):
        """This method is applied to an instantiated quantum circuit, edits it
        and returns a list with the relevant information.

        Returns:
            list (str): List with all the relevant information about the quantum
            circuit; the elements of this list are the non-blank, non-commented
            lines from the input .qasm file.
        """
        with open(
                f'{self.input_file_loc}{self.input_file_name}') as file_object:
            file_lines = list(file_object)

        return [l for l in file_lines if l != '\n' and not l.startswith('//')]

    def verify_input_circuit(self):
        """This method evaluates whether or not the quantum circuit instance
        encodes a valid input quantum circuit and edits it further, changing the
        names of the quantum and classical registers respectively to `q_stab` &
        `c_stab`.

        Raises:
            NonCliffordTGenInputError: Exception raised if disallowed operations
            are being used
            StabilizerInputError: Exception raised if the input quantum circuit
            is a stabilizer circuit
            MeasurementInputError: Exception raised if there are no measurements
            in the input quantum circuit
            RegisterInputError: Exception raised if the nr of quantum registers
            differs from 1 or the total nr of registers differs from 2
        
        Returns:
            tuple (list[str], int, int, int): If the quantum circuit intance is
            valid, we get an updated/edited list describing it, and 3 properties
            of that circuit: 1st the number of qubits, 2nd its T-count, and 3rd
            the number of measurements.
        """
        t_count = 0  # number of T gates in the quantum circuit (must be >0)
        mmt_count = 0  # number of measurements in the circuit (must be >0)
        nr_qregs = 0  # number of quantum registers in the circuit (must be =1)
        circuit_header, circuit_registers, circuit_operations = [[], [], []]

        circuit_list = self.get_circuit_data()

        for line in circuit_list:
            for element in self.qasm_all:
                if line.startswith(element):
                    if element in self.qasm_header:
                        circuit_header.append(line)
                    elif element in self.qasm_registers:
                        circuit_registers.append(line)
                    elif element in self.qasm_allowed_operations:
                        circuit_operations.append(line)

            if line.startswith('t '):
                t_count += 1
            elif line.startswith('measure '):
                mmt_count += 1
            elif line.startswith('qreg '):
                x = line.partition('[')
                y = x[2].partition(']')
                q_count = int(y[0])
                nr_qregs += 1

        if not (len(circuit_list) == len(circuit_header) +
                len(circuit_registers) + len(circuit_operations)):
            raise NonCliffordTGenInputError()
        elif not t_count > 0:
            raise StabilizerInputError()
        elif not mmt_count > 0:
            raise MeasurementInputError()
        elif not (len(circuit_registers) == 2 and nr_qregs == 1):
            raise RegisterInputError()

        # the next bit of code guarantees that the quantum register is named
        # `q_stab` and that the classical register is named `c_stab`:
        it = 0
        for line in circuit_list:
            if line.startswith('qreg '):
                x = line.partition('[')
                y = x[0].partition(' ')
                qreg_name = y[2]
                it += 1
            elif line.startswith('creg '):
                x = line.partition('[')
                y = x[0].partition(' ')
                creg_name = y[2]
                it += 1
            else:
                if it == 2:
                    break

        it = 0
        while it < len(circuit_list):
            circuit_list[it] = circuit_list[it].replace(
                f'{qreg_name}[', 'q_stab[')
            circuit_list[it] = circuit_list[it].replace(
                f'{creg_name}[', 'c_stab[')
            it += 1

        return circuit_list, q_count, t_count, mmt_count

    def msi_circuit(self, msi_clifford_file_loc, msi_clifford_file_name):
        """This method takes in a certain quantum circuit instance and creates
        a new .qasm file with the gadgetized (i.e., adaptive) Clifford quantum
        circuit to be used in the compilation. An important observation is that
        this method creates a second qubit register (which holds the auxiliary
        qubits). We must remember that these need to be initialized in the magic
        state |A⟩, even though this method does not do this because it is not
        convenient for the next stages of the computation.

        Args:
            msi_clifford_file_loc (str): Location for the output gadgetized
            Clifford .qasm file
            msi_clifford_file_name (str): Name for the output gadgetized
            Clifford .qasm file

        Returns:
            tuple (list[str], int, int, int): Tuple with (1) the list of
            operations associated with the adaptive Clifford circuit with magic
            state injection and 4 properties of the original circuit: (2) the
            number of qubits, (3) its T-count, and (4) the number of
            measurements of the (new) gadgetized quantum circuit.
        """
        circuit_list, q_count, t_count, mmt_count = self.verify_input_circuit()

        new_circuit_list = []
        inserted_registers = False
        k = 0
        for line in circuit_list:
            if not inserted_registers and line.startswith('creg '):
                new_circuit_list.append(f'qreg q_magic[{t_count}];\n')
                new_circuit_list.append(line)
                new_circuit_list.append(f'creg c_magic[{t_count}];\n')
                inserted_registers = True
                continue

            if line.startswith('t '):
                x = line.partition('[')
                y = x[2].partition(']')
                i = y[0]  # qubit the T gate is being applied to
                cx_line = line.replace('t ', 'cx ')
                cx_line = cx_line.replace(';', f',q_magic[{k}];')
                new_circuit_list.append(cx_line)
                new_circuit_list.append(
                    f'measure q_magic[{k}] -> c_magic[{k}];\n')
                new_circuit_list.append(
                    f'if(c_magic[{k}]==1) s q_stab[{i}];\n')
                k += 1
                continue

            new_circuit_list.append(line)

        if not msi_clifford_file_name[-5:] == '.qasm':
            msi_clifford_file_name += '.qasm'

        with open(f'{msi_clifford_file_loc}{msi_clifford_file_name}',
                  'w') as file_object:
            for line in new_circuit_list:
                file_object.write(line)

        nr_mmts = mmt_count + t_count
        return new_circuit_list, q_count, t_count, nr_mmts
