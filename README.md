# PACTS -- Pauli Adaptive Computation Tree Software

In this repository, we provide Python code that can be used to efficiently
specify the Pauli-based computation (PBC) tree associated with a given input
Clifford+$T$ quantum circuit. It uses the **Normal Form Theorem** formulated
in our recent preprint [].

The code uses different modules, each equipped with different classes, methods,
and functions all of which are suitably documented for readability and
user-friendliness.

In principle, users need only open and make changes to `main.py`.

Contact: fcrperes@onsager.ugr.es

-------------------------------------------------------------------------------


## System pre-requisites

This code was written using the following versions:
* python 3.14.4;
* numpy 2.4.4;
* matplotlib 3.10.8;
* qiskit 2.3.1;
* qiskit-aer 0.17.2.

For reproducibility, an environment with the essential python packages can be
created from the `environment.yml` file by means of the following code line in
the user's terminal: `conda env create -f environment.yml`.

-------------------------------------------------------------------------------


## Use instructions 

1. After ensuring you have a working environment will all the necessary
packages, make sure that all three Python modules (`main.py`, `input_prep.py`,
and `tree.py`) can be found in the same directory in your computer;

2. Copy the input (`.qasm`) toy files supplied in this repository into a folder
in your computer (or make your own files with suitable Clifford+$T$ quantum
circuits). The files output by the code will be saved to a folder named "output"
which will be found inside the folder where you have placed the input file(s);

3. In the `main.py` file, change the value assignments of the `file_loc` and
`input_file_name` variables to match the location and name of the input file(s)
in your computer. You can also change the output file names at will.

4. Adjust the parameters of `tree.naive_pbc_simulator()` or
`tree.naive_virtual_qubit_simulator` depending on the simulation you want to
run. For the small toy examples provided, a number of shots of 10^6 can be used
and the simulation is still performed in reasonable time;

5. Open a terminal window in the directory where you have placed the Python
modules and run the command: `python main.py`;

6. Check the output files at the location that you have selected.

-------------------------------------------------------------------------------


## Reading the output files

The most challenging part of using our code is probably reading the information
describing the PBC tree. The following is an example of an output file:

-- -- --
-ZZII
+ZIZI
-ZIIZ
+XXXX

Type of each of the measurements: ['a', 'q', 'q', 'q', 'a', 'a', 'a', 'd1', 'd1', 'q']

Branching at gadget measurements: [1, 2, 3]

Reassignment of qubits for branchers (i.e., Sdg gates): {1: 1, 2: 2, 3: 3}

Identifying PIPC: [1, 2, 3, 9]

Classical tracking of sign flipping: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

Classical correlations of outcomes: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

Pseudo-outcomes: [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
-- -- --

The first lines give the zero branch of the PBC tree, that is, the measurement
sequence that needs to be formed when all outputs are zero.

"Type of each of the measurements:" gives a list that indicates the type of
each Pauli operator along the full PBC branch. 'a' stands for Paulis that are
anti-commuting and whose outcome can, therefore, be set classically; 'q' stands
for Paulis that need to be measured in quantum hardware (these are the Paulis
that will comprise the PBC tree); 'd1' stands for Paulis that act trivially on
the magic register and whose outcome is therefore deterministically determined
based on their sign; for instance, in the example above, there are two 'd1'
measurements that deterministically return the outcome 1 (this is seen in 
"Pseudo-outcomes"); 'd2' stands for Pauli measurements that depend on previously
measured Paulis.

"Branching at gadget measurements:" stores which of the Pauli measurements cause
a branching in the PBC tree. Equivalently, it stores which Pauli measurements
give rise to (adaptively applied) Sdg gates. This is the information that
allows one to create the rest of the branches from the zero branch.

Our Normal Form Theorem allows the identification of qubits that might be
freely removed from the PBC. If that is the case, it is necessary to track the
correct qubit to which each Sdg gate needs to be applied to. This information is
tracked in "Reassignment of qubits for branchers (i.e., Sdg gates):". In the
simple case above, we see that there is no reasignment, so that the first
branching measurement (-ZZII) implies an adaptive Sdg gate is applied in qubit
1 if the measurement outcome is 1, and so on. We consider another example. For
instance, for an input circuit with 4 T gates, where, initially, the first three
gadget measurements are identified as causing branching, if the first magic
qubit is freely removable (but no others are), we will end up with a PBC on
only three qubits and with two branching measurements. Then,
"Branching at gadget measurements:" will read [1, 2] and the reassignment
dictionary will be {1: 0, 2: 1}.

"Identifying PIPC:" gives the full list of which Pauli operators create the
(quantum) PBC tree. This information can be read also from
"Type of each of the measurements:" by checking the position of 'q' types.
PIPC stands for Paulis that are Independent and Pairwise Commuting (following
the original terminology from https://github.com/fcrperes/CompHybPBC).

"Classical tracking of sign flipping:" and "Classical correlations of outcomes:"
track additional classical information about how some of the measurements
outcomes might need to be re-interpreted depending on classical cointoss
results or how a 'd2' type measurement might be related to some of the 'q'
type measurements.

"Pseudo-outcomes:" are used to store the outcome of 'd1' type measurements.
In this case, there are two such measurements whose outcomes are
deterministically 1. If there are no 'd1' type measurements, this is just
a list of all 0s.

PS: Note that Python indexing is being used so that the first measurement is
measurement 0, the first qubit is qubit 0, and so on...

-------------------------------------------------------------------------------


## Copyright

Copyright (C) 2026  Filipa C. R. Peres

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
