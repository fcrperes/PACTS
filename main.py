import tree

# Select the location and name of the input sample file(s)
file_loc = '/Users/folder/'
input_file_name = 'circuit_file'

# Select the number of virtual qubits (use 0 if you are not interested in
# simulating virtual qubits.)
vq = 0
prec = 0.01  # precision for virtual qubit simulation
cl = 0.99  # confidence level for virtual qubit simulation

# For zero virtual qubits, if you want to use the calculated PBC tree to obtain
# samples from the output distribution of the input circuit, change run to True.
run = False

# For nonzero virtual qubits, if you want to use the calculated PBC tree to
# estimate the desired probability, then set est = True.
est = False

# The user can change the following names at will
clifford_file_name = 'MSI'
info_file_name = 'PBC_tree_info'
histo_file_name = 'Histogram'
res_file_name = 'Probability'

if vq == 0:
    # Get the PBC tree obtained from our Normal Form Theorem:
    tree.get_pbc_tree(file_loc, input_file_name, clifford_file_name,
                      info_file_name)
    if run:
        tree.naive_pbc_simulator(file_loc,
                                 input_file_name,
                                 clifford_file_name,
                                 histo_file_name,
                                 shots=10000)

elif vq > 0:
    # Get the PBC tree with vq virtual qubits from our Normal Form Theorem:
    tree.get_virtualization_info(file_loc,
                                 input_file_name,
                                 clifford_file_name,
                                 info_file_name,
                                 virtual_qubits=vq)
    if est:
        tree.naive_virtual_qubit_simulator(file_loc,
                                           input_file_name,
                                           clifford_file_name,
                                           res_file_name,
                                           virtual_qubits=vq,
                                           precision=prec,
                                           confidence_level=cl)

else:
    raise ValueError
