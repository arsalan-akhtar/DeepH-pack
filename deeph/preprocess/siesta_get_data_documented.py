import os
import numpy as np
from numpy.core.fromnumeric import sort
import scipy as sp
import h5py
import json
from scipy.io import FortranFile

# Transfer SIESTA output to DeepH format
# DeepH-pack: https://deeph-pack.readthedocs.io/en/latest/index.html
# Coded by ZC Tang @ Tsinghua Univ. e-mail: az_txycha@126.com

def siesta_parse(input_path, output_path):
    """
    Function to parse SIESTA output files and convert them into a format suitable for DeepH input.

    Parameters:
    input_path (str): Path to the directory containing SIESTA output files.
    output_path (str): Path to the directory where the parsed data will be saved.

    Outputs:
    - lattice vectors saved in 'lat.dat'
    - atomic elements saved in 'element.dat'
    - atomic site positions saved in 'site_positions.dat'
    - Hamiltonian matrix and overlap matrix saved in HDF5 format (hamiltonians.h5 and overlaps.h5)
    """
    
    # Get absolute paths for input and output directories
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    # Ensure the output directory exists, if not, create it
    os.makedirs(output_path, exist_ok=True)
    
    # Search for the system name in the input directory by looking for files with extension ".ORB_INDX"
    f_list = os.listdir(input_path)
    for f_name in f_list:
        if f_name[::-1][0:9] == 'XDNI_BRO.':  # Reverse string comparison to match ".ORB_INDX"
            system_name = f_name[:-9]  # Extract system name by removing extension

    # Read structural information from the STRUCT_OUT file and extract lattice vectors and atom positions
    with open(f'{input_path}/{system_name}.STRUCT_OUT', 'r') as struct:  # Structure info from SIESTA standard output
        lattice = np.empty((3, 3))  # Initialize lattice array
        for i in range(3):  # Read the lattice vectors from the first 3 lines
            line = struct.readline()
            linesplit = line.split()
            lattice[i, :] = linesplit[:]
        np.savetxt(f'{output_path}/lat.dat', np.transpose(lattice), fmt='%.18e')  # Save lattice vectors in 'lat.dat'

        # Read the number of atoms from the next line
        line = struct.readline()
        linesplit = line.split()
        num_atoms = int(linesplit[0])

        # Initialize array for atom coordinates and elements
        atom_coord = np.empty((num_atoms, 4))
        for i in range(num_atoms):
            line = struct.readline()
            linesplit = line.split()
            atom_coord[i, :] = linesplit[1:]  # Read atomic information
        np.savetxt(f'{output_path}/element.dat', atom_coord[:, 0], fmt='%d')  # Save atomic elements in 'element.dat'


    # Convert atomic coordinates from the .XV file (in Bohr) to Angstroms and save the positions
    atom_coord_cart = np.genfromtxt(f'{input_path}/{system_name}.XV', skip_header=4)
    atom_coord_cart = atom_coord_cart[:, 2:5] * 0.529177249  # Convert from Bohr to Angstroms
    np.savetxt(f'{output_path}/site_positions.dat', np.transpose(atom_coord_cart))  # Save positions in 'site_positions.dat'

    # Read orbital index data from ORB_INDX file
    orb_indx = np.genfromtxt(f'{input_path}/{system_name}.ORB_INDX', skip_header=3, skip_footer=17)

    # Orbital index format:
    # 0: orbital id, 1: atom id, 2: atom type, 3: element symbol
    # 4: orbital id within atom, 5: n (principal quantum number), 6: l (orbital angular momentum)
    # 7: m (magnetic quantum number)

    # Ensure that columns 12 to 15 of orb_indx are properly filled
    orb_indx[:, 12:15] = orb_indx[:, 12:15]

    # Write out R_list.dat file, which contains unique R vectors
    with open('{}/R_list.dat'.format(output_path), 'w') as R_list_f:
        R_prev = np.empty(3)
        for i in range(len(orb_indx)):
            R = orb_indx[i, 12:15]  # Extract the R vector
            if (R != R_prev).any():  # If the current R differs from the previous R
                R_prev = R  # Update the previous R
                R_list_f.write('{} {} {}\n'.format(int(R[0]), int(R[1]), int(R[2])))  # Write R to file

    # Generate ia2Riua, which maps atoms to R vectors
    ia2Riua = np.empty((0, 4))  # Initialize array to store DeepH key mapping
    ia = 0  # Track current atom ID
    for i in range(len(orb_indx)):
        if orb_indx[i][1] != ia:  # When atom ID changes
            ia = orb_indx[i][1]
            Riua = np.empty((1, 4))  # Initialize temporary storage for atom-to-R mapping
            Riua[0, 0:3] = orb_indx[i][12:15]  # Store R vector
            iuo = int(orb_indx[i][15])  # Extract orbital information
            iua = int(orb_indx[iuo - 1, 1])  # Map orbital to atom ID
            Riua[0, 3] = int(iua)  # Store the atom ID
            ia2Riua = np.append(ia2Riua, Riua)  # Append to ia2Riua
    ia2Riua = ia2Riua.reshape(int(len(ia2Riua) / 4), 4)  # Reshape into 4-column array

    # Write out key system information in info.json
    info = {
        'nsites': num_atoms,  # Number of atomic sites
        'isorthogonal': False,  # Not orthogonal
        'isspinful': False,  # Not spinful
        'norbits': len(orb_indx)  # Total number of orbitals
    }
    with open('{}/info.json'.format(output_path), 'w') as info_f:
        json.dump(info, info_f)  # Write info to JSON file

    # Compute reciprocal lattice vectors and save to rlat.dat
    a1 = lattice[0, :]  # First lattice vector
    a2 = lattice[1, :]  # Second lattice vector
    a3 = lattice[2, :]  # Third lattice vector
    b1 = 2 * np.pi * np.cross(a2, a3) / (np.dot(a1, np.cross(a2, a3)))  # Reciprocal lattice vector b1
    b2 = 2 * np.pi * np.cross(a3, a1) / (np.dot(a2, np.cross(a3, a1)))  # Reciprocal lattice vector b2
    b3 = 2 * np.pi * np.cross(a1, a2) / (np.dot(a3, np.cross(a1, a2)))  # Reciprocal lattice vector b3
    rlattice = np.array([b1, b2, b3])  # Combine into reciprocal lattice matrix
    np.savetxt('{}/rlat.dat'.format(output_path), np.transpose(rlattice), fmt='%.18e')  # Save reciprocal lattice

    # Cope with orbital type information and write orbital_types.dat
    i = 0
    with open('{}/orbital_types.dat'.format(output_path), 'w') as orb_type_f:
        atom_current = 0
        while True:  # Loop over atoms in unit cell
            if atom_current != orb_indx[i, 1]:  # When atom ID changes
                if atom_current != 0:  # After the first atom
                    # Write the count of each orbital type for the previous atom
                    for j in range(4):
                        for _ in range(int(atom_orb_cnt[j] / (2 * j + 1))):
                            orb_type_f.write('{}  '.format(j))
                    orb_type_f.write('\n')

                atom_current = int(orb_indx[i, 1])  # Update current atom ID
                atom_orb_cnt = np.array([0, 0, 0, 0])  # Reset count of s, p, d, f orbitals

            l = int(orb_indx[i, 6])  # Extract angular momentum quantum number (l)
            atom_orb_cnt[l] += 1  # Increment count for this orbital type (s, p, d, f)

            i += 1  # Move to the next orbital
            if i > len(orb_indx) - 1:  # If at the end of the orbital list
                # Write orbital counts for the last atom
                for j in range(4):
                    for _ in range(int(atom_orb_cnt[j] / (2 * j + 1))):
                        orb_type_f.write('{}  '.format(j))
                orb_type_f.write('\n')
                break

            if orb_indx[i, 0] != orb_indx[i, 15]:  # If new atom starts
                # Write orbital counts for the current atom
                for j in range(4):
                    for _ in range(int(atom_orb_cnt[j] / (2 * j + 1))):
                        orb_type_f.write('{}  '.format(j))
                orb_type_f.write('\n')
                break


    # Yields key for *.h5 file
    orb2deephorb = np.zeros((len(orb_indx), 5))
    atom_current = 1
    orb_atom_current = np.empty((0))  # Stores orbitals' id in SIESTA, n, l, m, and z, will be reshaped into orb*5
    t = 0 
    
    for i in range(len(orb_indx)):  
        orb_atom_current = np.append(orb_atom_current, i)
        orb_atom_current = np.append(orb_atom_current, orb_indx[i, 5:9])  # Append orbital index data

        # If this is not the last atom and atom ID changes, process the orbitals
        if i != len(orb_indx) - 1:
            if orb_indx[i + 1, 1] != atom_current:
                orb_atom_current = np.reshape(orb_atom_current, ((int(len(orb_atom_current) / 5), 5)))

                # Handle orbital type (p, d, f) based on quantum number l, and rearrange m
                for j in range(len(orb_atom_current)):
                    if orb_atom_current[j, 2] == 1:  # p-orbitals
                        if orb_atom_current[j, 3] == -1:
                            orb_atom_current[j, 3] = 0
                        elif orb_atom_current[j, 3] == 0:
                            orb_atom_current[j, 3] = 1
                        elif orb_atom_current[j, 3] == 1:
                            orb_atom_current[j, 3] = -1

                    if orb_atom_current[j, 2] == 2:  # d-orbitals
                        if orb_atom_current[j, 3] == -2:
                            orb_atom_current[j, 3] = 0
                        elif orb_atom_current[j, 3] == -1:
                            orb_atom_current[j, 3] = 2
                        elif orb_atom_current[j, 3] == 0:
                            orb_atom_current[j, 3] = -2
                        elif orb_atom_current[j, 3] == 1:
                            orb_atom_current[j, 3] = 1
                        elif orb_atom_current[j, 3] == 2:
                            orb_atom_current[j, 3] = -1

                    if orb_atom_current[j, 2] == 3:  # f-orbitals
                        if orb_atom_current[j, 3] == -3:
                            orb_atom_current[j, 3] = 0
                        elif orb_atom_current[j, 3] == -2:
                            orb_atom_current[j, 3] = 1
                        elif orb_atom_current[j, 3] == -1:
                            orb_atom_current[j, 3] = -1
                        elif orb_atom_current[j, 3] == 0:
                            orb_atom_current[j, 3] = 2
                        elif orb_atom_current[j, 3] == 1:
                            orb_atom_current[j, 3] = -2
                        elif orb_atom_current[j, 3] == 2:
                            orb_atom_current[j, 3] = 3
                        elif orb_atom_current[j, 3] == 3:
                            orb_atom_current[j, 3] = -3

                # Sort orbitals based on quantum numbers
                sort_index = np.zeros(len(orb_atom_current))
                for j in range(len(orb_atom_current)):
                    sort_index[j] = (orb_atom_current[j, 3] + 
                                     10 * orb_atom_current[j, 4] + 
                                     100 * orb_atom_current[j, 1] + 
                                     1000 * orb_atom_current[j, 2])
                
                orb_order = np.argsort(sort_index)
                tmpt = np.empty(len(orb_order))

                # Assign sorted orbitals back
                for j in range(len(orb_order)):
                    tmpt[orb_order[j]] = j
                orb_order = tmpt

                # Fill the orb2deephorb array with sorted data
                for j in range(len(orb_atom_current)):
                    orb2deephorb[t, 0:3] = np.round(orb_indx[t, 12:15])  # Take x, y, z components
                    orb2deephorb[t, 3] = orb_indx[t, 1]  # Atom ID
                    orb2deephorb[t, 4] = int(orb_order[j])  # Orbital index in sorted order
                    t += 1

                # Reset for next atom
                atom_current += 1
                orb_atom_current = np.empty((0))  # Reset orbital storage for next atom

        # Open Hamiltonian and overlap files (H and S)
    f_h = FortranFile(f'{input_path}/{system_name}.H', 'r')  # Read Hamiltonian binary data
    f_s = FortranFile(f'{input_path}/{system_name}.S', 'r')  # Read overlap matrix binary data

    # Read the dimensions of the Hamiltonian and overlap matrices
    ndims = int(f_h.read_ints())
    
    # Initialize lists to store Hamiltonian and overlap matrix elements
    listh = np.zeros((ndims, ndims))  # Hamiltonian list
    lists = np.zeros((ndims, ndims))  # Overlap list

    for i in range(ndims):
        for j in range(ndims):
            listh[i, j] = f_h.read_reals(np.float64)  # Read Hamiltonian elements
            lists[i, j] = f_s.read_reals(np.float64)  # Read overlap elements

    # Fill sparse Hamiltonian and overlap matrices from orbital indices and parsed data
    H_block_sparse = {}  # Dictionary to store sparse Hamiltonian data
    S_block_sparse = {}  # Dictionary to store sparse overlap data

    for i1 in range(ndims):
        for j1 in range(ndims):
            atom_1 = int(orb2deephorb[i1, 3])  # Get the atom ID for the first orbital
            for k in range(len(listh[i1, j1])):
                atom_2 = int(orb2deephorb[listh[i1, j1] - 1, 3])  # Get atom ID for second orbital
                m = orb_indx[listh[i1, j1] - 1, 7]  # Magnetic quantum number for the second orbital
                Rijk = orb2deephorb[listh[i1, j1] - 1, 0:3].astype(int)  # Extract R vector

                # Store Hamiltonian sparse data
                key_h = '[{}, {}, {}, {}, {}]'.format(Rijk[0], Rijk[1], Rijk[2], atom_1, atom_2)
                if key_h not in H_block_sparse:
                    H_block_sparse[key_h] = []
                H_block_sparse[key_h].append([int(orb2deephorb[i1, 4]), int(orb2deephorb[listh[i1, j1] - 1, 4]),
                                              listh[i1, j1] * ((-1) ** m)])

                # Store overlap sparse data
                key_s = '[{}, {}, {}, {}, {}]'.format(Rijk[0], Rijk[1], Rijk[2], atom_1, atom_2)
                if key_s not in S_block_sparse:
                    S_block_sparse[key_s] = []
                S_block_sparse[key_s].append([int(orb2deephorb[i1, 4]), int(orb2deephorb[listh[i1, j1] - 1, 4]),
                                              lists[i1, j1]])

    # Convert sparse Hamiltonian and overlap matrices into full matrices
    for Rijkab in H_block_sparse.keys():
        sparse_form = H_block_sparse[Rijkab]
        ia1 = int(Rijkab[1:-1].split(',')[3])  # Atom ID 1
        ia2 = int(Rijkab[1:-1].split(',')[4])  # Atom ID 2
        tmpt = np.zeros((int(atom2nu[ia1 - 1]), int(atom2nu[ia2 - 1])))  # Initialize full matrix block
        for i in range(len(sparse_form)):
            tmpt[int(sparse_form[i][0]), int(sparse_form[i][1])] = sparse_form[i][2] / 0.036749324533634074 / 2
        H_block_sparse[Rijkab] = tmpt  # Replace sparse data with full matrix data

    for Rijkab in S_block_sparse.keys():
        sparse_form = S_block_sparse[Rijkab]
        ia1 = int(Rijkab[1:-1].split(',')[3])  # Atom ID 1
        ia2 = int(Rijkab[1:-1].split(',')[4])  # Atom ID 2
        tmpt = np.zeros((int(atom2nu[ia1 - 1]), int(atom2nu[ia2 - 1])))  # Initialize full matrix block
        for i in range(len(sparse_form)):
            tmpt[int(sparse_form[i][0]), int(sparse_form[i][1])] = sparse_form[i][2]
        S_block_sparse[Rijkab] = tmpt  # Replace sparse data with full matrix data

    # Save the full Hamiltonian matrix in h5 format
    with h5py.File(f'{output_path}/hamiltonians.h5', 'w') as f:
        for Rijkab in H_block_sparse.keys():
            f[Rijkab] = H_block_sparse[Rijkab]

    # Save the full overlap matrix in h5 format
    with h5py.File(f'{output_path}/overlaps.h5', 'w') as f:
        for Rijkab in S_block_sparse.keys():
            f[Rijkab] = S_block_sparse[Rijkab]

    # Close any remaining open files or resources
    f_h.close()  # Close the Hamiltonian file
    f_s.close()  # Close the overlap file

    # Wrap up the function and ensure that all data has been saved properly
    print(f'Data successfully parsed and saved in {output_path}')
