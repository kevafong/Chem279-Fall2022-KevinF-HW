# Numerical Algorithms for Quantum Mechanical Systems Modeling

This repository contains implementation for the modeling of moleculars using the Minimal basis set (STO-3G) of atoms, and the caluclation of energy by Hartree Fock.

## Repository Struture

- Chem279-Fall2022-KevinF-HW
	- Subdirectories:
		- README.md: this file, documentation on the conceptual introduction to the modeling structures used for Computational Quantum Systems.
		- HW1: Potential Energy Functions, Forces, and Local Optimization
		- HW2: Analytical and Numerical Integration for Overlap Integrals
		- HW3: Extended Huckel model
   		- HW4: SCF Energy implementation
    		- HW5: Analytical Gradient of SCF Energy
    		- project: Vibrational Analysis of Huckel Model


## Final Assignment:Vibrational Analysis

The final deliverable for this repository is a adaptation to perform vibrational analysis to obtain the vibrational frequencies and the normal modes of a molecule at its optimized geometry.
This is achieved by evaluating the hessian matrix by finite differences, and transforming to mass-weighted coordinates and diagonilized to determine eigenvalues.
The resulting vibrational frequencies can be comparedagainst experimental infrared results. 
