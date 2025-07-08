# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1.0e-5
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
variable atomjiggle equal 1.0e-8

# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3

# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 1.0e-10
variable maxiter equal 200
variable maxeval equal 1000
variable dmax equal 1.0e-2

# generate the box and atom positions using a diamond lattice

boundary	p p p

# **** for hcp ******
#lattice          custom 2.760 origin 0.0 0.0 0.0 orient x 1 0 0 orient y 0 1 0 orient z 0 0 1 a1 1.73205081 0.0 0.0 a2 0.0 1.0 0.0 a3 0.0 0.0 1.61521739130435 basis 0.0 0.0 0.0 basis 0.5 0.5 0.0 basis 0.33333333333333333333333333 0.0 0.5 basis 0.83333333333333333333333333 0.5 0.5
#region          box prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0
#create_box      1 box
#create_atoms    1 box
# **** end for hcp *****

# *** for bcc ****
lattice         bcc 3.1719118 #3.1773
#lattice         bcc 3.25
region		box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0
create_box	1 box
create_atoms	1 box
# *** end for bcc ***

# *** or read_data from file ***
#read_data      read_data.dat
# *** end read_data from file ***

# Need to set mass to something, just to satisfy LAMMPS
#mass 1 1.0e-20
mass 1 183.85

