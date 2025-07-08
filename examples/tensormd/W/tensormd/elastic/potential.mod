# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
#pair_style	eam/fs
#pair_coeff * * WRe_YC1.eam.fs W

pair_style	                tensormd
pair_coeff	                * * ../W_m3_k64_n128_squareplus.npz W interp 0.001

# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no
dump            1 all custom 10 dump.traj id type x y z
