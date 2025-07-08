echo "running elastic"
/home/oyyc/lammps/build/lmp -i in.elastic 1>runlog 2>err
grep C11 runlog
grep C12 runlog
grep C44 runlog
