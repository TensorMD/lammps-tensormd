echo "##############" >result
echo "### elastic###" >> result
echo "##############" >> result

echo "running elastic"
lmp -i in.elastic 1>runlog 2>err
tail -n33 runlog |head -n23 >> result

rm runlog err
grep C11 result
grep C12 result
grep C44 result
