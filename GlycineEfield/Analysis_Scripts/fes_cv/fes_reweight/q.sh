#!/bin/bash -l

list1='s05 d05'

for j in $list1; do
	cd $j
	test $? -ne 0 && exit 1
	rm fes* slurm-* *.o2* *.e2*
	sed -i s/"skiprows 500000"/"skiprows 0"/g q.pbs
	sed -i s/"blocks 5"/"blocks 3"/g q.pbs
	#sed -i s/"normal3"/"normal2"/g q.pbs
	#sed -i s/"nodes=node38:ppn=24"/"nodes=1:ppn=10"/g q.pbs
	qsub  q.pbs
	cd ..
done
