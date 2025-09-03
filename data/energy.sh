cd ./CLN025

gmx mdrun -s nvt_0.tpr -rerun all_traj.xtc -rerun -e energy.edr
# gmx energy -f energy.edr -o energy.xvg