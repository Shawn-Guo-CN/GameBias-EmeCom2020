# This script is to run 9 instances of symbolic refer and recon game to obtain the corresponding topo_sim log files.

for s in 1234 2233 3344 12345 828 1123 1991 1992 2018
do
    # run reconstructon game
    python ../run_recon_game.py --random_seed $s --topo_path ../log/recon_topsim/$s.txt --cuda_no 0

    # run referential game
    python ../run_refer_game.py --random_seed $s --topo_path ../log/refer_topsim/$s.txt --cuda_no 0
done
