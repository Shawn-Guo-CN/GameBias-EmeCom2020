# This script is to run 9 instances of symbolic refer and recon game to obtain the task transfer performance.

for s in 1234 2233 3344 12345 828 1123 1991 1992 2018 2020
do
    # run reconstructon game
    python ../recon_to_ref.py --random_seed $s --generalisation_path ../log/recon_to_refer/$s.txt --cuda_no 3 --topo_path ../log/topo_recon.txt --training_log_path ../log/recon_train/$s.txt

    # run referential game
    python ../ref_to_recon.py --random_seed $s --generalisation_path ../log/refer_to_recon/$s.txt --cuda_no 3 --topo_path ../log/topo_refer.txt --training_log_path ../log/refer_train/$s.txt
done