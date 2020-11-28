import os
import time

code_directory = '/home/slowikag/scratch/EGG_research'
directory = code_directory

slurm_logs = os.path.join(directory, "slurm_logs")
slurm_scripts = os.path.join(directory, "slurm_scripts")
tb_dir = os.path.join(directory, "runs")
model_dir = os.path.join(directory, "models")
acc_dir = os.path.join(directory, 'accs')

if not os.path.exists(slurm_logs):
    os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
    os.makedirs(slurm_scripts)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)

def _run_exp(batch, file_name='egg.zoo.dsprites_vae.train', job_time="10:00:00"):
    save_str = batch['save_str']
    #if batch['save_data']:
    #    batch['checkpoint_dir'] = os.path.join(model_dir, save_str)
    #    batch['acc_data_path'] = os.path.join(acc_dir, save_str)
    #    if not os.path.exists(batch['checkpoint_dir']):
    #        os.makedirs(batch['checkpoint_dir'])
    #    if not os.path.exists(batch['acc_data_path']):
    #        os.makedirs(batch['acc_data_path'])
    #if batch['tensorboard']:
    #    batch['tensorboard_dir'] = os.path.join(tb_dir, save_str)

    file_path = f'-m {file_name}'
    jobcommand = f'python {file_path}'
    #for key, value in batch.items():
    #    if key in ['save_data', 'save_str']:
    #        continue
    #    elif type(value) == bool:
    #        if value:
    #            jobcommand += f' --{key}'
    #    else:
    #        jobcommand += f' --{key} {value}'
    #print(jobcommand)

    slurmfile = os.path.join(slurm_scripts, save_str + '.sh')
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={save_str}\n")
        f.write(f'#SBATCH --output={os.path.join(slurm_logs, save_str + ".out")}\n')
        # f.write(f'#SBATCH --error={os.path.join(slurm_logs, save_str + ".err")}\n')
        f.write("module load cuda cudnn\n")
        f.write(f"cd {code_directory}\n")
        f.write(jobcommand + "\n")

    s = f"sbatch --gres=gpu:1 --mem=10G -c4 --time={job_time} --account=rrg-bengioy-ad"
    s += f' {slurmfile} &'
    os.system(s)
    time.sleep(1)

# test
job = {'n_epochs': 5, 'vocab_size': 5, 'batch_size': 128, 'lr': 1e-3}
job['save_str'] = f"TEST"
_run_exp(job, file_name='egg.zoo.dsprites_vae.train', job_time="3:00:00")
