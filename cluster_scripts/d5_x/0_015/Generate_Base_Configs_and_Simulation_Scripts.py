# ------------ This script generates the initial grid over which we will start our search ----------------

import os
import shutil
import pickle
cwd = os.getcwd()

# ------------ the fixed parameters: These are constant for all error rates -----------------------------

fixed_config = {"d": 5,
                "use_Y": False,
                "train_freq": 1,
                "batch_size": 32,
                "print_freq": 250,
                "rolling_average_length": 1000,
                "stopping_patience": 1000,
                "error_model": "X",
                "c_layers": [[64,3,2],[32,2,1],[32,2,1]],
                "ff_layers": [[512,0.2]],
                "max_timesteps": 1000000,
                "volume_depth": 5,
                "testing_length": 101,
                "buffer_size": 50000,
                "dueling": False,
                "masked_greedy": False,
                "static_decoder": True}

fixed_config_path = os.path.join(cwd, "fixed_config.p")
print(fixed_config_path)
pickle.dump(fixed_config, open(fixed_config_path, "wb" ) )

# ---------- The variable parameters grid --------------------------------------------------------------

p_phys = 0.015
success_threshold = 100000

learning_starts_list = [1000]
learning_rate_list = [0.0001, 0.00005, 0.00001]
exploration_fraction_list = [100000]
sim_time_per_ef = [10]
max_eps_list = [1.0]
target_network_update_freq_list = [2500]
gamma_list = [0.99]
final_eps_list = [0.02]
alpha_list = [0, 0.5, 0.7, 1.0]

config_counter = 1
for ls in learning_starts_list:
    for lr in learning_rate_list:
        for ef_count, ef in enumerate(exploration_fraction_list):
            for me in max_eps_list:
                for tnuf in target_network_update_freq_list:
                    for g in gamma_list:
                        for fe in final_eps_list:
                            for alpha in alpha_list:

                                variable_config_dict = {"p_phys": p_phys,
                                "p_meas": p_phys,
                                "success_threshold": success_threshold,
                                "learning_starts": ls,
                                "learning_rate": lr,
                                "exploration_fraction": ef,
                                "max_eps": me,
                                "target_network_update_freq": tnuf,
                                "gamma": g,
                                "alpha": alpha,
                                "final_eps": fe}

                                config_directory = os.path.join(cwd,"config_"+str(config_counter)+"/")
                                if not os.path.exists(config_directory):
                                    os.makedirs(config_directory)
                                else:
                                    shutil.rmtree(config_directory)           #removes all the subdirectories!
                                    os.makedirs(config_directory)

                                file_path = os.path.join(config_directory, "variable_config_"+str(config_counter) + ".p")
                                pickle.dump(variable_config_dict, open(file_path, "wb" ) )
                                            
                                # Now, write into the bash script exactly what we want to appear there
                                job_limit = str(sim_time_per_ef[ef_count])
                                job_name=str(p_phys)+"_"+str(config_counter)
                                output_file = os.path.join(cwd,"output_files/out_"+job_name+".out")
                                error_file = os.path.join(cwd,"output_files/err_"+job_name+".err")
                                python_script = os.path.join(cwd, "Single_Point_Continue_Training_Script.py")


                                f = open(config_directory + "/simulation_script.sh",'w')  
                                f.write('''#!/bin/bash
#SBATCH -p scavenger                         # scavenger division
#SBATCH -c 1                                 # Number of cores
#SBATCH --array=1-1  			             # How many jobs do you have                               
#SBATCH --job-name='''+job_name+'''          # Job name, will show up in squeue output
#SBATCH --mail-type=END
#SBATCH --mail-user=ql94@duke.edu	       # It will send you an email when the job is finished. 
#SBATCH --mem=10G                   # Memory per cpu in MB (see also --mem) 
#SBATCH --output=out.out         # File to which standard out will be written
#SBATCH --error=slurm.err           # File to which standard err will be written

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID


# ---------------------- JOB SCRIPT ---------------------------------------------

# ----------- Activate the environment  -----------------------------------------

#module load python/3.6.5
#module load tensorflow/1.14.0
#module load keras/1.14.0

# ------- run the script -----------------------

python '''+python_script+''' '''+str(config_counter)+'''

#----------- wait some time ------------------------------------

sleep 50''')
                                f.close()
                                config_counter += 1 
