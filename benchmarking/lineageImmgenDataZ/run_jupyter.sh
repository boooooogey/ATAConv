#!/bin/bash

#SBATCH --job-name=tiSFM
#SBATCH -p benos
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task
##SBATCH --output tiSFM.stdout

# GET TUNNELING INFO====

XDG_RUNTIME_DIR=""    

ipnport=$(shuf -i8000-9999 -n1)    
ipnip=$(hostname -i)    
token=$(xxd -l 32 -c 32 -p < /dev/random)    

# PRINT TUNNELING INSTRUCTIONS====

echo -e "    
Copy/Paste this in your local terminal to ssh tunnel with remote    
-----------------------------------------------------------------    

ssh -N -L $ipnport:$ipnip:$ipnport $USER@cluster.csb.pitt.edu    
-----------------------------------------------------------------    

Then open a browser on your local machine to the following address    
------------------------------------------------------------------    

http://localhost:$ipnport?token=$token    
------------------------------------------------------------------      
"    

jupyter lab --ServerApp.iopub_data_rate_limit=100000000000000  --port=$ipnport --ip=$ipnip --ServerApp.password='' --ServerApp.token="$token" --no-browser
