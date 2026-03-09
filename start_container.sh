enroot create --name verl-ali /scratch/users/ntu/ruiqi003/sqsh_images/verl-ali.sqsh
enroot start \
  --root \
  --rw \
  --env HF_HOME=/hf_home \
  --mount /raid/pbs.$PBS_JOBID/hf_home/:/hf_home \
  --mount ./data:/workspace/data/ocr \
  verl-ali \
  /bin/bash