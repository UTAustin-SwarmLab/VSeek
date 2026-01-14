1.  Torch 2.7.1 + cu12.8 (546-731) + vllm 0.11.0 + triton 3.3.X - this is the closest setup I have gotten to work 
to test this you need to run scripts/tacc/run_vseek_job.slurm

Problems:
this does not install triton from source, I have not fully explored this thread
initial runs resulted in the same error as the 2.7.1 install where either ray/gpu was running out of memory due to eager execution.

2. Torch 2.8 + cu12.9 (968-1164)
- this setup has some nvcc linking issues which does not load torch on the leaf ray nodes (ray head node)
to test this you need to run scripts/tacc/run_vseek_job12_9.slurm

3. Torch 2.9 + cu128 + vllm 0.12.0  (733-893) 

Problems:
this does not install triton from source, I have not fully explored this thread
initial runs resulted in the same error as the 2.7.1 install where either ray/gpu was running out of memory due to eager execution.

4. Torch 2.7.1 + cu12.6 (546-731) + vllm 0.11.0 + triton 3.3.X - I dont remember what happened here


5. Torch 2.9 + cu128 + vllm 0.12.0 (pip install)  (SP's) 

Problems:
verl compatibility issue where it was not able to spawn vllm asyncserver


To run the slurm script

bash scripts/tacc/run_vseek_job.slurm

1) Dataset - 
2) Index
