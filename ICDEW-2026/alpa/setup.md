## FABRIC experiment setup
1. Setup the environment (https://portal.fabric-testbed.net)
    1. Use `jupyter-examples-rel1.9.0/start_here.ipynb`
2. Check for available GPUs
    1. Use `Listing_Available_Resources/v2025-04-15/sites_and_resources/list_selected_resources.ipynb`
3. Setup a slice on FABRIC
    1. Say 12 cores, 32 GB RAM, 500 GB storage, `Ubuntu 20.04` should work; do not chose Custom Ubuntu 20.04
    2. Attach desired number of GPUs to each VM (Note. Alpa can handle only powers of two)
    3. Single site or multisite
        - If single site, use any NIC with L2Bridge
        - If multisite, use shared NIC and L2STS; only some sites support PTP if that's needed
    
4. Connect to FABRIC slice using VSCode or SSH terminal

## Software installation
1. Clone the LaMB repo
    ```bash
    git clone https://github.com/raopr/LaMB.git
    ```

2. Install CUDA drivers on each VM
    ```bash
    $HOME/LaMB/alpa/install_cuda.sh
    sudo reboot
    ```

3. Login again to each VM

    1. If IPV6, add the below line to `$HOME/.bazelrc`
    ```bash    
    echo "startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true" >> $HOME/.bazelrc
    ```
    2. Run the below script to install additional packages to correctly build/install jaxlib and Alpa
    ```bash
    $HOME/LaMB/alpa/install_packages.sh
    ```

4. Setup IP addresses for VMs as in the below example
    ```bash
    ip link show
    sudo ifconfig enp8s0 up # Change interface name
    sudo ip addr add 192.168.1.1/24 dev enp8s0
    ip addr show
    ```

5. Setup NCCL network interface on each VM. Add the below line to `/etc/nccl.conf`.

    ```bash
    sudo sh -c 'echo "NCCL_SOCKET_IFNAME=enp8s0" >> /etc/nccl.conf' # Pick interface name using ifconfig 
    ```
6. To `SSH` between VMs, do the following.

    Add a VM's `~/.ssh/id_rsa.pub` to the other VM's `~/.ssh/authorized_keys` to allow SSH without keys. Do this for every VM.

    ```bash
    ssh-keygen -t rsa -P ""
    ```
7. Make sure `IP hostname` line is added to `/etc/hosts` on all VMs. Otherwise, `connect()` errors will appear during Alpa training. See example below.
    ```
    127.0.0.1 localhost
    192.168.1.1 eaab160a-9156-4f2a-8081-031a208235c7-vm1.novalocal vm1
    ```

8. Test if Ray uses the right interface/IP address; otherwise, pre-training will timeout due to connection error
    ```bash
    python3
    >>> import ray
    >>> ray._private.services.get_node_ip_address()
    ```
    If `192.168.1.*` is not output, do the following:
    ```bash
    vi $HOME/.local/lib/python3.8/site-packages/ray/_private/services.py
    ```
    Replace lines in the `try` block of function `node_ip_address_from_perspective(address: str)`
    ```python
    s.connect((ip_address, int(port)))
    node_ip_address = s.getsockname()[0]
    ```
    with
    ```python
    host_name = socket.getfqdn(socket.gethostname())
    node_ip_address = socket.gethostbyname(host_name)
    ```
9. If nodes are rebooted, then do step (4) and (7). Otherwise, Ray may not start.

## Starting Ray
1. To start Ray on 
    1. A single VM
        ```bash
        ray start --head
        ```  
    2. Multiple VMs

        a. On master VM
        ```bash
        ray start --head --node-ip-address 192.168.1.1
        ```
        b. On worker VM
        ```bash
        ray start --address='192.168.1.1:6379' --node-ip-address 192.168.1.2 
        ```
    3. Check `ray status`
    4. If you want to limit # of GPUs (for head/worker) to make a regular 2D mesh for Alpa training, then restrict the number of GPUs. Otherwise, Alpa will throw errors.
        ```bash
        ray start --head --node-ip-address 192.168.1.1 --num-gpus=2
        ```
## LLM Pretraining

1. Training a tokenizer on one VM and copy the directory to all VMs

    a. GPT2Config
    ```bash
    cd $HOME/LaMB/alpa/examples/gpt2
    mkdir ./wikipedia-gpt2-small
    python3 train_tokenizer.py
    python3 create_config.py
    cp -r ./wikipedia-gpt2-small $HOME/alpa/examples/gpt2/
    # scp -r ./wikipedia-gpt2-small 192.168.1.2:$HOME/alpa/examples/gpt2/
    ```
    b. LlamaConfig
    ```bash
    cd $HOME/LaMB/alpa/examples/llama
    mkdir ./wikipedia-llama
    python3 train_tokenizer.py
    python3 create_config.py
    cp -r ./wikipedia-llama $HOME/alpa/examples/gpt2/
    ```

    c. Download (from Hugging Face) an LLM model for training a new tokenizer 
    ```bash
    hf auth login    # Provide the Hugging Face token; say 'no' to git credential question
    hf download [modelID]   # e.g., meta-llama/Llama-2-13b-chat, meta-llama/Llama-2-7b-chat
    ```

2. Finally, we are ready to pre-train the model! (Use of `screen` is preferred.) 
    a. For different parallel approaches, copy some modified files to
    `alpa/examples/gpt2`. You can
     manually control the meshes and try different parallelization strategies (e.g., data, shard, pipeline shard) in Alpa. For `each VM`, do the following:

    ```bash
    cp ${HOME}/LaMB/alpa/examples/gpt2/run_clm_flax_v2.py ${HOME}/alpa/examples/gpt2/
    cp ${HOME}/LaMB/alpa/examples/gpt2/parallel_method.py ${HOME}/alpa/alpa/
    cp ${HOME}/LaMB/alpa/examples/gpt2/stage_construction.py ${HOME}/alpa/alpa/pipeline_parallel/
    cp ${HOME}/LaMB/alpa/examples/gpt2/stage_profiling.py ${HOME}/alpa/alpa/pipeline_parallel/
    ```

    ```bash
    cd $HOME/alpa/examples/gpt2
    python3 run_clm_flax_v2.py --output_dir="./wikipedia-gpt2-small"  --model_type="gpt2" --config_name="./wikipedia-gpt2-small"  --tokenizer_name="./wikipedia-gpt2-small" --dataset_name="wikimedia/wikipedia"  --dataset_config_name="20231101.ace" --do_train --do_eval  --block_size="64" --per_device_train_batch_size="32" --per_device_eval_batch_size="32" --num_micro_batches="4" --dtype="float16"  --learning_rate="1e-3" --warmup_steps="1000" --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01"  --overwrite_output_dir --num_train_epochs="20"  --logging_steps="100"  --save_steps="2500" --eval_steps="2500"
    ```
    b. To start with pretrained weights for a model, pass the argument `--model_name_or_path` when executing `run_clm_flax_v2.py`
    ```bash
    python3 run_clm_flax_v2.py --model_name_or_path="./wikipedia-gpt2-small" ...
    ```
    c. To reuse previously saved profile results, pass the argument `--cached_profile_result` when executing `run_clm_flax_v2.py`
    ```bash
    python3 run_clm_flax_v2.py --cached_profile_result="/home/ubuntu/alpa/examples/gpt2/profile-results-2026-05-26-14-33-17.pkl" ...
    ```
3. For debug mode, use `python3 -m pdb` or use VS Code Python debugger (use `launch.json` to pass arguments)



## LLM inference using Ray
1. Downgrade libraries on all VMs (create virtual environment)
    ```bash
    pip3 install -U "pyarrow<7.0.0"
    ```
2. Download the model on all VMs (including local pretrained models)
3. Run the inference script
    ```bash
    python3 alpa/ray/llm_inference.py <input file> <model>
    ```
    Tested HF models include `gpt2`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `google/gemma-2b-it`, etc. You can also use a model pre-trained using Alpa such as `wikipedia-gpt2`.

## Troubleshooting
1. Testing NCCL if it works correctly on multiple VMs
    First build the test code on each VM
    ```bash
    sudo apt install mpich -y; pip3 install mpi4py
    git clone https://github.com/NVIDIA/nccl-tests.git
    export CUDA_HOME=/usr/local/cuda-11.8/
    export NCCL_HOME=/home/ubuntu/.cupy/cuda_lib/11.x/nccl/2.15.1/
    export MPI_HOME=/usr/lib/x86_64-linux-gnu/mpich
    cd nccl-tests/
    make MPI=1
    ```
    Next, run with `mpirun` on one of the VMs
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.cupy/cuda_lib/11.x/nccl/2.15.1/lib/
    cd nccl-tests
    mpirun --hostfile ../hostfile -np 2  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
    # mpirun --hostfile ../hostfile -np 4  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 
    ```

    Change `-np` to equal the number of required MPI processes. See `cupy/hostfile` for an example hostfile.

2. Even if NCCL is properly configured and working, why is Alpa not working for multiple VMs with GPUs?

    The `get_node_ip_address()` function of Ray was returning the IP of another n/w interface. Hence, the `connect()` call in `device_mesh.py` was failing with the below information about the server address:

        `HostID: 1 Mesh_ID: None Server_addr: 10.30.6.192:20860`
        python3
        >>> import ray
        >>> ray._private.services.get_node_ip_address()
        '10.30.6.192'

    See ~/.local/lib/python3.8/site-packages/ray/_private for the API implementation.

    The solution was simple.
    In `/etc/hosts`, add the correct IP to use along with hostname.

    ```
    127.0.0.1 localhost
    192.168.1.1 7d94c095-3fec-41e2-b2d2-77db490fb2e1-vm1 vm1
    ```

    It worked immediately!

3. Pre-training throws `out of memory` errors or `replica not found` errors
    1. Reduce `--per_device_train_batch_size` and see if it works
    2. Reduce model size and see if it works
    2. Restart Ray on the cluster and see if it works

4. If model evaluation causes error during pre-training, remove `--do_eval` when running pre-training.

5. Ray does not start properly after nodes are rebooted.
    1. Check item (9) under `Software Installation`
       
## Acknowledgments

Thanks to Alpa, Ray, HuggingFace, NVIDIA, and FABRIC; we are grateful for their software/resources/infrastructure.      

## References
1. https://github.com/NVIDIA/nccl-tests
2. https://github.com/ray-project/ray
3. https://github.com/alpa-projects/alpa    
4. https://huggingface.co
5. https://developer.nvidia.com
6. https://portal.fabric-testbed.net
