# PyAMI: System Prototype
This repository is the official implementation of PyAMI for the paper: 
**Quantum-Secure Authentication and Key Agreement via Abstract Multi-Agent Interaction**

# Description
The PyAMI project implements the AMI authentication and key generation protocol in a multi-agent system consisting of multiple virtual machines. It also has an option to run the multi-agent system instead entirely on a local machine if no remote machines are available.

# Requirements
### Local Requirements

* [Python 3](https://www.python.org/downloads/) 

### Remote Requirements
If multi-agent system specified to run on remote machines, `run.py` will use [pip](https://pypi.org/project/pip/) to install the latest version of the following packages on provided machines:
* [Scipy](https://www.scipy.org/install.html)
* [PyTorch](https://pytorch.org/get-started/locally/)

# Start-up Guide
The following sections describe how to get up and running with the supported multi-agent system deployments:

1. Local directory structure.
2. JSON file format for agent parameters.
3. Run commands for supported deployments.

### 1. Local directory structure 
The `run.py` requires a specific directory/subdirectory structure on the local machine for agent information:

    agent_info_files/
        agent0_info.txt
        agent1_info.txt
        agent2_info.txt
        
    shared_secrets_files/
        agent0_secrets.txt
        agent1_secrets.txt
        agent2_secrets.txt
        
    agent_neuralnet_models/
        agent0_neuralnet_models/
            agent1.pth
            agent2.pth
        agent1_neuralnet_models/
            agent1.pth
        agent2_neuralnet_models/
            agent2.pth
            
    secret_neuralnet_models/
        agent0_secret_models/
            agent1.pth
            agent2.pth
        agent1_secret_models/
            agent0.pth
        agent2_secret_models/
            agent0.pth

**Note:** `agent_neuralnet_models/`'s subdirectories may contain none or less than *group size* neural net models, as agents may or may not use a neural net as their behavioral model. The same is true for `secret_neuralnet_models/`.

**Prefacing directory names:** The outer directories `agent_info_files/`, `shared_secrets_files/`, `agent_neuralnet_models/`, and `secret_neuralnet_models/` may be prefaced with the letter **c_** for *centralized* or **d_** indicating *decentralized*. 
Example: `c_agent_info_files/`, etc. 

##### Directories
Below are the required directories for the centralized setting. Repeat the below procedure using the decentralized prefix to run that setting. 

`c_agent_neuralnet_models/`
- Provided in repo.

`c_shared_secrets_files/`
 - Provided in repo.

`c_agent_neuralnet_models/`
 - Create directory and subdirectories by: 
   ```bash
   mkdir c_agent_neuralnet_models/
   mkdir c_agent_neuralnet_models/agent0_neuralnet_models/
   mkdir c_agent_neuralnet_models/agent1_neuralnet_models/
   mkdir c_agent_neuralnet_models/agent2_neuralnet_models/
   ```
  - The inner subdirectories are empty by default, but the subdirectories themselves must be created for the program to run.
  
`c_secret_neuralnet_models/`
 - Create directory and subdirectories by: 
   ```bash
   mkdir secret_neuralnet_models/
   mkdir secret_neuralnet_models/agent0_secret_models/
   mkdir secret_neuralnet_models/agent1_secret_models/
   mkdir secret_neuralnet_models/agent2_secret_models/
   ```
  - Again, the inner subdirectories are empty by default, but the subdirectories themselves must be created for the program to run. 
 
### 2. JSON file format for agent parameters
There are two multi-agent deployed settings available, along with their own agent classes:
 - Centralized setting.
   - Central Server Agent (**aka 'the central server'**)
   - Centralized System Agent
 - Decentralized setting.
   - Decentralized System Agent

##### 2A. Centralized Setting
In this setting, a central authority is entrusted with authentication of and key distribution to all users. This is similar to a public-key infrastructure where a certifying authority provides assurance of a user’s identity.

The default group size is 3 agents. 
The central server agent (class: **Central Server Agent**) typically has group ID of 0. 
The other two agents (class: **Centralized System Agent**) have group ID 1 and 2, respectively. 

Below are important adjustable parameters for these agents per file. 

**Central Server Agent**

`agent0_info.txt`:
```
group_id: 0 
# The central server typically has group ID of 0.

is_central_server: true
# Identify this agent as the central server.

central_server_group_id: 0 

auth_method: [-1, 'hypothesis_test', 'hypothesis_test'] 
# Authentication test for *agent 1* and *agent 2*. No authentication of self, so leave index 0 as null. 

ip_addr: 34.82.122.236
# IP address of remote machine for this agent.

models_list: [-1, {...}, {...}]
# The central server stores behavioral models to interact with agent 1 and agent 2 each in the multi-agent system.
```

`agent0_secrets.txt`:
```
secrets_list: [-1, {...}, {...}] 
# The central server stores the true behavioral model belonging to agent 1 and agent 2 as shared secrets.
```

**Centralized System Agent**

`agent1_info.txt`:
```
group_id: 1 

is_central_server: false 
# Identify that self is not central seerver.

central_server_group_id: 0

auth_method: 'hypothesis_test' 

ip_addr: 35.197.123.164
# IP address of remote machine for this agent.

models_list: [-1, {...}, -1]
# Store a behavioral model at own index, to interact (only) with central server in the multi-agent system.
```

`agent1_secrets.txt`:
```
secrets_list: [{...},-1,-1]
# Store the true model of the central server at index 0 as a shared secret.
```

`agent2_info.txt`:
```
group_id: 2

is_central_server: false
# Identify that self is not central server.

central_server_group_id: 0

auth_method: 'hypothesis_test'

ip_addr: 35.199.162.254
# IP address of remote machine for this agent.

models_list: [-1, -1, {...}]
# Store a behavioral model at own index, to interact (only) with central server in the multi-agent system.
```

`agent2_secrets.txt`:
```
secrets_list: [{...},-1,-1]
# Store the true model of the central server at index 0 as a shared secret.
```

##### 2B. Decentralized Setting
In this setting, no central authority exists, so users are individually responsible for authentication and key establishment.

The default group size is 3 agents. 
There is only one class of agent in this system.
The decentralized system agents (class: **Decentralized System Agent**) have group IDs of 0, 1, and 2.

Below are important adjustable parameters for these agents per file.

**Decentralized System Agent**
Decentralized system agents by default use multi-trees, a probabilistic decision tree variant which takes as input actions from all agents in the system. Agents generate multitree parameters from a unique seed parameter, which typically has the value of their group ID.

`agent0_info.txt`:
```
group_id: 0
models_list: [{'model_type':'multitree', 'unique_seed': 0,...}, -1, -1]
# Keep own model at own group index. Use multitree for group interaction process. Unique seed for multitree parameter generation is own index.
```

`agent0_secrets.txt`:
```
secrets_list: [-1, {'model_type':'multitree', 'unique_seed': 1,...}, {'model_type':'multitree', 'unique_seed': 2,...}]
# Store true behavioral models of *agent 1* and *agent 2* at their respective indices. 
```

`agent1_info.txt`:
```
group_id: 1
models_list: [-1, {'model_type':'multitree', 'unique_seed': 0,...}, -1]
# Keep own model at own group index. Use multitree for group interaction process. Unique seed for multitree parameter generation is own index.
```

`agent1_secrets.txt`:
```
secrets_list: [{'model_type':'multitree', 'unique_seed': 0,...}, -1, {'model_type':'multitree', 'unique_seed': 2,...}]
# Store true behavioral models of *agent 0* and *agent 2* at their respective indices. 
```

`agent2_info.txt`:
```
group_id: 2
models_list: [-1, -1, {'model_type':'multitree', 'unique_seed': 2,...}]
# Keep own model at own group index. 
```

`agent2_secrets.txt`:
```
secrets_list: [ {'model_type':'multitree', 'unique_seed': 0,...}, {'model_type':'multitree', 'unique_seed': 1,...}, -1]
# Store true behavioral models of *agent 0* and *agent 1* at their respective indices. 
```

### 3. Run commands for supported deployments.
PyAMI multi-agent system may be deployed either across remote machines (if available), or entirely on local machine for ease of use. 

#### 3A. Running with remote machines

##### Running (centralized system)
Agent json info files (`agent0_info.txt`, `agent1_info.txt`, `agent2_info.txt`) must contain IP address of live remote machines agents will run on, in "ipaddr" field as a string. 

Example:
```json
"ipaddr": "34.82.122.236"
```

Then run with:
```python
python run.py --system_type centralized --run_remotely yes --group_size 3 --path_agent_files agent_info_files --path_shared_secrets shared_secret_files --path_neuralnet_models agent_neuralnet_models --path_secret_neuralnets secret_neuralnet_models --system_type centralized
```

##### Running (decentralized system)
Provide IP addresses as in centralized case, then run with:

```python
python run.py --system_type decentralized --run_remotely yes --group_size 3 --path_agent_files agent_info_files --path_shared_secrets shared_secret_files --path_neuralnet_models agent_neuralnet_models --path_secret_neuralnets secret_neuralnet_models 
```
    
**Results:**

**Agent output files:**
Results for authentication and symmetric session keys are written to individual output files for each agent, and stored on the local machine. 
```
file_out_0.txt
file_err_0.txt
```

**Sample authentication results (centralized setting):**
A central server successfully authenticates agents a1 and a2:
```
Agent auth results:  [-1, True, True]
```

**Sample session key results (centralized setting):**
A central server creates group session key for two agents:
```
group_key: 0.302,0.390,0.328,0.329,0.329*0.355,0.330,0.433,0.318,0.420*0.964,0.406,0.786,0.881,0.900*0.533,0.676,0.844,0.844,0.972
```


#### 3B. Running entirely on single local machine

##### Running (centralized system)
Simply run with:
```bash
python run.py --system_type centralized --run_remotely no --group_size 3 --path_agent_files agent_info_files --path_shared_secrets shared_secret_files --path_neuralnet_models agent_neuralnet_models --path_secret_neuralnets secret_neuralnet_models 
```

**Results:**
The results are simply printed to the screen, instead of being stored as local output files. 

Below, the central server performs successful mutual authentication with *agent 1* and *agent 2*:

```
Authentication results for all agents:

agent  0 :  [-1, True, True]

agent  1 :  [True, -1, -1]

agent  2 :  [True, -1, -1]
```

and all agents possess an identical group session key (shown unhashed). 

```
Key agreement results for all agents: 

agent  0 :  0.302,0.417,0.275,0.335,0.329,0.264,0.277,0.402,0.385,0.264*0.355,0.330,0.433,0.318,0.420,0.248,0.371,0.248,0.201,0.377*0.964,0.594,0.900,0.079,0.930,0.900,0.186,0.843,0.133,0.881*0.533,0.676,0.844,0.844,0.972,0.985,0.972,0.951,0.726,0.950

agent  1 :  0.302,0.417,0.275,0.335,0.329,0.264,0.277,0.402,0.385,0.264*0.355,0.330,0.433,0.318,0.420,0.248,0.371,0.248,0.201,0.377*0.964,0.594,0.900,0.079,0.930,0.900,0.186,0.843,0.133,0.881*0.533,0.676,0.844,0.844,0.972,0.985,0.972,0.951,0.726,0.950

agent  2 :  0.302,0.417,0.275,0.335,0.329,0.264,0.277,0.402,0.385,0.264*0.355,0.330,0.433,0.318,0.420,0.248,0.371,0.248,0.201,0.377*0.964,0.594,0.900,0.079,0.930,0.900,0.186,0.843,0.133,0.881*0.533,0.676,0.844,0.844,0.972,0.985,0.972,0.951,0.726,0.950

```


##### Running (decentralized system)
Simply run with:
```bash
python run.py --system_type decentralized --run_remotely no --group_size 3 --path_agent_files agent_info_files --path_shared_secrets shared_secret_files --path_neuralnet_models agent_neuralnet_models --path_secret_neuralnets secret_neuralnet_models
```

**Results:**

Below, each agent successfully authenticates all others:

```
Authentication results for all agents:

agent  0 :  [-1, True, True]

agent  1 :  [True, -1, True]

agent  2 :  [True, True, -1]
```

and all agents compute an identical group session key (shown unhashed):

```
Key agreement results for all agents: 

agent  0 :  0.937,0.957,0.739,0.656,0.995,0.978,0.995,0.555,0.700,0.345*0.964,0.963,0.987,0.343,0.274,0.427,0.703,0.396,0.998,0.239*0.426,0.404,0.524,0.838,0.992,0.190,0.597,0.148,0.997,0.980

agent  1 :  0.937,0.957,0.739,0.656,0.995,0.978,0.995,0.555,0.700,0.345*0.964,0.963,0.987,0.343,0.274,0.427,0.703,0.396,0.998,0.239*0.426,0.404,0.524,0.838,0.992,0.190,0.597,0.148,0.997,0.980

agent  2 :  0.937,0.957,0.739,0.656,0.995,0.978,0.995,0.555,0.700,0.345*0.964,0.963,0.987,0.343,0.274,0.427,0.703,0.396,0.998,0.239*0.426,0.404,0.524,0.838,0.992,0.190,0.597,0.148,0.997,0.980
```

