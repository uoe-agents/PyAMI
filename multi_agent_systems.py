"""Classes for multi-agent systems and their agents."""
from agents import create_random_pdt_agent, create_random_pdt_multiagent

from network_functions import get_random_port_number, connect_socket_pairs
from network_functions import receive_from_socket, receive_action_from_socket
from network_functions import send_over_sockets, send_actions_over_sockets
from network_functions import setup_server_style_socket, setup_client_style_socket

from file_handling import update_json_file
from hyp_test import hypothesis_test_v1
from key_gen import session_key_v1, pbkd, encrypt, decrypt

from itertools import combinations 
import numpy as np
import random, torch
import time
from datetime import datetime

class LocalSystem(object):
    """Multi-agent system which lives entirely on local machine."""

    def __init__(self,system):
        """Create local system, either centralized or decentralized."""
        self.system = system
        self.system_type = system.system_type
        self.max_interaction_length = system.max_interaction_length
        self.group_size = system.group_size

    def local_interaction(self):
        """Perform interaction process between all agents locally."""
        raise NotImplementedError("System must implement local interaction.")

    def authenticate(self):
        """Have all agents perform authentication of others."""
        raise NotImplementedError("System must implement agent authentication.")

    def setup_session_key(self):
        """Have all agents create (or receive) group session key."""
        raise NotImplementedError("System must implement group session key.")

class CentralizedLocalSystem(LocalSystem):
    """Centralized multi-agent system which lives on local machine."""

    def __init__(self,system):
         # Use parent constructor
        super(CentralizedLocalSystem, self).__init__(system)

        # Store variables
        self.central_server_group_id = system.central_server_group_id

    def local_interaction(self):
        """Perform interaction process among all agents locally."""

        # Define variables
        group_interaction_history = []
        current_time = 0
        cs_gid = self.central_server_group_id

        # Build interaction history
        while(current_time < self.max_interaction_length):
            #print("\ncurrent_time: ", current_time)

            group_current_decisions = [-1 for x in range(self.group_size)]
            # example group_current_decisions: [[-1,s_a1,s_a2], a1, a2]

            # Get current action from each agent
            for l in range(self.group_size):

                # Get sublist of group actions from previous timestep
                if current_time == 0:
                    group_prev_decisions = []

                    # Agent's slice from history is just empty list also
                    prev_decisions = group_prev_decisions

                else:
                    # Properly format history slice as per type of agent

                    group_prev_decisions = group_interaction_history[-1][:] # shallow copy

                    # Build input depending on agent type
                    if l == cs_gid:

                        # Central server takes input as: [[-1, s_a1, s_a2], a1, a2]
                        prev_decisions = group_prev_decisions[:]

                    else:

                        # Other agents take input as: [s_a1, a1, -1] for example
                        cs_prev_action = group_prev_decisions[cs_gid][l] # i.e. 's_a1'

                        agent_prev_action = group_prev_decisions[l] # i.e. 'a1'

                        prev_decisions = [-1 for x in range(self.group_size)]

                        prev_decisions[cs_gid] = cs_prev_action

                        prev_decisions[l] = agent_prev_action

                # Now get agent output from formatted input
                agent_current_decision = self.system.agents_list[l].local_interact(
                                                                    current_time,
                                                                    prev_decisions)
                """
                If agent is central server, agent_current_decision is sublist:
                    [-1, s_a1, s_a2], for example

                Else, agent_current_decision is single action:
                    'a1', for example
                """

                # Update group current decisions for this agent
                group_current_decisions[l] = agent_current_decision
                # example group_current_decisions: [[-1, s_a1, s_a2], a1, a2]

            # Update group interaction history with all agent decisions
            group_interaction_history.append(group_current_decisions[:])

            # Update time counter
            current_time += 1

        # After while loop, check group H_t
        print("\ngroup_interaction_history: ", group_interaction_history)

        # After interaction process, update each agent's group interaction history
        for l in range(self.group_size):

            if l == cs_gid:

                # Central server stores its H_t as: [[[-1, s_a1, s_a2], a1, a2],...]
                cs_history = group_interaction_history[:]

                self.system.agents_list[l].set_group_interaction_history(cs_history)

            else:
                # Other agents store their H_t as: [[s_a1,a1,-1],...] for example
                
                # Build central server actions as: [s_a1 (t=0), s_a1 (t=1),...] for example
                cs_actions = [x[cs_gid][l] for x in group_interaction_history]

                # Build agent actions as: [a1 (t=0), a1 (t=1),...] for example
                agent_actions = [x[l] for x in group_interaction_history]

                # Build agent's specific H_t
                agent_history = []
                
                for (cs_action,agent_action) in zip(cs_actions,agent_actions):

                    # Build agent sublist for this timestep
                    history_sublist = [-1 for x in range(self.group_size)]
                    history_sublist[cs_gid] = cs_action
                    history_sublist[l] = agent_action
                    # example history_sublist: [cs_a1,a1,-1]

                    # Append this timestep sublist to agent historys
                    agent_history.append(history_sublist)

                # Now update agent's group interaction history member variable
                self.system.agents_list[l].set_group_interaction_history(agent_history)

    def authenticate_local_agents(self):
        """Have central server do mutual auth with other agents and store result."""

        # Have each agent perform its individual auth process
        for l in range(self.group_size):

            self.system.agents_list[l].authenticate(
                            self.system.agents_list[l].group_interaction_history, 
                            self.max_interaction_length)

            # Print results
            #print("\nAgent auth results: ", self.system.agents_list[l].auth_results)

    def setup_session_key(self):
        """Have all agents create and set session key(s)."""

        # Define variables
        cs_gid = self.central_server_group_id

        # Have each agent set own mutual keys (server may also make group key)
        for l in range(self.group_size):
            if l == cs_gid:
                encrypted_group_keys_sublist = self.system.agents_list[l].local_setup_key()
            else:
                self.system.agents_list[l].local_setup_key()

        # Reformat encrypted group keys list is its: [gk1_m1, gk2_m2] for example
        encrypted_group_keys = [-1 for x in range(self.group_size)]
        for l in range(self.group_size):
            if l == cs_gid:
                continue
            else:
                # Get next agent encrypted group key from sublist
                encrypted_group_keys[l] = encrypted_group_keys_sublist.pop(0)

        # now encrypted_group_keys is as: [-1,gk1_m1, gk2_m2] for example

        # Have agents manually decrypt encrypted group keys from central server
        if self.group_size > 2:

            # Get group key from central server
            group_key = self.system.agents_list[cs_gid].group_key

            # Other agents decrypt their encrypted group key
            for l in range(self.group_size):

                if l == cs_gid:
                    continue
                else:
                    # Group key encrypted by agent's mutual key
                    group_key_encr = encrypted_group_keys[l]

                    # Agent decrypts with its mutual key
                    group_key = self.system.agents_list[l].decrypt_group_key(
                                                                group_key_encr)
                    # Agent sets the group key for itself
                    self.system.agents_list[l].set_group_key(group_key)

class DecentralizedLocalSystem(LocalSystem):
    """Decentralized multi-agent system which lives on local machine."""

    def __init__(self,system):
         # Use parent constructor
        super(DecentralizedLocalSystem, self).__init__(system)

    def local_interaction(self):
        """Perform interaction process among all agents locally."""

        # Define variables
        group_interaction_history = []
        current_time = 0

        # Build interaction history
        while(current_time < self.max_interaction_length):

            group_current_decisions = [-1 for x in range(self.group_size)]
            # example group_current_decisions: [a0, a1, a2]

            # Get current action from each agent
            for l in range(self.group_size):

                # Get sublist of group actions from previous timestep
                if current_time == 0:
                    group_prev_decisions = []
                else:
                    group_prev_decisions = group_interaction_history[-1][:] # shallow copy

                # Get agent output from input: i.e. [a0,a1,a2]
                agent_current_decision = self.system.agents_list[l].local_interact(
                                                                    current_time,
                                                                    group_prev_decisions)

                # Update group current decisions with this agent action
                group_current_decisions[l] = agent_current_decision

            # Update group history with all agent actions for this timestep
            group_interaction_history.append(group_current_decisions[:])

            # Update time counter
            current_time += 1

        # After while loop, check group H_t
        print("\ngroup_interaction_history: ", group_interaction_history)

        # After interaction process, update each agent's personal history variable
        for l in range(self.group_size):

            # Every agents stores H_t as: [[a0,a1,a2],...]
            self.system.agents_list[l].set_group_interaction_history(
                                                    group_interaction_history)

    def authenticate_local_agents(self):
        """Have decentralized agents authenticate each other and store result."""

        # Have each agent perform its individual auth process
        for l in range(self.group_size):

            self.system.agents_list[l].authenticate(
                            self.system.agents_list[l].group_interaction_history, 
                            self.max_interaction_length)

            # Print results
            #print("\nAgent auth results: ", self.system.agents_list[l].auth_results)

    def setup_session_key(self):
        """Have all agents create and set group session key."""

        for l in range(self.group_size):
            self.system.agents_list[l].local_setup_key()

class RemoteSystem(object):
    """Single-agent system which lives on remote machine."""

    def __init__(self, system_settings, agent_file, secrets_file, 
                        model_statedicts, secret_statedicts):
        """
        Create remote system with single agent.

        Args:
        agent_file: json info dict for the sole system agent.
        secrets_file: json secrets dict for the system agent's shared secrets.
        
        model_statedicts: list of statedict model(s) for system agent.
        secret_statedicts: list of statedict models for system agent.
        """

        # Get variables from system settings file
        self.system_type = system_settings['system_type']
        self.group_size = system_settings['group_size']
        self.max_interaction_length = system_settings['max_interaction_length']

        if self.system_type == 'centralized':
            self.central_server_group_id = system_settings['central_server_group_id']

        # Create single agent for this system
        self.agent = self.create_system_agent(self.group_size, 
                                agent_file, secrets_file, 
                                model_statedicts, secret_statedicts)

    def create_system_agent(self, group_size,
                            agent_file, secrets_file, 
                            model_statedicts, secret_statedicts):
        """Create a single agent for remote system."""

        # Get agent-specific info       
        agent_group_id = agent_file['group_id']
        
        # Create agent depending on its specified type
        if self.system_type == 'centralized':
            agent_is_central_server = agent_file['is_central_server']

            if agent_is_central_server:
                agent = CentralServerAgent(agent_file, secrets_file, 
                                    model_statedicts, secret_statedicts,
                                    self.max_interaction_length)
            else:
                agent = CentralizedSystemAgent(agent_file, secrets_file, 
                                    model_statedicts, secret_statedicts,
                                    self.max_interaction_length)
        else:
            agent = DecentralizedSystemAgent(agent_file, secrets_file, 
                                    model_statedicts, secret_statedicts,
                                    self.max_interaction_length)

        # Return created agent object
        return agent

    def connect_sockets(self, agent_json_dict, network_protocol):
        """Create and connect either TCP sockets or UDP sockets."""

        if network_protocol == 'tcp':
            self.agent.connect_tcp_sockets(agent_json_dict)

        elif network_protocol == 'udp':
            self.agent.connect_udp_sockets(agent_json_dict)

        else:
            exit("Exiting: network_protocol incorrectly specified.")

    def remote_interaction(self, network_protocol='tcp'):
        """
        Build interaction history with remote agents by sending and receiving actions.
        Also update remote system agent's shared secrets for later key generation.
        """

        # Define variables
        group_interaction_history = [] # contains sublist for each timestep
        current_time = 0

        # Start time
        start_time = time.time()
        start_datetime = datetime.now()

        # Build interaction history by sending and receiving actions
        while(current_time < self.max_interaction_length):
            #print("\ncurrent_time: ", current_time)

            # DEBUG: write current time to file
            #write_str = 'current_time_'+str(current_time)
            #f = open(write_str, 'w')
            #f.close()

            # Get sublist of group actions from previous timestep
            if current_time == 0:
                prev_decisions = []
            else:
                prev_decisions = group_interaction_history[-1][:] # shallow copy

            #print("\nprev_decisions provided: ", prev_decisions)
            #print("group_interaction_history after providing: ", group_interaction_history)

            # Get next action(s), update secret(s), transmits action(s), receive action(s)
            group_current_decisions = self.agent.interact(current_time, prev_decisions, network_protocol)
            #print("\nGot from interact() the group_current_decisions: ", group_current_decisions)
            
            """
            group_current_decisions is different for each type of system agent.

            Example for CentralizedSystemAgent at id=1: 
                [cs_action, a1_action, -1]

            Example for CentralServerAgent: 
                [[-1, action_for_a1, action_for_a2], a1_action, a2_action]

            Example for DecentralizedSystemAgent at id=0:
                [a0_action, a1_action, a2_action]
            """

            # Update group history
            group_interaction_history.append(group_current_decisions)
            #print("\nUpdated group_interaction_history: ", group_interaction_history)

            # Update timestep
            current_time += 1

        # End timer
        end_time = time.time()
        end_datetime = datetime.now()

        # Update system agent with group interaction history
        self.agent.set_group_interaction_history(group_interaction_history)

        # Print interaction results
        #print("\ngroup_interaction_history: ", group_interaction_history)
        
        # Print behavioral model interaction histories
        #print("\nAgent model history: ", self.print_info('model_history'))

        # Print shared secrets interaction histories
        #print("\nAgent shared secret histories: ", self.print_info('secrets_histories'))

        # Return remote interaction time
        elapsed_time = end_time - start_time
        return (elapsed_time, start_datetime, end_datetime)

    def authenticate_remote_agents(self, agent_file):
        """Have agent authenticate other remote agent(s) and store result."""

        # Have agent perform authentication
        self.agent.authenticate(self.agent.group_interaction_history, 
                                self.max_interaction_length, 
                                agent_file)
        # Print results
        #print("\nAgent auth results: ", self.agent.auth_results)

    def setup_session_key(self):
        """Have agent create and set session key(s)."""

        # Have agent set up or receive group session key
        self.agent.setup_key()


    def print_info(self, info_str):
        """
        Print specified member variable for system agent. 

        Args:
        info_str: a choice from ['agent_model', 'agent_secrets', 
                                'agent_sockets', 'interaction_history']
        """
        print("\nPrinting: ", info_str)
            
        if info_str == 'agent_model':
            self.agent.print_agent_model()
        elif info_str == 'agent_secrets':
            self.agent.print_agent_secrets()
        elif info_str == 'agent_sockets':
            self.agent.print_agent_sockets()
        elif info_str == 'model_history':
            self.agent.print_model_history()
        elif info_str == 'secrets_histories':
            self.agent.print_secrets_histories()
        else:
            exit("Exiting: Incorrect info string supplied to print_info()!")


class MultiAgentSystem(object):
    """Parent class."""

    def __init__(self, group_size, agent_files, secrets_files):
        """
        Set system specs.

        Args:
        agent_files: sorted list of json dicts for each agent
        secrets_files: sorted list of json dicts for each agent's shared secrets
        """

        # Check number of provided files matches group size
        assert len(agent_files) == group_size, "Exiting: Incorrect number of agent files for group size!"
        assert len(secrets_files) == group_size, "Exiting: Incorrect number of secrets files for group size!"

        # Set variables
        self.group_size = group_size
        self.agents_list = -1
        self.max_interaction_length = 10 # length of agent-to-agent interaction histories

    def create_system_agents(self, group_size, agent_files, secrets_files):
        """Instantiate agents in the system."""
        raise NotImplementedError("System must implement agent creation.")

    def setup_socket_info(self):
        """Assign IP address and port numbers to each agent in the system."""
        raise NotImplementedError("System must assign socket information.")


    def print_info(self, info_str):
        """
        Print specified member variable for all agents.

        Args:
        info_str: a choice from ['agent_model', 'agent_secrets', 'agent_sockets']
        """

        print("\nPrinting: ", info_str)

        for (agent, id) in zip(self.agents_list, range(self.group_size)):
            print("\nFor agent id %d" % id)
            
            if info_str == 'agent_model':
                agent.print_agent_model()
            elif info_str == 'agent_secrets':
                agent.print_agent_secrets()
            elif info_str == 'agent_sockets':
                agent.print_agent_sockets()
            else:
                exit("Exiting: Incorrect info string supplied to print_info()!")


class CentralizedSystem(MultiAgentSystem):
    """MAS with central server and public agents."""

    def __init__(self, group_size, agent_files, secrets_files, 
                        agent_model_statedicts, shared_secret_statedicts):
        """Initalize central server (group index=0) and other public agents.

        Args:
        agent_model_statedicts: i.e. [
                                [-1, agent1.pth, agent2.pth], 
                                [-1, agent1.pth, -1], 
                                [-1, -1, agent2.pth] 
                            ]
        """

        # Use parent constructor
        super(CentralizedSystem, self).__init__(group_size, agent_files, secrets_files)

        # Define variables
        self.system_type = 'centralized'
        self.group_size = group_size

        # Set central server as first agent in group; update agent files accordingly
        self.central_server_group_id = 0

        # Create agents
        self.agents_list = self.create_system_agents(group_size, agent_files, secrets_files, 
                                            agent_model_statedicts, shared_secret_statedicts)


    def create_system_agents(self, group_size, agent_files, secrets_files, 
                                agent_model_statedicts, shared_secret_statedicts):
        """Create agents for centralized system.

        Args:
        agent_model_statedicts: i.e. [ [-1,agent1.pth,-1], [-1,agent1.pth,-1], [-1,-1,-1] ]

        shared_secret_statedicts: i.e. [ [-1,agent1.pth,-1], [agent0.pth,-1,-1], [-1,-1,-1]]
        """

        agents_list = [-1 for _ in range(group_size)]

        for l in range(group_size):

            agent_file = agent_files[l]
            secrets_file = secrets_files[l]
            statedict_models = agent_model_statedicts[l]
            statedict_secrets = shared_secret_statedicts[l]

            # Create either central server or system agent
            if l == self.central_server_group_id:
                agents_list[l] = CentralServerAgent(agent_file, secrets_file, 
                                            statedict_models, statedict_secrets,
                                            self.max_interaction_length)
            else:
                agents_list[l] = CentralizedSystemAgent(agent_file, secrets_file, 
                                            statedict_models, statedict_secrets,
                                            self.max_interaction_length)

        return agents_list

    def setup_socket_info(self, path_agent_files):
        """
        Central server is assigned socket info for all other agents,...
        while other agents get socket info only for central server. 
        """

        # Declare variables
        group_socket_info_list = [-1 for _ in range(self.group_size)]
        central_server_socket_info_list = [-1 for _ in range(self.group_size)]

        cs_gid = self.central_server_group_id
        central_server_ipaddr = self.agents_list[cs_gid].ipaddr

        # Assign central server's ports to use for other agents
        for l in range(self.group_size):

            if l == cs_gid:
                continue

            else:
                agent_ip = self.agents_list[l].ipaddr

                port1 = get_random_port_number(2000,1000)
                port2 = get_random_port_number(2000,1000)

                central_server_socket_info_list[l] = [(agent_ip, port1), (agent_ip, port2)]

        # Record for group socket info list and for central server agent
        group_socket_info_list[cs_gid] = central_server_socket_info_list

        self.agents_list[cs_gid].set_socket_info(central_server_socket_info_list, path_agent_files)

        # Now assign ports to other agents from the ports central server uses
        for l in range(self.group_size):
            
            if l == cs_gid:
                continue

            else:   
                agent_socket_info = [-1 for _ in range(self.group_size)]

                # Example: [(agent_ip, port1), (agent_ip, port2)]
                central_server_agent_info = central_server_socket_info_list[l]

                # Get pre-assigned ports central server uses for this agent
                agent_port1 = central_server_agent_info[0][1] # first tuple from sublist
                agent_port2 = central_server_agent_info[1][1] # second tuple from sublist

                agent_socket_info[cs_gid] = [(central_server_ipaddr, agent_port1), 
                                            (central_server_ipaddr, agent_port2)]

                # Record in group socket info list and for this agent
                group_socket_info_list[l] = agent_socket_info

                self.agents_list[l].set_socket_info(agent_socket_info, path_agent_files)


        # Record group socket info list for system
        self.group_socket_info_list = group_socket_info_list

class DecentralizedSystem(MultiAgentSystem):
    """MAS with all group agents."""

    def __init__(self, group_size, agent_files, secrets_files, 
                        agent_model_statedicts, shared_secret_statedicts):
        """Initalize group of agents."""

        # Use parent constructor
        super(DecentralizedSystem, self).__init__(group_size, agent_files, secrets_files)

        # Define variables
        self.system_type = 'decentralized'
        self.group_size = group_size

        # Create agents
        self.agents_list = self.create_system_agents(group_size, agent_files, secrets_files,
                                            agent_model_statedicts, shared_secret_statedicts)


    def create_system_agents(self, group_size, agent_files, secrets_files, 
                                agent_model_statedicts, shared_secret_statedicts):
        """Create agents for decentralized system."""

        agents_list = [-1 for _ in range(group_size)]

        for l in range(group_size):

            agent_file = agent_files[l]
            secrets_file = secrets_files[l]
            statedict_models = agent_model_statedicts[l]
            statedict_secrets = shared_secret_statedicts[l]

            # Create system agent for decentralized system
            agents_list[l] = DecentralizedSystemAgent(agent_file, secrets_file,
                                            statedict_models, statedict_secrets,
                                            self.max_interaction_length)

        return agents_list

    def setup_socket_info(self, path_agent_files):
        """
        Assign all agents in group IP addresses and port numbers to connect with other agents.
        """

        # Declare variables
        group_socket_info_list = []
        for l in range(self.group_size):
            group_socket_info_list.append([-1 for _ in range(self.group_size)])

        # Example: [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]

        # Create list of combinations of every two agents in group
        agent_combos = combinations(range(self.group_size), 2)

        # Example for group_size=3:  [(0, 1), (0, 2), (1, 2)]
        agent_combos_list = [c for c in agent_combos] # convert to python list
        num_combos = len(agent_combos_list)

        # For each combination of two agents, set random port numbers to be *shared*
        agent_conn_info_list = []

        for combo in agent_combos_list:

            port1 = get_random_port_number(2000,1000) # i.e. port 35
            port2 = get_random_port_number(2000,1000) # i.e. port 36

            # Get agent group id's for this combo, i.e. (0,2)
            first_id = combo[0]
            second_id = combo[1]

            # Assign socket info for first agent in combo

            group_socket_info_list[first_id][first_id] = -1 # set own info to -1

            second_ipaddr = self.agents_list[second_id].ipaddr # ipaddr of other agent

            # Example: [(ipaddr_a2, port35), (ipaddr_a2, port36)]
            sublist_for_second = [(second_ipaddr, port1), (second_ipaddr, port2)]

            group_socket_info_list[first_id][second_id] = sublist_for_second


            # Assign socket info for second agent in combo
            group_socket_info_list[second_id][second_id] = -1 # set own info to -1

            first_ipaddr = self.agents_list[first_id].ipaddr # ipaddr of other agent

            # Example: [(ipaddr_a1, port35), (ipaddr_a1, port36)]
            sublist_for_first = [(first_ipaddr, port1), (first_ipaddr, port2)]

            group_socket_info_list[second_id][first_id] = sublist_for_first


        # Record socket info for each agent
        for l in range(self.group_size):
            self.agents_list[l].set_socket_info(group_socket_info_list[l], path_agent_files)

        # Record group socket info for system
        self.group_socket_info_list = group_socket_info_list

class SystemAgent(object):
    """Agent which exists in the multi-agent system."""

    def __init__(self, info_dict):
        """Initialize the system agent. 
        
        Args:
        info_dict: python dict from json object
        secrets_info_list: python list of dicts for shared secrets

        """

        # Agent-specific info
        self.group_id = info_dict['group_id']
        self.group_size = info_dict['group_size']
        self.auth_method =  info_dict['auth_method']
        self.ipaddr = info_dict['ipaddr']
        self.action_space_size = info_dict['action_space_size']

        # Common variables
        self.sockets_list = [-1 for _ in range(self.group_size)]
        # Updated later as: i.e. [[send_sock,recv_sock],-1,-1]

        self.received_actions_list = -1 # set by inherited class

        self.auth_results = [-1 for _ in range(self.group_size)]

        self.group_key = -1 # group session key

    def set_group_key(self, group_key):
        """Set the group session key."""
        self.group_key = group_key

    def print_agent_model(self):
        """Print agent model(s) depending on whether agent re-uses its model."""
        raise NotImplementedError("Agent must implement printing of its models.")

    def print_agent_secrets(self):
        """Print shared secrets for all other agents."""
        print("\nPrinting shared secrets:")
        for secret in self.shared_secrets_list:
            print("agent secret: ", secret)

    def print_agent_sockets(self):
         """Print socket info for all other agents."""
         print("Printing sockets info:")
         for info in self.socket_info_list:
            print("socket info: ", info)

    def set_seed(self, seed):
        """Helper function to seed all RNG."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_agent_model(self, info_dict, statedict):
        """
        Create either PDTAgent from unique seed or neural network class from statedict.
        Function is meant to be re-used by subclasses.

        Args:
        info_dict: python dict of agent-specific info.
        statedict: .pth file with specific neural net
        """

        # Create agent model from unique seed
        if info_dict['model_type'] == 'pdt':

            # Seed from agent's unique creation seed
            unique_seed = info_dict['unique_seed']
            self.set_seed(unique_seed)  

            # Get PDT-specific parameters
            depth = info_dict['depth']
            n_actions = info_dict['n_actions']
            window = info_dict['window']
            temperature = info_dict['temperature']

            # Create pdt model
            agent_model = create_random_pdt_agent(depth, n_actions, window, temperature)
        
        elif info_dict['model_type'] == 'multitree':

            # Seed from agent's unique creation seed
            unique_seed = info_dict['unique_seed']
            self.set_seed(unique_seed)  

            # Get Multitree-specific parameters
            depth = info_dict['depth']
            n_actions = info_dict['n_actions'] # action space size
            window = info_dict['window']
            temperature = info_dict['temperature']

            # Create pdt model
            agent_model = create_random_pdt_multiagent(depth, n_actions, 
                                n_other_agents=self.group_size-1, 
                                window=window, temperature=temperature)

        else:
            # Create neural net from statedict and parameters in info_dict
            exit("Exiting: Neural net agent model creation not yet implemented!")

        return agent_model


    def create_shared_secrets(self, secrets_info_dict, statedict_secrets):
        """Create list of shared secrets for this agent.

        Args:
        secrets_info_dict: agent-specific json object with permitted secrets 
                           i.e. {
                                "secrets_list": [-1, {agent1 info}, -1]
                                }
        """

        # Initialize empty list of secrets for other agents
        shared_secrets_list = [-1 for _ in range(self.group_size)]

        # Collect list of secrets dicts from json dict
        secrets_info_list = secrets_info_dict["secrets_list"] 

        # Create shared secret for other agents
        for l in range(self.group_size):

            if l == self.group_id:
                continue

            # Create actual agent model from dict
            if secrets_info_list[l] != -1:
                info_dict = secrets_info_list[l]
                statedict = statedict_secrets[l]

                shared_secrets_list[l] = self.create_agent_model(info_dict, statedict)

        return shared_secrets_list

    def set_shared_secrets(self, secrets_info_dict, statedict_secrets):
        """Set list of shared secrets for other agents.

        Args: 
        secrets_info_dict: agent-specific json object with permitted secrets 
                           i.e. {
                                "secrets_list": [-1, {agent1 info}, -1]
                                }
        statedict_secrets: list of .pth files for other agents which use neural net
        """

        self.shared_secrets_list = self.create_shared_secrets(secrets_info_dict, statedict_secrets)

    def set_socket_info(self):
        """Record IP address and port numbers to connect to other agents."""

        raise NotImplementedError("Agent must implement setting socket info.")

    def connect_tcp_sockets(self):
        """Connect to other agent(s) in group using socket pair for them."""

        raise NotImplementedError("Agent must implement connecting sockets to other agents.")

    def check_receive_sockets(self):
        """Collect received actions from other agents during remote interaction."""

        raise NotImplementedError("Agent must implement collecting received actions.")

    def broadcast_actions(self, actions_for_group, t, network_protocol):
        """Send selected action(s) from behavioral model(s) to rest of group."""

        raise NotImplementedError("Agent must implement broadcasting actions.")

    def interact(self, t, prev_actions, network_protocol):
        """
        Perform remote interaction for this timestep.

        1. Get next action(s)
        2. Update shared secret(s) 
        3. Transmits action(s)
        4. Receive action(s).

        Args: 
        t: current timestep in interaction process
        prev_actions: sublist with group actions from previous timestep 
        """
        raise NotImplementedError("Agent must implement remote interaction.")

    def authenticate(self):
        """Authenticate other agent(s) using either hyp test or neural net."""
        raise NotImplementedError("Agent must implement authentication.")

    def setup_key(self):
        """Create and set session keys for encryption/decryption."""
        raise NotImplementedError("Agent must implement key generation.")

    def print_secrets_histories(self):
        """Print interaction histories for all shared secrets."""

        print("\nPrinting agent shared secrets histories: ")

        for secret_model, id in zip(self.shared_secrets_list, range(self.group_size)):

            if secret_model == -1:
                print("\nsecret id: ", id, " history: -1")
                continue

            print("\nsecret id: ", id, " history: ", secret_model._interaction_history)


class CentralServerAgent(SystemAgent):
    """A central server for authenticating multiple clients in a centralized settting."""

    def __init__(self, info_dict, secrets_info_dict, statedict_models, statedict_secrets,
                    max_interaction_length):
        """Initialize a central server."""

        # Use inherited constructor
        super(CentralServerAgent, self).__init__(info_dict)

        # Set new variables
        self.is_central_server = info_dict['is_central_server']
        self.reuse_agent_model = info_dict['reuse_agent_model']
        
        if 'hypothesis_test' in info_dict['auth_method']:
            self.alpha = info_dict['alpha']

        self.group_interaction_history = -1
        

        # Set agent model(s)
        self.set_agent_model(info_dict, statedict_models)

        # Set shared secrets using inherited method
        self.set_shared_secrets(secrets_info_dict, statedict_secrets)

        # Set received actions list as [-1, [-1,..,-1], [-1,..,-1]]
        # Each sublist is sized interaction len, to be filled during interaction
        self.set_received_actions_list(self.group_size, max_interaction_length)

        # Initialize mutual key list
        self.mutual_keys_list = [-1 for _ in range(self.group_size)]
        
    def set_mutual_keys_list(self, mutual_keys_list):
        """Set the list of mutual keys for all agents."""
        self.mutual_keys_list = mutual_keys_list

    def set_group_interaction_history(self, history):
        """
        Set group interaction history.

        Args:
        history: list of sublists for each timestep in interaction
        
        Format: [
                    [[-1, action_for_a1, action_for_a2], a1_action, a2_action],
                    [[-1, action_for_a1, action_for_a2], a1_action, a2_action],
                    ...
                ]
        """
        self.group_interaction_history = history

    def set_received_actions_list(self, group_size, max_interaction_length):
        """
        Set received actions list by max interaction length.
        Format of list: [-1, [-1,...,-1], [-1,...,-1]]
        """

        # Define variables
        received_actions_list = [-1 for _ in range(group_size)]

        # Create set-length sublist for other agents
        for l in range(self.group_size):

            if l == self.group_id:
                continue

            received_actions_list[l] = [-1 for _ in range(max_interaction_length)]

        # Set for self
        self.received_actions_list = received_actions_list

    def set_agent_model(self, info_file, statedict_models):
        """Central server may use an indentical model for all users...
        or it may keep a different model for each user.

        Args:
        info_file: json file with following example format
            {
                "group_id": 
                "reuse_agent_model":
                ...

                "models_list": [-1, 
                                {"unique_seed":0, "model_type": 'pdt', ...}, 
                                -1]
            }
        statedict: list of .pth files for agent to use, i.e. [-1, agent1.pth, agent2.pth]
        """

        # Get list of behavioural models from dict
        models_list = info_file["models_list"]

        """
        Central server may have kept an identical model for all others...
        or it may haev kept a unique model for others.
        """
        
        # Example: model_list = [-1, {...}, {...}]
        self.agent_models = [-1 for _ in range(self.group_size)]

        for l in range(self.group_size):
            if l == self.group_id:
                continue
            else:
                info_dict = models_list[l]
                statedict = statedict_models[l] # get model to use with other agent

                self.agent_models[l] = self.create_agent_model(info_dict, statedict)

    def set_socket_info(self, socket_info_list, path_agent_files):
        """Record IP address and port numbers to connect to other agents."""
        
        # Example: [-1, [(a1_ip, port25),(a1_ip, port26)], [(a2_ip, port34),(a2_ip, port35)]]
        self.socket_info_list = socket_info_list

        # Update agent info json file (i.e. 'agent_info_files/agent0_info.txt')
        agent_info_filepath = path_agent_files + '/agent'+str(self.group_id)+'_info.txt'
        
        update_json_file(agent_info_filepath, key='socket_info_list', value=socket_info_list)

    def connect_tcp_sockets(self, info_dict):

        """(Remote use) Setup pair of server-style sockets for all others agents,...
        ...then connect them to remote sockets."""

        # Format: [-1, [(a1_ip,port26),(a1_ip,port27)], [(a2_ip,port35),(a2_ip,port36)]]
        socket_info_list = info_dict['socket_info_list']

        # For each agent in group, connect my sockets to theirs
        for l in range(self.group_size):

            # Skip myself
            if l == self.group_id:
                continue

            # Central server sockets are server-style, other agents have client-style
            setup_function = setup_server_style_socket

            # Get connection info for each socket (IP addresses are the same)
            (ipaddr1, port1) = socket_info_list[l][0]  # (a1_ipaddr, port1)
            (ipaddr2, port2) = socket_info_list[l][1]  # (a1_ipaddr, port2)

            """ 
            From joint port numbers [(ip,port1), (ip, port2)], central server
            will send over port1 and receive over port2. Other agents
            send over port2 and receive over port1. 

            This is why thread labels designate thread1 to send over port1 and 
            thread2 to recieve over port2.
            """

            # Labels for each thread socket's function in the connection
            thread1_type = 'send'
            thread2_type = 'receive'

            # Prepare socket info tuples for function
            socket1_info_tuple = (ipaddr1, port1, thread1_type)
            socket2_info_tuple = (ipaddr2, port2, thread2_type)

            # Get prepared sockets as [send_socket, recv_socket]
            sockets_for_agent = connect_socket_pairs(setup_function, 
                                        socket1_info_tuple, socket2_info_tuple)

            # Update sockets list 
            # Example format: [-1,[send_sock,recv_sock],[send_sock,recv_sock]])
            self.sockets_list[l] = sockets_for_agent

            print("\nself.sockets_list: ", self.sockets_list)

    def local_interact(self,t,prev_actions):
        """
        Perform local interaction for this timestep.
        
        1. Get next action(s)
        2. Update shared secret(s)

        Args: 
        t: current timestep in interaction process
        prev_actions: sublist with group actions from previous timestep 
        """
        
        # 1. Get next action from models, which also store their (action,dist)
        own_actions_list = self.model_next_action(t, prev_actions)

        # example own_actions_list: [-1, action_for_a1, action_for_a2]

        # 2. Update shared secrets which store their own dists
        self.secrets_next_action(t, prev_actions)

        # return just own actions sublist for this timestep
        return own_actions_list


    def interact(self, t, prev_actions, network_protocol):
        """
        Perform remote interaction for this timestep.

        1. Get next action(s)
        2. Update shared secret(s) 
        3. Transmits action(s)
        4. Receive action(s).

        Args: 
        t: current timestep in interaction process
        prev_actions: sublist with group actions from previous timestep 
        """

        # 1. Get next action from models, which also store their (action,dist)
        own_actions_list = self.model_next_action(t, prev_actions)

        # example own_actions_list: [-1, action_for_a1, action_for_a2]

        # 2. Update shared secrets which store their own dists
        self.secrets_next_action(t, prev_actions)

        # 3. Transmit own actions to other agents
        self.broadcast_actions(own_actions_list, t, network_protocol)

        # 4. Collect actions from receive sockets

        # blocking call: collect from receive sockets and store in self
        self.check_receive_sockets(t)

        # example of self.received_actions_list: [-1, [-1,..,-1], [-1,..,-1]]

        # collect received actions from self
        group_actions_list = [-1 for _ in range(self.group_size)]
        group_actions_list[self.group_id] = own_actions_list

        for l in range(self.group_size):
            if l == self.group_id:
                continue

            received_actions_sublist = self.received_actions_list[l]

            # get action from other agent at current timestep
            group_actions_list[l] = received_actions_sublist[t]

        """
        example of group_actions_list (group actions at this timestep): 

        [ [-1, action_for_a1, action_for_a2], a1_action, a2_action]
        """

        # return interaction history for this timestep
        return group_actions_list


    def model_next_action(self,t, group_prev_actions):
        """
        Behavioral models provide and personally store next action & dist.
        
        Args:
        t: current timestep
        group_prev_actions: group actions from previous timestep 
                    i.e. [[-1, action_for_a1, action_for_a2], a1_action, a2_action]
        """

        # Define variables
        next_actions_list = [-1 for _ in range(self.group_size)]

        for l in range(self.group_size):

            if l == self.group_id:
                continue

            if self.agent_models[l]._type == 'pdt':

                # PDTAgent conditions on most recent other agent action
                if t == 0:
                    inputs = []
                else:
                    inputs = [group_prev_actions[l]]

                # Agent model self-stores then provides action,dist
                (action, dist) = self.agent_models[l].act(t, inputs)

            elif self.agent_models[l]._type == 'neuralnet':

                # Neuralnet conditions on: [server_i, client_i]
                if t == 0:
                    inputs = []
                else: 
                    s_i = group_prev_actions[self.group_id][l]
                    c_i = group_prev_actions[l]

                    inputs = [s_i, c_i]

                # Agent model stores then provides integer action, dist
                (logits, action) = self.agent_models[l].act(t,inputs,self.action_space_size)

            else:
                exit("Exiting: Agent model not of supported types.")

            # Place in next_actions_list at index
            next_actions_list[l] = action

        # Return list of next actions from all models
        return next_actions_list

    def secrets_next_action(self, t, group_prev_actions):
        """
        Have shared secrets update with next action and dist.
        Only the dists are significant for later key computation.

        Args:
        t: current timestep
        group_prev_actions: group actions from previous timestep 
                i.e. [[-1, action_for_a1, action_for_a2], a1_action, a2_action]
        """

        for l in range(self.group_size):

            if l == self.group_id:
                continue

            if self.shared_secrets_list[l]._type == 'pdt':

                # PDTAgent conditions on most recent other agent action
                if t == 0:
                    inputs = []
                else:   # provide own action to condition on
                    inputs = [group_prev_actions[self.group_id][l]]

                # Secret model stores own dist,action
                (_, _) = self.shared_secrets_list[l].act(t, inputs)

            elif self.shared_secrets_list[l]._type == 'neuralnet':

                """
                Secret neuralnet conditions on: [server_i, client_i], not on own actions.
                The shared secret model just needs to self-store expected dists for
                ...later key computation. Its action output is not required.
                """
                if t == 0:
                    inputs = []
                else: 
                    s_i = group_prev_actions[self.group_id][l][:]
                    c_i = group_prev_actions[l][:]

                    inputs = [s_i, c_i]

                # Secret model stores own action, dist
                (_, _) = self.shared_secrets_list[l].act(t,inputs,self.action_space_size)

            else:
                exit("Exiting: Agent model not of supported types.")


    def broadcast_actions(self, actions_for_group, t, network_protocol):
        """
        Send own selected actions to other agents at this timestep in interaction.

        Args:
        actions_for_group: list of own selected actions at this timestep in interaction.
                          i.e. [-1, action_for_a1, action_for_a2]
        t: current timestep in remote interaction process.
        """

        # Get list of send sockets, and own actions for other agents
        send_sock_list = []
        actions_list = []

        for l in range(self.group_size):

            if l == self.group_id:
                continue

            # Get send socket for agent from tuple: (send_sock, recv_sock)
            send_sock = self.sockets_list[l][0]
            send_sock_list.append(send_sock)

            # Get own action for this agent
            action = actions_for_group[l]
            actions_list.append(action)

        # Send collected actions over collected send sockets
        send_actions_over_sockets(send_sock_list, actions_list, t, network_protocol)

    def check_receive_sockets(self, t):
        """
        Collect received actions from other agents during remote interaction.
        Update self.received_actions_list with all received actions.
        """
        #print("\nInside check_receive_sockets!")

        for l in range(self.group_size):
            #print("\nl: ", l)

            if l == self.group_id:
                continue

            # Get receive socket for central server from tuple: (send_sock, recv_sock)
            receive_sock = self.sockets_list[l][1]

            # Place blocking call in script until action appears at recv socket
            (msg_list, msg_list_tuples) = receive_action_from_socket(receive_sock)

            # Example msg_list: ['3F_t0_C', '2F_t1_C'] if more than one message received
            # Example time-ordered msg_list_tuples: [(3,0), 2,1]

            # Ensure only a single msg in the list is for this current timestep
            current_time_str = 't' + str(t) # i.e. 't3' 
            msg_list_current_timestep = [current_time_str in x for x in msg_list]

            assert msg_list_current_timestep.count(True) == 1, "Only one msg should have current time!"

            # Place in agent's sublist at the index for timestep 
            # Format: [-1, [a1_t1, a1_t2,...,a1_T], [a2_t1, a2_t2,...,a2_T]]

            for msg_tuple in msg_list_tuples: # i.e. [(3,0), (2,1)]
                
                msg_action = msg_tuple[0]
                msg_time = msg_tuple[1]

                self.received_actions_list[l][msg_time] = int(msg_action)

    def authenticate(self, group_history, max_history_len): #, agent_info_dict):
        """Authenticate other agents in the group.
        
        Args: 
        group_history: [ [[-1,cs_for_a1,cs_for_a2],a1_action,a2_action],...]
        max_history_len: length of complete interaction history.
        """

        # Run authentication test for each agent
        for l in range(self.group_size):
            if l == self.group_id:
                continue

            if self.auth_method[l] == 'hypothesis_test':
                
                # Define variables
                alpha = self.alpha #agent_info_dict['alpha']
                
                # Get action of unknown agent from group history
                unknown_agent_actions = [x[l] for x in group_history]

                # Get dists of shared secret for central server
                secret_history = self.shared_secrets_list[l]._interaction_history 
                # Example secret_history: [(action,dist), (action,dist),...]

                known_agent_dists = [x[1] for x in secret_history]

                # Get pvalues from hypothesis test
                (pvalue_reg, pvalue_ratio) = hypothesis_test_v1(
                                                unknown_agent_actions, 
                                                known_agent_dists, 
                                                update_times = [max_history_len-1],
                                                num_actions = self.action_space_size)
                # Determine authentication success
                result = True if pvalue_ratio >= alpha else False
                self.auth_results[l] = result

            elif self.auth_method[l] == 'classifier':
                pass
            else:
                exit("Exiting: Specified authentication method not supported.")

    def collect_own_histories(self):
        """
        Helper function to collect formatted histories of own models.
        Collect histories from self.agent_models.
        Format each history as: [(action,dist), (action,dist),...].
        """
        # Define variables
        own_histories_list = [-1 for _ in range(self.group_size)]

        own_history_sublists = [x[self.group_id] for x in self.group_interaction_history]
        
        """
        Example group_interaction_history: [ [[-1,cs_for_a1,cs_for_a2], a1_action, a2_action],...]
        
        Example own_history: [ [-1,cs_for_a1,cs_for_a2], [-1,cs_for_a1,cs_for_a2], ...]
        """

        # Go through own agent models
        for l in range(self.group_size):

            # Skip self
            if l == self.group_id:
                continue
            
            # Collect own history for other agent
            own_history_actions = [x[l] for x in own_history_sublists]

            # Collect own history dists for this agent
            own_history_dists = [x[1] for x in self.agent_models[l]._interaction_history]
            # example _interaction_history: [(action,dist), (action,dist),...]

            # Assemble own history as [(action, dist),...]
            own_history = []
            for (action, dist) in zip(own_history_actions, own_history_dists):
                own_history.append((action, dist))

            # Record history in list
            own_histories_list[l] = own_history

        # Return list of own histories
        return own_histories_list

    def collect_agent_histories(self):
        """
        Helper function to collect formatted histories of other agents.
        Collect histories from self.shared_secrets_list.
        Format each history as: [(action,dist), (action,dist),...].
        """

        # Define variables 
        agent_histories_list = [-1 for _ in range(self.group_size)]

        # Go over other agents
        for l in range(self.group_size):

            # Skip self
            if l == self.group_id:
                continue

            # Collect history for other agent
            agent_history_actions = [x[l] for x in self.group_interaction_history]
            # example group_.._history: [ [[-1,cs_for_a1,cs_for_a2], a1_action, a2_action],...]

            # Collect dists for agent from shared secret model
            agent_history_dists = [x[1] for x in self.shared_secrets_list[l]._interaction_history]
            # example _interaction_history: [(action,dist), (action,dist),...]

            # Assemble agent history as [(action,dist),...]
            agent_history = []
            for (action, dist) in zip(agent_history_actions, agent_history_dists):
                agent_history.append((action, dist))

            # Record in list
            agent_histories_list[l] = agent_history

        # Return list
        return agent_histories_list


    def create_mutual_keys(self):
        """Create mutual key between self and other agent(s)."""

        # Define variables
        mutual_keys_list = [-1 for _ in range(self.group_size)]

        own_histories_list = self.collect_own_histories()
        # example: [-1, [history_model1], [history_model2], ...]

        agent_histories_list = self.collect_agent_histories()
        # example: [-1, [history_agent1], [history_agent2], ...]

        # Create mutual key for other agents
        for l in range(self.group_size):

            # Skip self
            if l == self.group_id:
                continue

            # Assemble histories in order of group ID
            assert self.group_id < l, "Error: CS ID > Agent ID!"

            own_history = own_histories_list[l]  # from lth own model 
            agent_history = agent_histories_list[l] # from lth own shared secret 

            input_histories_list = [own_history, agent_history]
            
            # Create mutual key
            mutual_key = session_key_v1(input_histories_list)
            #print("\nmutual_key: ", mutual_key)

            # Record mutual key
            mutual_keys_list[l] = mutual_key

        # Return created mutual keys
        return mutual_keys_list

    def create_group_key(self):
        """Create group key over own model histories and other agent histories."""

        # Define variables
        own_histories_list = self.collect_own_histories()
        # example: [-1, [history_model1], [history_model2], ...]

        agent_histories_list = self.collect_agent_histories()
        # example: [-1, [history_agent1], [history_agent2], ...]

        # Trim lists to prepare for key gen function
        own_histories_list_trim = own_histories_list[1:]

        agent_histories_list_trim = agent_histories_list[1:]

        # Build group key in format: 'model1*model2*model3*agent1*agent2*agent3'
        all_histories_list = own_histories_list_trim[:] + agent_histories_list_trim[:]

        group_key = session_key_v1(all_histories_list)
        #print("\ngroup_key: ", group_key)
        
        # Return group key
        return group_key

    def encrypt_group_key(self, group_key):
        """
        Encrypt group key by all mutual keys.

        Return: list of differently encrypted group key.
        """
        encrypted_group_keys = []

        # For each other agent, encrypt with its mutual key 
        for l in range(self.group_size):

            if l == self.group_id:
                continue

            # Encrypt group key with mutual key for this agent

            # Derive cipher key from mutual key (salt is agent group id)
            cipher_key = pbkd(password=str(self.mutual_keys_list[l]), salt=l)

            # Build ciphersuite from cipher key and encrypt group key
            group_key_encr = encrypt(cipher_key, plaintext=str(self.group_key))

            # Append to list
            encrypted_group_keys.append(group_key_encr)

        # Return list
        #print("\nencrypted_group_keys: ", encrypted_group_keys)
        return encrypted_group_keys
        

    def broadcast_group_keys(self, encrypted_group_keys, network_protocol='tcp'):
        """
        Broadcast encrypted group keys to agents.
        Other agents receive group key encrypted by their mutual key.

        Args:
        encrypted_group_keys: list of group key as encrypted by mutual keys.
                            Keys are of type bytes
                            i.e. [ m_key1(g_key), m_key2(g_key) ]
        """
        send_sock_list = []
        
        # Get list of sockets for other agents
        for l in range(self.group_size):

            if l == self.group_id:
                continue

            send_sock_list.append(self.sockets_list[l][0])
            # example self.sockets_list[l]: (send_sock, recv_socks)

        # Convert encrypted group keys from bytes to strings
        encrypted_group_keys_str = []
        for key in encrypted_group_keys:
            if type(key) == bytes:
                encrypted_group_keys_str.append(str(key.decode()))

        # Broadcast encrypted group keys 
        send_over_sockets(send_sock_list, encrypted_group_keys_str, network_protocol)

    def local_setup_key(self):
        """
        1.Create mutual key for agent(s)
        2. If group size more than 2 agents, create group key
        2a. Create group key over all agents
        2b. For each agent, encrypt group key with mutual key
        """

        # 1. Create and set mutual key for agent(s)
        mutual_keys_list = self.create_mutual_keys()
        self.set_mutual_keys_list(mutual_keys_list)

        # If group size > 2, then create, set, and encrypt group key 
        if self.group_size > 2:
            # 2a. Create group key
            group_key = self.create_group_key()

            # Set group key
            self.set_group_key(group_key)

            # 2b. Encrypt group key by different mutual keys
            encrypted_group_keys = self.encrypt_group_key(group_key)

        return encrypted_group_keys

    def setup_key(self):
        """
        1.Create mutual key for agent(s)
        2. If group size more than 2 agents, create group key
        2a. Create group key over all agents
        2b. For each agent, encrypt group key with mutual key
        2c. For each agent, transmit encrypted group key
        """

        # 1. Create and set mutual key for agent(s)
        mutual_keys_list = self.create_mutual_keys()
        self.set_mutual_keys_list(mutual_keys_list)

        # If group size > 2, then create, set, encrypt, and broadcast group key 
        if self.group_size > 2:
            # 2a. Create group key
            group_key = self.create_group_key()

            # Set group key
            self.set_group_key(group_key)

            # 2b. Encrypt group key by different mutual keys
            encrypted_group_keys = self.encrypt_group_key(group_key)

            # 2c. Broadcast encrypted group keys
            self.broadcast_group_keys(encrypted_group_keys)


    def print_agent_model(self):
        """Print agent models (may be identical)."""
        
        print("\nPrinting agent models: ")
        for agent_model, id in zip(self.agent_models, range(self.group_size)):
            print("\nagent model for group id", id, ":", agent_model)

    def print_model_history(self):
        """Print interaction history for all behavioral models."""

        print("\nPrinting agent model histories: ")
        
        for agent_model, id in zip(self.agent_models, range(self.group_size)):

            if id == self.group_id:
                print("\nmodel id: ", id, " history: -1")
                continue

            print("\nmodel id: ", id, " history: ", agent_model._interaction_history)

    
class CentralizedSystemAgent(SystemAgent):
    """An agent in a centralized setting (not central server)."""

    def __init__(self, info_dict, secrets_info_dict, statedict_models, statedict_secrets,
                    max_interaction_length):
        """Create agent."""

        # Use inherited constructor
        super(CentralizedSystemAgent, self).__init__(info_dict)

        # Set new variables
        self.is_central_server = info_dict['is_central_server']
        self.central_server_group_id = info_dict['central_server_group_id']
        self.reuse_agent_model = info_dict['reuse_agent_model']

        if info_dict['auth_method'] == 'hypothesis_test':
            self.alpha = info_dict['alpha']

        self.group_interaction_history = -1

        # Set agent model
        self.set_agent_model(info_dict, statedict_models)

        # Set shared secrets using inherited method
        self.set_shared_secrets(secrets_info_dict, statedict_secrets)

        # Set received actions list as [[-1,..,-1], -1, -1]
        # The sole sublist is sized interaction len, to be filled during interaction
        self.set_received_actions_list(self.group_size, max_interaction_length)

        # Initialize mutual key
        self.mutual_key = -1

    def set_mutual_key(self, mutual_key):
        """Set the mutual key for communication with central server."""
        self.mutual_key = mutual_key

    def set_group_interaction_history(self, history):
        """
        Set group interaction history.

        Args:
        history: list of sublists for each timestep in interaction
        
        Format for agent id=1: 
                [ 
                    [cs_action, a1_action, -1],
                    [cs_action, a1_action, -1],
                    ...
                ]
        """
        self.group_interaction_history = history


    def set_received_actions_list(self, group_size, max_interaction_length):
        """
        Set received actions list by max interaction length.
        Format of list: [[-1,..,-1], -1, -1]
        """

        # Define variables
        received_actions_list = [-1 for _ in range(group_size)]

        # Create set-length sublist for central server
        set_length_sublist = [-1 for _ in range(max_interaction_length)]

        received_actions_list[self.central_server_group_id] = set_length_sublist

        # Set for self
        self.received_actions_list = received_actions_list


    def set_agent_model(self, info_file, statedict_models):
        """System agent in centralized setting only uses single model.

        Args:
        info_file: json file with following example format
            {
                "group_id": 
                "reuse_agent_model":
                ...

                "models_list": [-1, 
                                {"unique_seed":0, "model_type": 'pdt', ...}, 
                                -1]
            }
        statedict_models: list with single available .pth file of trained neural net
        """

        # Only available model kept at agent group index
        agent_model_dict = info_file["models_list"][self.group_id]

        statedict = statedict_models[self.group_id]
        self.agent_model = self.create_agent_model(agent_model_dict, statedict)

    def set_socket_info(self, socket_info_list, path_agent_files):
        """Record IP address and port numbers to connect to central server only."""
        
        # Example: [[(central_server_ip, port25),(central_server_ip, port26)], -1, -1]
        self.socket_info_list = socket_info_list

        # Update agent info json file (i.e. 'agent_info_files/agent1_info.txt')
        agent_info_filepath = path_agent_files + '/agent'+str(self.group_id)+'_info.txt'
        
        update_json_file(agent_info_filepath, key='socket_info_list', value=socket_info_list)

    def connect_tcp_sockets(self, info_dict):
        
        """(Remote use) Setup pair of client-style sockets and connect them ...
        ...to central server's sockets."""

        # Set variables
        socket_info_list = info_dict['socket_info_list']
        cs_gid = info_dict['central_server_group_id']

        # Get socket info [(cs_ipaddr, port1), (cs_ipaddr, port2)] for central server
        central_server_socket_info = socket_info_list[cs_gid]

        # Central server sockets are server-style, other agents have client-style
        setup_function = setup_client_style_socket

        # Get connection info for each socket (IP addresses are the same)
        (ipaddr1, port1) = central_server_socket_info[0]  # (cs_ipaddr, port1)
        (ipaddr2, port2) = central_server_socket_info[1]  # (cs_ipaddr, port2)

        """ 
        From joint port numbers [(ip,port1), (ip, port2)], central server
        will send over port1 and receive over port2. Other agents
        send over port2 and receive over port1.

        This is why thread labels designate thread1 to receive over port1
        and thread2 to send over port2.
        """

        # Labels for each thread socket's function in the connection
        thread1_type = 'receive'
        thread2_type = 'send'

        # Prepare socket info tuples for function
        socket1_info_tuple = (ipaddr1, port1, thread1_type)
        socket2_info_tuple = (ipaddr2, port2, thread2_type)

        # Get prepared sockets as [send_socket, recv_socket]
        sockets_for_central_server = connect_socket_pairs(setup_function,
                                    socket1_info_tuple, socket2_info_tuple)

        # Update agent's sockets list (example format: [[send_sock,recv_sock],-1,-1] )
        self.sockets_list[cs_gid] = sockets_for_central_server

        print("\nself.sockets_list: ", self.sockets_list)

    def local_interact(self, t, prev_actions):
        """
        Perform local interaction for this timestep.

        1. Get next action(s)
        2. Update shared secret(s) 

        Args: 
        t: current timestep in interaction process
        prev_actions: sublist with group actions from previous timestep 
        """
        
        # 1. Get next action from model, which also stores the (action,dist)
        own_action = self.model_next_action(t, prev_actions)

        # 2. Update shared secret which also stores the dist
        self.secrets_next_action(t, prev_actions)

        # Return just own action for this timestep
        return own_action


    def interact(self, t, prev_actions, network_protocol):
        """
        Perform remote interaction for this timestep.

        1. Get next action(s)
        2. Update shared secret(s) 
        3. Transmits action(s)
        4. Receive action(s).

        Args: 
        t: current timestep in interaction process
        prev_actions: sublist with group actions from previous timestep 
        """

        # Define variables
        cs_gid = self.central_server_group_id

        # 1. Get next action from model, which also stores the (action,dist)
        own_action = self.model_next_action(t, prev_actions)

        # 2. Update shared secret which also stores the dist
        self.secrets_next_action(t, prev_actions)

        # 3. Transmit action to central server
        action_broadcast_list = [-1 for _ in range(self.group_size)]
        action_broadcast_list[cs_gid] = own_action

        # example: [action_for_cs, -1, -1]
        self.broadcast_actions(action_broadcast_list, t, network_protocol)

        # 4. Collect received action from central server for this timestep

        # blocking call: collect from receive sockets and store in self
        self.check_receive_sockets(t)

        # example received_actions_list: [[-1,...,-1], -1, -1]
        cs_action = self.received_actions_list[cs_gid][t]

        # Assemble interaction history for this timestep
        group_actions_list = [-1 for _ in range(self.group_size)]

        group_actions_list[self.group_id] = own_action
        group_actions_list[cs_gid] = cs_action

        # Return interaction history for this timestep
        return group_actions_list

    def model_next_action(self,t, group_prev_actions):
        """
        Behavioral model provides and personally stores next action & dist.
        
        Args:
        t: current timestep
        group_prev_actions: group actions from previous timestep i.e. [cs_action, a1, -1]
        """
        # Define variables
        cs_gid = self.central_server_group_id

        if self.agent_model._type == 'pdt':

            # PDTAgent conditions on most recent other agent action
            if t == 0:
                inputs = []
            else:
                inputs = [group_prev_actions[cs_gid]]

            # Agent model stores then provides action,dist
            (action, dist) = self.agent_model.act(t, inputs)

        elif self.agent_model._type == 'neuralnet':

            # Neuralnets conditions on: [server_i, client_i]
            if t == 0:
                inputs = []
            else: 
                inputs = [group_prev_actions[cs_gid][:], group_prev_actions[self.group_id][:]]

            # Agent model stores then provides integer action, dist
            (logits, action) = self.agent_model.act(t,inputs,self.action_space_size)

        else:
            exit("Exiting: Agent model not of supported types.")

        # Return integer action
        return action

    def secrets_next_action(self, t, group_prev_actions):
        """
        Have shared secrets update with next action and dist.
        Only the dists are significant for later key computation.

        Args:
        t: current timestep
        group_prev_actions: group actions from previous timestep i.e. [cs_action, a1, -1]
        """

        # Define variables
        cs_gid = self.central_server_group_id

        if self.shared_secrets_list[cs_gid]._type == 'pdt':

            # PDTAgent conditions on most recent other agent action
            if t == 0:
                inputs = []
            else:   # provide central server secret own client action as input
                inputs = [group_prev_actions[self.group_id]]

            # Secret model stores own dist,action
            (_, _) = self.shared_secrets_list[cs_gid].act(t, inputs)

        elif self.shared_secrets_list[cs_gid]._type == 'neuralnet':

            """
            Neuralnet conditions on: [server_i, client_i], not on own actions.
            The shared secret model just needs to self-store expected dists for
            ...later key computation. Its action output is not required.
            """
            if t == 0:
                inputs = []
            else: 
                inputs = [group_prev_actions[cs_gid], group_prev_actions[self.group_id]]

            # Secret model stores own action, dist
            (_, _) = self.shared_secrets_list[cs_gid].act(t,inputs,self.action_space_size)

        else:
            exit("Exiting: Agent model not of supported types.")

    
    def broadcast_actions(self, actions_for_group, t, network_protocol):
        """
        Send own selected action to central server at this timestep in interaction.

        Args:
        actions_for_group: list of own selected actions at this timestep in interaction.
                          i.e. [action_for_cs, -1, -1]
        t: current timestep in remote interaction process.
        """

        # Get send socket for central server from tuple: (send_sock, recv_sock)
        send_sock = self.sockets_list[self.central_server_group_id][0]

        # Get action for central server
        action = actions_for_group[self.central_server_group_id]

        # Send action to central server
        send_actions_over_sockets([send_sock], [action], t, network_protocol)

    def check_receive_sockets(self, t):
        """
        Collect received actions from other agents during remote interaction.
        Update self.received_actions_list with all received actions.
        """
        #print("\nInside check_receive_sockets!")

        # Define variables
        cs_gid = self.central_server_group_id

        # Get receive socket for central server from tuple: (send_sock, recv_sock)
        receive_sock = self.sockets_list[self.central_server_group_id][1]

        # Place blocking call in script until action appears at recv socket
        (msg_list, msg_list_tuples) = receive_action_from_socket(receive_sock)

        # Example msg_list: ['3F_t0_C', '2F_t1_C'] if more than one message received
        # Example time-ordered msg_list_tuples: [(3,0), 2,1]

        # Ensure only a single msg in the list is for this current timestep
        current_time_str = 't' + str(t) # i.e. 't3' 
        msg_list_current_timestep = [current_time_str in x for x in msg_list]

        assert msg_list_current_timestep.count(True) == 1, "Only one msg should have current time!"

        # Place in central server sublist at the index for timestep 
        # Format: [cs_t1, cs_t2,...cd_T], -1, -1)

        for msg_tuple in msg_list_tuples: # i.e. [(3,0), (2,1)]
            
            msg_action = msg_tuple[0]
            msg_time = msg_tuple[1]

            self.received_actions_list[cs_gid][msg_time] = int(msg_action)


    def authenticate(self, group_history, max_history_len): #, agent_info_dict):
        """Authenticate the central server.

        Args:
        group_interaction_history: [[cs_0,a1_0,-1], [cs_1,a1_1,-1],...]
        max_history_len: length of complete interaction history
        """

        # Define variables
        cs_gid = self.central_server_group_id
        
        # Run authentication test
        if self.auth_method == 'hypothesis_test':

            alpha = self.alpha #agent_info_dict['alpha']

            # Get action of unknown agent from group history
            unknown_agent_actions = [x[cs_gid] for x in group_history]

            # Get dists of shared secret for central server
            secret_history = self.shared_secrets_list[cs_gid]._interaction_history
            # Example secret_history: [(action,dist), (action,dist),...]

            known_agent_dists = [x[1] for x in secret_history]

            # Get pvalues from hypothesis test
            (pvalue_reg, pvalue_ratio) = hypothesis_test_v1(
                                            unknown_agent_actions, 
                                            known_agent_dists, 
                                            update_times = [max_history_len-1],
                                            num_actions = self.action_space_size)
            # Determine authentication success
            result = True if pvalue_ratio >= alpha else False
            self.auth_results[cs_gid] = result

        elif self.auth_method == 'classifier':
            pass
        else:
            exit("Exiting: Specified authentication method not supported.")

    def create_mutual_key(self):
        """Create mutual session key with central server."""

        # Define variables
        cs_gid = self.central_server_group_id

        # Collect own history
        own_history_actions = [x[self.group_id] for x in self.group_interaction_history]
        # example group_interaction_history: [[cs_action, a1_action, -1], ...]

        own_history_dists = [x[1] for x in self.agent_model._interaction_history]

        # Assemble own history as [(action, dist),...]
        own_history = []
        for (action, dist) in zip(own_history_actions, own_history_dists):
            own_history.append((action, dist))

        # Collect history for central server
        cs_history_actions = [x[cs_gid] for x in self.group_interaction_history]

        cs_history_dists = [x[1] for x in self.shared_secrets_list[cs_gid]._interaction_history]

        # Assemble central server history as [(action,dist),...]
        cs_history = []
        for (action, dist) in zip(cs_history_actions, cs_history_dists):
            cs_history.append((action, dist))

        # Assemble histories in order of group ID
        assert self.group_id > self.central_server_group_id, "Group ID < CS ID!"

        input_histories_list = [cs_history, own_history]

        # Create mutual key
        mutual_key = session_key_v1(input_histories_list)
        #print("\nmutual_key: ", mutual_key)

        # Return mutual key
        return mutual_key

    def receive_group_key(self):
        """
        Collect encrypted group key sent by central server.

        Return encrypted (by mutual key) group key as type str.
        """

        # Define variables
        cs_gid = self.central_server_group_id

        # Get recv socket for central server 
        recv_sock = self.sockets_list[cs_gid][1]
        # example sockets_list[cs_gid]: (send_sock, recv_sock)

        # Block until socket receives, then collect as string
        group_key_encr = receive_from_socket(recv_sock)
        #print("\ngroup_key_encr: ", group_key_encr)

        # Return encrypted group key
        return group_key_encr

    def decrypt_group_key(self, group_key_encr):
        """
        Decrypt group key using mutual key.

        Args: 
        group_key_encr: group key of type str
                        Note: group key was encrypted by mutual key
        """

        # Convert to type bytes as required by Fernet suite
        if type(group_key_encr) != bytes:
            group_key_encr = group_key_encr.encode()

         # Derive cipher key from mutual key (salt is group id)
        cipher_key = pbkd(password=str(self.mutual_key), salt=self.group_id)

        # Build ciphersuite from cipher key and decrypt group key
        group_key_decr = decrypt(cipher_key, ciphertext=group_key_encr)

        # Return decrypted group key
        return group_key_decr

    def local_setup_key(self):
        """Create mutual key."""

        # Create and set mutual key
        mutual_key = self.create_mutual_key()

        self.set_mutual_key(mutual_key)

    def setup_key(self):
        """Create mutual key (and receive group key from central server)."""
        
        # Create and set mutual key
        mutual_key = self.create_mutual_key()
        self.set_mutual_key(mutual_key)

        # If other agents, receive and decrypt group key from server
        if self.group_size > 2:
            
            # Block until group key received 
            group_key_encr = self.receive_group_key() # type str

            # Decrypt group key with mutual key
            group_key = self.decrypt_group_key(group_key_encr) # type str

            # Set group key
            self.set_group_key(group_key)

    def print_agent_model(self):
        """Print agent model."""
        print("\nPrinting agent model: ", self.agent_model)

    def print_model_history(self):
        """Print history for behavioral model."""
        print("\nPrinting agent model history: ")
        print(self.agent_model._interaction_history)
    

class DecentralizedSystemAgent(SystemAgent):
    """An agent in a decentralized setting."""

    def __init__(self, info_dict, secrets_info_dict, statedict_models, statedict_secrets,
                    max_interaction_length):
        """Create agent."""

        # Use inherited constructor
        super(DecentralizedSystemAgent, self).__init__(info_dict)

        # Set new variables
        self.group_interaction_history = -1

        if info_dict['auth_method'] == 'hypothesis_test':
            self.alpha = info_dict['alpha']

        # Set agent model
        self.set_agent_model(info_dict, statedict_models)

        # Set shared secrets using inherited method
        self.set_shared_secrets(secrets_info_dict, statedict_secrets)

        # Set received actions list as [-1, [-1,..,-1], [-1,..,-1]] if a0
        # Each sublist is sized interaction len, to be filled during interaction
        self.set_received_actions_list(self.group_size, max_interaction_length)
        
    def set_group_interaction_history(self, history):
        """
        Set group interaction history.

        Args:
        history: list of sublists for each timestep in interaction
        
        Format for agent id=0: 
            [
                [a0_action, a1_action, a2_action],
                [a0_action, a1_action, a2_action],
                ...
            ]
        """
        self.group_interaction_history = history

    def set_received_actions_list(self, group_size, max_interaction_length):
        """
        Set received actions list by max interaction length.
        Format of list if a0: [-1, [-1,...,-1], [-1,...,-1]]
        """

        # Define variables
        received_actions_list = [-1 for _ in range(group_size)]

        # Create set-length sublist for other agents
        for l in range(self.group_size):

            if l == self.group_id:
                continue

            received_actions_list[l] = [-1 for _ in range(max_interaction_length)]

        # Set for self
        self.received_actions_list = received_actions_list


    def set_agent_model(self, info_file, statedict_models):
        """System agent in decentralized setting only uses single model.

        Args:
        info_file: json file with following example format
            {
                "group_id": 
                "reuse_agent_model":
                ...

                "models_list": [{"unique_seed":0, "model_type": 'pdt', ...}, 
                                -1, 
                                -1]
            }
        statedict_models: list with single available .pth file of trained neural net
        """

        # Only available model kept at agent group index
        agent_model_dict = info_file["models_list"][self.group_id]

        statedict = statedict_models[self.group_id]
        self.agent_model = self.create_agent_model(agent_model_dict, statedict)

    def set_socket_info(self, socket_info_list, path_agent_files):
        """Record IP address and port numbers to connect to other agents."""
        
        # Example: [-1, [(a1_ip, port26), (a1_ip, port27)], [(a2_ip, port35), (a2_ip, port36)] ]
        self.socket_info_list = socket_info_list

        # Update agent info json file (i.e. 'agent_info_files/agent0_info.txt')
        agent_info_filepath = path_agent_files + '/agent'+str(self.group_id)+'_info.txt'
        
        update_json_file(agent_info_filepath, key='socket_info_list', value=socket_info_list)

    def connect_tcp_sockets(self, info_dict):
        """
        (Remote use) Setup pair of sockets for all other agents...
        then connect them to remote sockets.
        """

        # Format: [-1, [(a1_ip,port26),(a1_ip,port27)], [(a2_ip,port35),(a2_ip,port36)]]
        socket_info_list = info_dict['socket_info_list']

        # For each agent in group, connect my sockets to theirs
        for l in range(self.group_size):

            # Skip myself
            if l == self.group_id:
                continue

            # If own group_id < this other group_id, use server-style sockets
            if self.group_id < l:
                setup_function = setup_server_style_socket
            else:
                setup_function = setup_client_style_socket

            # Get connection info for each socket (IP addresses are the same)
            (ipaddr1, port1) = socket_info_list[l][0]  # (a1_ipaddr, port1)
            (ipaddr2, port2) = socket_info_list[l][1]  # (a1_ipaddr, port2)

            """
            From joint port numbers [(ip,port1), (ip, port2)], agent with
            lower group_id will send over port1 and receive over port2. 
            Other agent of higher group_id will send over port2 and 
            receive over port1. 

            This is why thread labels may designate thread1 to either
            send or receive over port1 and thread2 to either 
            send or receive over port2, depending on agents' group_id.
            
            Example for group with agents [a0,a1,a2]:

              | a0              | a1                | a2
             -------------------------------------------------------
            a0| x               | send over port1   | send over port1
              |                 | recv over port2   | recv over port2
             -------------------------------------------------------
            a1| recv over port1 | x                 | send over port1
              | send over port2 |                   | recv over port2
             -------------------------------------------------------
            a2| recv over port1 | recv over port1   | x
              | send over port2 | send over port2   | 
            
            """

            # Labels for each thread socket's function in the connection
            if self.group_id < l:
                thread1_type = 'send'
                thread2_type = 'receive'
            else:
                thread1_type = 'receive'
                thread2_type = 'send'

            # Prepare socket info tuples for function
            socket1_info_tuple = (ipaddr1, port1, thread1_type)
            socket2_info_tuple = (ipaddr2, port2, thread2_type)

            # Get prepared sockets as [send_socket, recv_socket]
            sockets_for_agent = connect_socket_pairs(setup_function, 
                                        socket1_info_tuple, socket2_info_tuple)

            # Update sockets list 
            # Example format: [-1,[send_sock,recv_sock],[send_sock,recv_sock]])
            self.sockets_list[l] = sockets_for_agent

            print("\nself.sockets_list: ", self.sockets_list)

    def local_interact(self, t, prev_actions):
        """
        Perform local interaction for this timestep.

        1. Get next action(s)
        2. Update shared secret(s) 

        Args: 
        t: current timestep in interaction process
        prev_actions: sublist with group actions from previous timestep 
                    i.e. [a0_action, a1_action, a2_action]
        """

        # 1. Get next action, which model stores itself
        own_action = self.model_next_action(t, prev_actions)

        # 2. Update shared secrets, which will store their dists
        self.secrets_next_action(t, prev_actions)

        # Return just own action for this timestep
        return own_action

    def interact(self, t, prev_actions, network_protocol):
        """
        Perform remote interaction for this timestep.

        1. Get next action(s)
        2. Update shared secret(s) 
        3. Transmits action(s)
        4. Receive action(s).

        Args: 
        t: current timestep in interaction process
        prev_actions: sublist with group actions from previous timestep 
                    i.e. [a0_action, a1_action, a2_action]
        """
        #print("\nInside interact() with prev_actions: ", prev_actions)

        # 1. Get next action, which model stores itself
        own_action = self.model_next_action(t, prev_actions)
        #print("\nown_action: ", own_action)

        # 2. Update shared secrets, which will store their dists
        self.secrets_next_action(t, prev_actions)

        # 3. Broadcast the same action to all other agents in group
        action_broadcast_list = [own_action for _ in range(self.group_size)]
        action_broadcast_list[self.group_id] = -1

        # example: [-1, own_action, own_action]

        self.broadcast_actions(action_broadcast_list, t, network_protocol)

        # 4. Collect received actions from receive sockets and store in self

        # blocking call: wait until all expected actions are received
        self.check_receive_sockets(t)

        # example of self.received_actions_list: [-1, [-1,..,-1], [-1,..,-1]]

        # collect received actions from self

        #print("\nBuilding group_actions_list")
        group_actions_list = [-1 for _ in range(self.group_size)]
        group_actions_list[self.group_id] = own_action

        for l in range(self.group_size):
            if l == self.group_id:
                continue

            received_actions_sublist = self.received_actions_list[l]

            # get action from other agent at current timestep
            group_actions_list[l] = received_actions_sublist[t]

        """
        example of group_actions_list (group actions at this timestep): 

        [ a0_action, a1_action, a2_action]
        """
        #print("\ngroup_actions_list: ", group_actions_list)

        # return group interaction history for this timestep
        return group_actions_list

    def model_next_action(self,t, group_prev_actions):
        """
        Behavioral model provides and personally stores next action & dist.
        
        Args:
        t: current timestep
        group_prev_actions: group actions from previous timestep 
                            i.e. [a0_action, a1_action, a2_action]
        """
        #print("\nInside model_next_action with:")
        #print("t: ", t, " and group_prev_actions: ", group_prev_actions)

        if self.agent_model._type == 'multitree':
            inputs = -1 # initialize

            if t == 0: 
                inputs = []
            else:
                # Condition on id-ordered actions of other agents
                group_prev_actions_copy = group_prev_actions[:]
                group_prev_actions_copy.pop(self.group_id) 
                inputs = group_prev_actions_copy

                # Example if a0: [a1_action, a2_action]

            # Multitree act() will convert inputs to branching code
            #print("\nCall agent_model.act() with inputs: ", inputs)
            (action, dist) = self.agent_model.act(t, inputs)

        elif self.agent_model._type == 'neuralnet':

            if t == 0: 
                inputs = []
            else:
                # Condition on entire interaction history from t-1
                inputs = group_prev_actions[:]
                # example if am a0: [a0_action, a1_action, a2_action]

            # Agent model stores then provides integer action, dist
            (logits, action) = self.agent_model.act(t,inputs,self.action_space_size)
    
        else:
            exit("Exiting: Agent model not of supported types.")

        # Return integer action
        return action

    def secrets_next_action(self, t, group_prev_actions):
        """
        Have shared secrets update with next action and dist.
        Only the dists are significant for later key computation.

        Args:
        t: current timestep
        group_prev_actions: group actions from previous timestep 
                            i.e. [a0_action, a1_action, a2_action]
        """
        #print("\nInside secrets_next_action() with t: ", t, " and group_prev_actions: ", group_prev_actions)

        for l in range(self.group_size):

            if l == self.group_id:
                continue

            if self.shared_secrets_list[l]._type == 'multitree':
                
                if t == 0:
                    inputs = []
                else:
                    # Secret model conditions on public actions of other agents
                    group_prev_actions_copy = list(group_prev_actions[:]) # shallow copy
                    group_prev_actions_copy.pop(l)
                    #print("group_prev_actions_copy with pop: ", group_prev_actions_copy)
                    #print("original group_prev_actions: ", group_prev_actions)
                    inputs = group_prev_actions_copy[:]
                    # example: secret for a1 takes in [a0_action, a2_action]

                (_,_) = self.shared_secrets_list[l].act(t, inputs)

            elif self.shared_secrets_list[l]._type == 'neuralnet':
                
                if t == 0:
                    inputs = []
                else:
                    # Condition on entire interaction history at time: t-1
                    inputs = group_prev_actions[:]

                # Secret model will self-store action and dist
                (_,_) = self.shared_secrets_list[l].act(t,inputs,self.action_space_size)

            else:
                exit("Exiting: shared secrets list type incorrectly specified.")

    def broadcast_actions(self, actions_for_group, t, network_protocol):
        """
        Send same selected action to other agents at this timestep in interaction.

        Args:
        actions_for_group: list of same single action at this timestep in interaction.
                          i.e. for agent a0: [-1, action_for_others, action_for_others]
        t: current timestep in remote interaction process.
        """

        # Get list of send sockets, and own actions for other agents
        send_sock_list = []
        actions_list = []

        for l in range(self.group_size):

            if l == self.group_id:
                continue

            # Get send socket for agent from tuple: (send_sock, recv_sock)
            send_sock = self.sockets_list[l][0]
            send_sock_list.append(send_sock)

            # Get own action for this agent
            action = actions_for_group[l]
            actions_list.append(action)

        # Send collected actions over collected send sockets
        send_actions_over_sockets(send_sock_list, actions_list, t, network_protocol)

    def check_receive_sockets(self, t):
        """
        Collect received actions from other agents during remote interaction.
        Update self.received_actions_list with received action.
        """
        #print("\nInside check_receive_sockets!")

        for l in range(self.group_size):

            if l == self.group_id:
                continue

            # Get receive socket for central server from tuple: (send_sock, recv_sock)
            receive_sock = self.sockets_list[l][1]

            # Place blocking call in script until action appears at recv socket
            (msg_list, msg_list_tuples) = receive_action_from_socket(receive_sock)

            # Example msg_list: ['3F_t0_C', '2F_t1_C'] if more than one message received
            # Example time-ordered msg_list_tuples: [(3,0), 2,1]

            # Ensure only a single msg in the list is for this current timestep
            current_time_str = 't' + str(t) # i.e. 't3' 
            msg_list_current_timestep = [current_time_str in x for x in msg_list]

            assert msg_list_current_timestep.count(True) == 1, "Only one msg should have current time!"

            # Place in agent's sublist at the index for timestep 
            # Format if a0: [-1, [a1_t1, a1_t2,...,a1_T], [a2_t1, a2_t2,...,a2_T]]

            for msg_tuple in msg_list_tuples: # i.e. [(3,0), (2,1)]
                
                msg_action = msg_tuple[0]
                msg_time = msg_tuple[1]

                self.received_actions_list[l][msg_time] = int(msg_action)

    def authenticate(self, group_history, max_history_len): #, agent_info_dict):
        """Authenticate other agents in the group.

        Args:
        group_history: [[a0_0,a1_0,a2_0], [a0_1,a1_1,a2_1],...]
        max_history_len: length of complete interaction history
        """

        for l in range(self.group_size):

            # Run authentication test (same for all others)
            if self.auth_method == 'hypothesis_test':
                if l == self.group_id:
                    continue

                # Define variables
                alpha = self.alpha #agent_info_dict['alpha']
                
                # Get action of unknown agent from group history
                unknown_agent_actions = [x[l] for x in group_history]

                # Get dists of shared secret for central server
                secret_history = self.shared_secrets_list[l]._interaction_history
                # Example cs_secret_history: [(action,dist), (action,dist),...]

                known_agent_dists = [x[1] for x in secret_history]

                # Get pvalues from hypothesis test
                (pvalue_reg, pvalue_ratio) = hypothesis_test_v1(
                                                unknown_agent_actions, 
                                                known_agent_dists, 
                                                update_times = [max_history_len-1],
                                                num_actions = self.action_space_size)
                # Determine authentication success
                result = True if pvalue_ratio >= alpha else False
                self.auth_results[l] = result

            elif self.auth_method == 'classifier':
                pass
            else:
                exit("Exiting: Specified authentication method not supported.")

    def create_group_key(self):

        # Define variables
        input_histories_list = []

        # Collect history for each agent in group (self included)
        for l in range(self.group_size):

            # Collect own history as: [(action,dist), ...]
            if l == self.group_id:

                history_actions = [x[self.group_id] for x in self.group_interaction_history]

                history_dists = [x[1] for x in self.agent_model._interaction_history]
            else:

                history_actions = [x[l] for x in self.group_interaction_history]

                history_dists = [x[1] for x in self.shared_secrets_list[l]._interaction_history]

            # Organize history as [(action,dist),...]
            agent_history = []

            for (action, dist) in zip(history_actions, history_dists):
                agent_history.append((action, dist))

            # Include in list of agent histories
            input_histories_list.append(agent_history)

        # Create group key (ordered by agent ID)
        group_key = session_key_v1(input_histories_list)
        #print("\ngroup_key: ", group_key)

        # Return group key
        return group_key

    def set_group_key(self, group_key):
        """Set member variables group_key."""
        self.group_key = group_key

    def setup_key(self):
        """Create and set group key for encryption/decryption."""

        # Create group key
        group_key = self.create_group_key()

        # Set group key
        self.set_group_key(group_key)

    def local_setup_key(self):
        """Create and set group key for encryption/decryption."""

        # Create group key
        group_key = self.create_group_key()

        # Set group key
        self.set_group_key(group_key)


    def print_agent_model(self):
        """Print agent model."""
        print("\nPrinting agent model: ", self.agent_model)

    def print_model_history(self):
        """Print history for behavioral model."""
        print("\nPrinting agent model history: ")
        print(self.agent_model._interaction_history)
