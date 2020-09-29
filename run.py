"""Run script to set up multi-agent system and run on remote machines."""

from cli import arg_parser
from file_handling import collect_json_from_dir, collect_neuralnets_from_dir
from multi_agent_systems import CentralizedSystem, DecentralizedSystem
from multi_agent_systems import CentralizedLocalSystem, DecentralizedLocalSystem
from remote_envs_setup import RemoteEnvsSetup

def run():
    """Create local multi-agent system and run agents in remote envs."""    
    args = arg_parser()
 
    # Set variables for multi-agent system
    system_type = str(args.system_type)
    group_size = args.group_size
    path_agent_files = str(args.path_agent_files)
    path_shared_secrets = str(args.path_shared_secrets)
    path_neuralnet_models = str(args.path_neuralnet_models)

    # Collect json objects for agents and for shared secrets
    agents_json = collect_json_from_dir(args.path_agent_files, group_size)

    shared_secrets_json = collect_json_from_dir(args.path_shared_secrets, group_size)

    # Collect available neural net agent models from nested subdirs
    # List will contain -1 if statedict file not listed in directory
    agent_model_subdirs = []
    for l in range(group_size):
        agent_model_subdirs.append('agent' + str(l) + '_neuralnet_models')
    (agent_model_statedicts, _) = collect_neuralnets_from_dir(args.path_neuralnet_models, 
                                                        agent_model_subdirs,
                                                        group_size)

    # Collect available neural net shared secrets from nested subdirs
    agent_secret_subdirs = []
    for l in range(group_size):
        agent_secret_subdirs.append('agent' + str(l) + '_secret_models')
    (shared_secret_statedicts, _) = collect_neuralnets_from_dir(args.path_secret_neuralnets,
                                                            agent_secret_subdirs, 
                                                            group_size)
    # Print all retrieved objects
    for agent in agents_json:
        print("\nagent info: ", agent.values())

    for secret in shared_secrets_json:
        print("\nsecret info: ", secret.values())

    print("\nPrinting retrieved models:")
    for l in range(len(agent_model_statedicts)):
        print("\nl: ", l)
        for m in agent_model_statedicts[l]:
            if m == -1:
                print("model: ", m)
            else:
                print("model: ", m['model.0.weight'][0])

    print("\nPrinting retrieved secrets:")
    for l in range(len(shared_secret_statedicts)):
        print("\nl: ", l)
        for s in shared_secret_statedicts[l]:
            if s == -1:
                print("secret: ", s)
            else:
                print("secret: ", s['model.0.weight'][0])

    # Create multi-agent system, along with agents
    if system_type == 'centralized':
        system = CentralizedSystem(group_size, agents_json, shared_secrets_json, 
                                    agent_model_statedicts, shared_secret_statedicts)
    else:
        system = DecentralizedSystem(group_size, agents_json, shared_secrets_json, 
                                    agent_model_statedicts, shared_secret_statedicts)

    # Print all agent's behavioural models and shared secrets
    system.print_info('agent_model')
    system.print_info('agent_secrets')


    if args.run_remotely == 'yes':

        # Set up socket information (ip address/port number) for the system
        system.setup_socket_info(path_agent_files)
        system.print_info('agent_sockets')


        # Prepare remote machines to run agents 
        remote_envs_setup = RemoteEnvsSetup(system, 
                                args.path_agent_files, args.path_shared_secrets,
                                args.path_neuralnet_models, args.path_secret_neuralnets)

        # Create system settings file for remote system
        remote_envs_setup.create_settings_file(filename = 'system_settings.txt')

        # Clear remote machines of existing dir structure
        remote_envs_setup.remove_remote_directories()

        # Build list of directories for remote dir structure
        remote_envs_setup.build_remote_directories(args.path_agent_files, args.path_shared_secrets,
                                            args.path_neuralnet_models, args.path_secret_neuralnets)

        # Send required root dir scripts, agent info files, and any neural net models
        root_dir_scripts = ['install_pylibs.sh', 'system_settings.txt', 'device_settings.py',
                            'agents.py', 'multi_agent_systems.py', 
                            'network_functions.py', 'file_handling.py', 
                            'run_virtual_agent.py', 'hyp_test.py',
                            'key_gen.py']

        remote_envs_setup.send_required_files(root_dir_scripts, 
                                            args.path_agent_files, args.path_shared_secrets,
                                            args.path_neuralnet_models, args.path_secret_neuralnets)

        # Install required python libaries on remote machines
        remote_envs_setup.install_required_pylibs('install_pylibs.sh')

        # Run agents on remote machines with run script
        remote_envs_setup.run_agents_remotely('run_virtual_agent.py')

    else:
        # Create multi-agent system entirely on local machine
        if args.system_type == 'centralized':
            local_system = CentralizedLocalSystem(system)
        else:
            local_system = DecentralizedLocalSystem(system)

        # Have agents in local system perform interaction process
        local_system.local_interaction()

        # Have agents in local system perform authentication
        local_system.authenticate_local_agents()

        print("\nAuthentication results for all agents:")
        for l in range(group_size):
            print("\nagent ", l, ": ", local_system.system.agents_list[l].auth_results)

        # Have agents in local system compute keys
        local_system.setup_session_key()

        print("\nKey agreement results for all agents: ")
        
        if group_size > 2:
            for l in range(group_size):
                print("\nagent ", l, ": ", local_system.system.agents_list[l].group_key)
        
        else:
            cs_gid = local_system.central_server_group_id

            for l in range(group_size):
                if l == cs_gid:
                    print("\nCentral server mutual keys: ")
                    print(local_system.agents_list[cs_gid].mutual_keys_list)
                else:
                    print("\nAgent mutual key:")
                    print(local_system.agents_list[l].mutual_key)


if __name__ == "__main__":
    run()
