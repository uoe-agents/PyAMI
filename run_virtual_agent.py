"""A script which runs an agent on a remote machine."""
from file_handling import collect_json_from_dir, collect_neuralnets_from_dir, collect_neuralnets_from_subdir
from multi_agent_systems import RemoteSystem
import json, statistics
from datetime import datetime

def run_agent():

    # Retrieve system json file 
    with open("system_settings.txt", "r") as f:
        system_settings = json.load(f)

    # Retrieve agent-specific and shared secrets files
    group_size = system_settings['group_size']

    path_agent_files = system_settings['path_agent_files']
    path_shared_secrets = system_settings['path_shared_secrets']
    path_neuralnet_models = system_settings['path_neuralnet_models']
    path_secret_neuralnets = system_settings['path_secret_neuralnets']


    # Collect json objects for agents and for shared secrets
    agents_json = collect_json_from_dir(path_agent_files, num_files=1)
    agent_group_id = agents_json[0]['group_id'] # get id from agent info file

    shared_secrets_json = collect_json_from_dir(path_shared_secrets, num_files=1)
   
    # unwrap this agent's json info/secrets objects from list
    agent_info_json = agents_json[0]
    agent_secrets_json = shared_secrets_json[0]

    # Collect available neural net agent models from nested subdir
    # List will contain -1 if statedict file not listed in directory
    agent_model_subdir = 'agent' + str(agent_group_id) + '_neuralnet_models'
    subdir_path = path_neuralnet_models + '/' + agent_model_subdir
    model_statedicts = collect_neuralnets_from_subdir(subdir_path, group_size)

    # Collect available neural net shared secrets from nested subdirs
    agent_secret_subdir = 'agent'+str(agent_group_id) + '_secret_models'
    subdir_path = path_secret_neuralnets + '/' + agent_secret_subdir
    secret_statedicts = collect_neuralnets_from_subdir(subdir_path, group_size)

    # Print all retrieved objects
    print("\nagent_info_json: ", agent_info_json.values())
    print("\nagent_secrets_json: ", agent_secrets_json.values())
    print("\nmodel_statedicts: ", model_statedicts)
    print("\nsecret_statedicts: ", secret_statedicts)


    # Create remote system using system settings
    remote_system = RemoteSystem(system_settings, 
                                agent_info_json, agent_secrets_json,
                                model_statedicts, secret_statedicts)

    # Set up socket conn between this agent and other remote agents
    remote_system.connect_sockets(agent_info_json, network_protocol='tcp')


    num_interactions = 100
    elapsed_time_list, start_datetime_list, end_datetime_list = [], [], []

    for i in range(num_interactions):
        #print("\ni: ", i)

        # Start datetime
        start_datetime = datetime.now()

        # Perform interaction process with other remote agents
        (_, _, _) = remote_system.remote_interaction()
        
        # Have agent authenticate other remote agents
        remote_system.authenticate_remote_agents(agent_info_json)

        # End datetime
        end_datetime = datetime.now()

        # Record time in lists
        start_datetime_list.append(datetime.strftime(start_datetime,'%Y-%m-%d %H:%M:%S:%f'))
        end_datetime_list.append(datetime.strftime(end_datetime,'%Y-%m-%d %H:%M:%S:%f'))


    #sprint("\nelapsed_time_list: ", elapsed_time_list)
    print("\nstart_datetime_list: ", start_datetime_list)
    print("\nend_datetime_list: ", end_datetime_list)


    #return # REMOVE AFTER TIME TESTING

    # Have agent generate session key(s)
    remote_system.setup_session_key()

if __name__ == "__main__":
    run_agent()