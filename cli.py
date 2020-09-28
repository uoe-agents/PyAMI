"""Function to parse CLI."""
import argparse


def arg_parser():
    """CLI for run.py."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--system_type', type=str, default='centralized',
                        help="Specify either centralized/decentrized system.")
    
    parser.add_argument('--run_remotely', type=str, default='yes',
                        help="Specify yes/no to run agents all remotely or locally.")

    parser.add_argument('--group_size', type=int, default=3,
                        help="Number of agents in the system.")

    parser.add_argument('--path_agent_files', type=str, default=None,
                        help="Path to agent json files.")

    parser.add_argument('--path_shared_secrets', type=str, default=None,
                        help="Path to shared secret json files.")

    parser.add_argument('--path_neuralnet_models', type=str, default=None,
                        help="Path to neural net agent models.")

    parser.add_argument('--path_secret_neuralnets', type=str, default=None,
                        help="Path to shared secret neural net models.")

    args = parser.parse_args()
    print("args: ", args)
    
    return args
