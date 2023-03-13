import os
import socket

# Setup information.
workspace_name = 'workspace'

# Data location mapping.
datastore = {
        'default':{
            'datasets':{
                },
            }
        }


def setup_workspace():
    """Sets up the workspace for the project by creating a virtual environment,
    installing the required packages and creating symlinks for the dataset.
    """

    hostname = socket.gethostname()

    # Create the workspace folder if it does not exist.
    workspace_path = os.path.abspath(workspace_name)
    if not os.path.isdir(workspace_path):
        print('Creating workspace: {}'.format(workspace_path))
        os.makedirs(workspace_path)


    # Set up symlinks and project folders.
    from termcolor import colored
    print('Setting up for {}'.format(colored(hostname, 'red')))

    paths = None
    if hostname not in datastore:
        print("""Hostname {} is not setup. Using default Cluster""".format(colored(hostname, 'red')))
        paths = datastore['default']
    else:
        paths = datastore[hostname]

    for parent, files in paths.items():
        print('Preparing {}'.format(colored(parent)))
        parent_path = os.path.join(workspace_path, parent)
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)

        for key, value in files.items():
            target_path = os.path.join(parent_path, key)
            print('\t{} -> {}'.format(colored(target_path, 'green'),
                colored(value, 'blue')))
            if not os.path.islink(target_path):
                os.system('ln -s "{}" "{}"'.format(value, target_path))
    print('Setting up symlinks finished.')


if __name__ == '__main__':
    setup_workspace()