# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import yaml
import os
import datetime
import pkgutil
import argparse
import subprocess


# Required configuration fields.
LOG_DIR = 'log_directory'
MODULE = 'module'
DATA_DIR = 'data_dir'
CONTAINER = 'container'
MACHINES = 'machines'
MODEL_TYPE = 'model_type'

# Model types.
IMAGE_CLASSIFICATION = 'image_classification'
TRANSLATION = 'translation'
SPEECH_TO_TEXT = 'speech_to_text'

# Optional configurations fields.
DISTRIBUTED_BACKEND = 'distributed_backend'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'
LEARNING_RATE_POLICY = 'learning_rate_policy'
WEIGHT_DECAY = 'weight_decay'
EPOCHS = 'epochs'
CONFIG_FILE = 'config_file'
PRINT_FREQUENCY = 'print_frequency'
NO_INPUT_PIPELINING = 'no_input_pipelining'
VERBOSE_FREQUENCY = 'verbose_frequency'
LR_WARMUP = 'lr_warmup'
SYNTHETIC_DATA = 'synthetic_data'
RECOMPUTE = 'recompute'
MACROBATCH = 'macrobatch'


'''
Remaining TODOs:
    1) Support restarting from checkpoints.
    2) Assuming for now runtime is in the same folder as driver.
    3) Support non-NFS checkpoint directories.
'''


def create_output_folder(conf):
    output_folder_path = os.path.join(
        conf[LOG_DIR], datetime.datetime.now().isoformat().split('.')[0])
    sys.stdout.write('Creating output folder: %s\n' % output_folder_path)

    try:
        os.makedirs(output_folder_path)
    except OSError:
        raise OSError

    return output_folder_path


class WorkerInfo(object):
    def __init__(self, ip, gpu_id):
        self.ip = ip
        self.gpu_id = int(gpu_id)

    def __repr__(self):
        return '%s:%d' % (self.ip, self.gpu_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True,
                        help="Path to configuration file")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--launch_single_container', action='store_true',
                        help='launch a single container per machine')
    parser.add_argument('--mount_directories', type=str, nargs='+',
                        help='list of directories to mount')
    parser.add_argument('--quiet', action='store_true',
                        help='quiet execution')
    args = parser.parse_args()

    sys.stdout.write('Using configuration path {0}\n'.format(args.config_file))
    assert os.path.isfile(args.config_file), args.config_file

    with open(args.config_file, 'r') as stream:
        configurations = yaml.load(stream=stream)

    # Check that necessary fields are filled out in the configuration file.
    assert LOG_DIR in configurations
    assert MODULE in configurations
    assert DATA_DIR in configurations
    assert CONTAINER in configurations
    assert MACHINES in configurations
    assert MODEL_TYPE in configurations
    assert DISTRIBUTED_BACKEND in configurations

    # TODO: make python path a configuration the user must specify.
    if configurations[MODEL_TYPE] == TRANSLATION:
        print("WARNING: Make sure the container used has python "
              "translation/setup.py run!")
        main_with_runtime_folder = 'translation'
        disable_gpu_gpu_communication = False
        python_path = 'python'
    elif configurations[MODEL_TYPE] == IMAGE_CLASSIFICATION:
        main_with_runtime_folder = 'image_classification'
        disable_gpu_gpu_communication = False
        python_path = 'python'
    elif configurations[MODEL_TYPE] == SPEECH_TO_TEXT:
        main_with_runtime_folder = 'speech_to_text'
        disable_gpu_gpu_communication = True
        python_path = 'python'
    else:
        raise NotImplementedError

    # Check that specified files & directories exist.
    assert os.path.isdir(configurations[LOG_DIR]), \
        configurations[LOG_DIR]
    assert os.path.isdir(configurations[DATA_DIR]), \
        configurations[DATA_DIR]
    if CONFIG_FILE in configurations:
        config_file_path = os.path.join(main_with_runtime_folder,
                                        configurations[CONFIG_FILE])
        assert os.path.isfile(config_file_path), config_file_path

    # Get the modules directory. Note that the driver program should be
    # run using Python3.
    module_path = '%s.%s' % (main_with_runtime_folder,
                             configurations[MODULE])
    if not args.quiet:
        module_directory = pkgutil.get_loader(module_path).path
        if not os.path.isdir(module_directory):
            module_directory = os.path.dirname(module_directory)
        assert os.path.isdir(module_directory), module_directory

    # Check that machine list is non-empty.
    assert isinstance(configurations[MACHINES], (list,))
    assert len(configurations[MACHINES]) > 0

    # Parse IP and GPU_ID information.
    workers = []
    nodes_to_workers_mapping = {}
    for machine in configurations[MACHINES]:
        machine_info = machine.split(":")
        assert len(machine_info) == 2, machine
        workers.append(WorkerInfo(ip=machine_info[0],
                                  gpu_id=machine_info[1]))
        if machine_info[0] not in nodes_to_workers_mapping:
            nodes_to_workers_mapping[machine_info[0]] = []
        nodes_to_workers_mapping[machine_info[0]].append(workers[-1])
    assert len(workers) == len(configurations[MACHINES])

    # Create output directory.
    output_dir = create_output_folder(conf=configurations)

    # Copy configuration file to output folder.
    copy_command = 'cp %s %s' % (args.config_file, output_dir)
    if not args.quiet:
        subprocess.check_output(copy_command, shell=True)

    # Copy module (this copies configuration file over as well).
    if not args.quiet:
        copy_command = 'cp -r %s %s' % (module_directory, output_dir)
        subprocess.check_output(copy_command, shell=True)

    # Save git hash and diff.
    git_log_file_path = os.path.join(output_dir, 'git_info.log')
    git_info_cmd = 'git rev-parse HEAD > %s' % git_log_file_path
    if not args.quiet:
        subprocess.check_output(git_info_cmd, shell=True)

    git_info_cmd = 'git diff >> %s' % git_log_file_path
    if not args.quiet:
        subprocess.check_output(git_info_cmd, shell=True)

    # Path to record command history.
    command_history_file_path = os.path.join(output_dir,
                                             'command_history.log')
    command_history_file = open(command_history_file_path, "w")

    # Create machine file with all IP addresses.
    machine_file_path = os.path.join(output_dir, 'machinefile')
    machine_file_fh = open(machine_file_path, "w")
    machine_to_workers_map = {}
    for worker in workers:
        if worker.ip in machine_to_workers_map:
            machine_to_workers_map[worker.ip] += 1
        else:
            machine_to_workers_map[worker.ip] = 1
            machine_file_fh.write(worker.ip + '\n')
    machine_file_fh.close()

    main_runtime_filename = 'main_with_runtime.py'

    runtime_cmd_preamble_list = []
    if args.launch_single_container and CONFIG_FILE in configurations:
        runtime_cmd_preamble_list.append(
            ('cp %(working_dir)s/launch.py '
             '%(working_dir)s/%(main_with_runtime_folder)s; ') % {
                "working_dir": os.getcwd(),
                "main_with_runtime_folder": main_with_runtime_folder,
        })
    runtime_cmd_preamble_list.append('cd %(working_dir)s/%(main_with_runtime_folder)s; '
                                     '%(python_path)s' % {
                                         "working_dir": os.getcwd(),
                                         "main_with_runtime_folder":
                                            main_with_runtime_folder,
                                         "python_path": python_path,
                                     })
    runtime_cmd_list = ['%(main_runtime_filename)s '
                        '--data_dir %(data_dir)s '
                        '--master_addr %(master_addr)s --module %(module)s '
                        '--checkpoint_dir %(checkpoint_dir)s '
                        '--distributed_backend %(distributed_backend)s ' % {
                            "main_runtime_filename": main_runtime_filename,
                            "data_dir": configurations[DATA_DIR],
                            "master_addr": workers[0].ip,
                            "module": configurations[MODULE],
                            "checkpoint_dir": output_dir,
                            "distributed_backend": configurations[DISTRIBUTED_BACKEND],
                        }]

    # Add additional arguments.
    if BATCH_SIZE in configurations:
        runtime_cmd_list.append('-b %d' % configurations[BATCH_SIZE])

    if LEARNING_RATE in configurations:
        runtime_cmd_list.append('--lr %f' % configurations[LEARNING_RATE])

    if LEARNING_RATE_POLICY in configurations:
        runtime_cmd_list.append('--lr_policy %s' %
                                configurations[LEARNING_RATE_POLICY])

    if WEIGHT_DECAY in configurations:
        runtime_cmd_list.append('--weight-decay %f' %
                                configurations[WEIGHT_DECAY])

    if EPOCHS in configurations:
        runtime_cmd_list.append('--epochs %d' % configurations[EPOCHS])

    if PRINT_FREQUENCY in configurations:
        runtime_cmd_list.append('--print-freq %d' %
                                configurations[PRINT_FREQUENCY])

    if NO_INPUT_PIPELINING in configurations and \
        configurations[NO_INPUT_PIPELINING]:
        runtime_cmd_list.append('--no_input_pipelining')

    if VERBOSE_FREQUENCY in configurations:
        runtime_cmd_list.append('--verbose %d' % \
            configurations[VERBOSE_FREQUENCY])

    if args.resume:
        runtime_cmd_list.append('--resume %s' % args.resume)

    if LR_WARMUP in configurations and configurations[LR_WARMUP]:
        runtime_cmd_list.append('--lr_warmup')

    if SYNTHETIC_DATA in configurations and configurations[SYNTHETIC_DATA]:
        runtime_cmd_list.append('--synthetic_data')

    if RECOMPUTE in configurations and configurations[RECOMPUTE]:
        runtime_cmd_list.append('--recompute')

    if MACROBATCH in configurations and configurations[MACROBATCH]:
        runtime_cmd_list.append('--macrobatch')

    common_runtime_cmd = " ".join(runtime_cmd_list)

    # If launching in a single container per node, use launch utility to spawn
    # required number of processes in the same container.
    if args.launch_single_container:
        all_runtime_cmds = []
        for node_rank, (node_ip, workers) in \
            enumerate(nodes_to_workers_mapping.items()):
            docker_cmd = 'nvidia-docker run -d %(mount_directories)s ' \
                         '--net=host ' \
                         '--shm-size 16g %(container)s /bin/bash -c' % {
                "container": configurations[CONTAINER],
                "mount_directories":
                    " ".join(["-v %s:%s" % (x, x)
                              for x in args.mount_directories])
            }

            log_file_path = '%s/output.log.%d' % (output_dir, node_rank)

            num_ranks_in_server = \
                machine_to_workers_map[worker.ip] \
                if not disable_gpu_gpu_communication else 1

            runtime_cmd_list = [common_runtime_cmd,
                                '--num_ranks_in_server %d' % num_ranks_in_server]

            launch_module = ''
            if CONFIG_FILE in configurations:
                runtime_cmd_list.append('--config_path %s' % (
                    configurations[CONFIG_FILE]))
                launch_module = '-m launch --nnodes %(nnodes)d --node_rank %(node_rank)d ' \
                                '--nproc_per_node %(nproc_per_node)d' % {
                    "nnodes": len(nodes_to_workers_mapping),
                    "node_rank": node_rank,
                    "nproc_per_node": num_ranks_in_server
                }

            runtime_cmd_list = runtime_cmd_preamble_list + [launch_module] + runtime_cmd_list
            runtime_cmd_list.append('2>&1 | tee %s' % log_file_path)
            runtime_cmd = " ".join(runtime_cmd_list) + "; rm launch.py"

            launch_cmd = '%s \'%s\'' % (docker_cmd, runtime_cmd)
            if node_ip != 'localhost' and node_ip != '127.0.0.1':
                launch_cmd = 'ssh -n %s -o StrictHostKeyChecking=no \"%s\"' % (node_ip,
                                                                               launch_cmd)
            print(launch_cmd)
            command_history_file.write(launch_cmd + "\n")

            if not args.quiet:
                subprocess.check_output(launch_cmd, shell=True)
    else:
        for rank, worker in enumerate(workers):
            docker_cmd = 'nvidia-docker run -d %(mount_directories)s ' \
                         '--net=host ' \
                         '--ipc=host %(container)s /bin/bash -c' % {
                "container": configurations[CONTAINER],
                "mount_directories":
                    " ".join(["-v %s:%s" % (x, x)
                              for x in args.mount_directories])
            }

            log_file_path = '%s/output.log.%d' % (output_dir, rank)

            num_ranks_in_server = \
                machine_to_workers_map[worker.ip] if not disable_gpu_gpu_communication else 1

            runtime_cmd_list = [" ".join(runtime_cmd_preamble_list),
                                common_runtime_cmd,
                                '--num_ranks_in_server %d' % num_ranks_in_server]

            if CONFIG_FILE in configurations:
                runtime_cmd_list.append('--config_path %s --rank %d --local_rank %d' % (
                    configurations[CONFIG_FILE], rank, rank % num_ranks_in_server))
            runtime_cmd_list.append('2>&1 | tee %s' % log_file_path)
            runtime_cmd = " ".join(runtime_cmd_list)

            launch_cmd = '%s \'%s\'' % (docker_cmd, runtime_cmd)
            if worker.ip != 'localhost':
                launch_cmd = 'ssh -n %s -o StrictHostKeyChecking=no \"%s\"' % (worker.ip,
                                                                               launch_cmd)
            print (launch_cmd)
            command_history_file.write(launch_cmd + "\n")

            if not args.quiet:
                subprocess.check_output(launch_cmd, shell=True)

    command_history_file.close()
