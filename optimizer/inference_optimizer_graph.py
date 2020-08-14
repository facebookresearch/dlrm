# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import csv
import math
import os

import sys
sys.path.append("..")
import graph
import utils

def main(num_machines, profile_filename, time_between_inputs, network_bandwidth, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression, output_directory, num_machines_in_first_level=None,
         print_configuration=True, verbose=False):
    if (num_machines_in_first_level is not None and
        num_machines_in_first_level > num_machines):
        raise Exception("num_machines_in_first_level has to less than num_machines!")

    gr = graph.Graph.from_str(open(profile_filename, 'r').read())
    antichain_gr = gr.antichain_dag()
    states = antichain_gr.topological_sort()
    if verbose:
        print("Total number of states: %d" % len(states))
    states_indices = {}
    for i in range(len(states)):
        states_indices[states[i]] = i
    for i in range(len(states)):
        for antichain_node in states[i].antichain:
            states[i].output_activation_size += gr.nodes[antichain_node].activation_size

    A = []
    for i in range(len(states)):
        row_A = []
        for j in range(num_machines):
            row_A.append((None, None, None, None))
        A.append(row_A)

    for i in range(len(states)):
        antichain = states[i].antichain
        all_predecessors = gr.all_predecessors(antichain)
        states[i].compute_time = 0.0
        states[i].activation_size = 0.0
        states[i].parameter_size = 0.0
        for predecessor in all_predecessors:
            states[i].compute_time += (predecessor.forward_compute_time  / 1000.0)
            states[i].activation_size += predecessor.activation_size
            states[i].parameter_size += predecessor.parameter_size
    gr.reset()

    for i in range(len(states)):
        cum_compute_time = states[i].compute_time
        cum_activation_size = states[i].activation_size
        cum_parameter_size = states[i].parameter_size
        max_j = 1 if straight_pipeline else num_machines
        for j in range(max_j):
            stashed_data_size = cum_activation_size + cum_parameter_size
            if use_memory_constraint and stashed_data_size > memory_size:
                A[i][j] = (None, None, None, None)
                continue
            if num_machines_in_first_level is not None and j != (num_machines_in_first_level - 1):
                A[i][j] = (None, None, None, None)
            else:
                if (cum_compute_time / (j+1)) < time_between_inputs:
                    A[i][j] = (cum_compute_time / (j+1), cum_compute_time, None, (j+1))

    min_machines = 1 if num_machines_in_first_level is None else num_machines_in_first_level
    for m in range(min_machines, num_machines):
        for i in range(1, len(states)):
            (min_pipeline_time, min_pipeline_latency, optimal_split, optimal_num_machines) = A[i][m]
            if use_fewer_machines and m > 0 and (min_pipeline_time is None or A[i][m-1][0] < min_pipeline_time):
                (min_pipeline_time, min_pipeline_latency, optimal_split, optimal_num_machines) = A[i][m-1]
            predecessors = antichain_gr.predecessors(states[i].node_id)
            predecessor_ids = [states_indices[predecessor] for predecessor in predecessors]
            for j in predecessor_ids:
                max_m_prime = 2 if straight_pipeline else (m+1)
                for m_prime in range(1, max_m_prime):
                    input_transfer_time = states[j].output_activation_size / (network_bandwidth * m_prime)
                    output_transfer_time = None
                    if i < len(states) -1:
                        output_transfer_time = states[i].output_activation_size / (network_bandwidth * m_prime)

                    last_stage_time = states[i].compute_time - states[j].compute_time
                    last_stage_parameter_size = states[i].parameter_size - states[j].parameter_size
                    stashed_data_size = (states[i].activation_size - states[j].activation_size) + last_stage_parameter_size
                    if use_memory_constraint and stashed_data_size > memory_size:
                        continue
                    last_stage_time /= m_prime

                    if A[j][m-m_prime][0] is None:
                        continue

                    pipeline_latency = sum([A[j][m-m_prime][1], last_stage_time * m_prime])
                    pipeline_time = max(A[j][m-m_prime][0], last_stage_time)
                    if not activation_compression:
                        pipeline_time = max(pipeline_time, input_transfer_time)
                        pipeline_latency = sum([pipeline_latency, input_transfer_time * m_prime])
                        if output_transfer_time is not None:
                            pipeline_time = max(pipeline_time, output_transfer_time)
                            pipeline_latency = sum([pipeline_latency, output_transfer_time * m_prime])

                    if pipeline_time > time_between_inputs:
                        continue
                    if min_pipeline_latency is None or min_pipeline_latency > pipeline_latency:
                        optimal_split = (j, m-m_prime)
                        optimal_num_machines = m_prime
                        min_pipeline_time = pipeline_time
                        min_pipeline_latency = pipeline_latency
            A[i][m] = (min_pipeline_time, min_pipeline_latency, optimal_split, optimal_num_machines)

    metadata = A[len(states)-1][num_machines-1]
    next_split = metadata[2]
    remaining_machines_left = num_machines
    splits = []
    replication_factors = []
    prev_split = len(states)
    while next_split is not None:
        num_machines_used = metadata[3]
        if verbose:
            print("-------------------------------------")
            print("Number of machines used: %d..." % num_machines_used)
            print("Split between layers %d and %d..." % (next_split[0], next_split[0] + 1))
            print("Split before antichain %s..." % (states[next_split[0]+1].antichain))
        splits.append(next_split[0]+1)
        compute_time = states[prev_split-1].compute_time - states[next_split[0]].compute_time
        parameter_size = states[prev_split-1].parameter_size - states[next_split[0]].parameter_size

        pp_communication_time_input = states[next_split[0]].output_activation_size / network_bandwidth
        pp_communication_time_output = states[prev_split-1].output_activation_size / network_bandwidth
        if activation_compression:
            pp_communication_time_input = 0.0
            pp_communication_time_output = 0.0

        compute_time /= num_machines_used

        if verbose:
            print("Compute time = %f, Pipeline-parallel communication time = %f..." % (
                compute_time, max(pp_communication_time_input, pp_communication_time_output)))
        prev_split = splits[-1]
        metadata = A[next_split[0]][next_split[1]]
        next_split = metadata[2]
        replication_factors.append(num_machines_used)
        remaining_machines_left -= num_machines_used
    if verbose:
        print("-------------------------------------")
        print("Number of machines used: %d..." % metadata[3])

    num_machines_used = metadata[3]
    remaining_machines_left -= num_machines_used
    compute_time = states[prev_split-1].compute_time
    parameter_size = states[prev_split-1].parameter_size
    compute_time /= num_machines_used

    if verbose:
        print("Compute time = %f..." % compute_time)
        print("-------------------------------------")
    if print_configuration:
        print("Number of machines in budget not used: %d..." % remaining_machines_left)

    if print_configuration:
        print("(Split start, split end) / compute time taken per stage / replication factor per stage:")
    prev_split = 0
    splits.reverse()
    splits.append(len(states))
    replication_factors.append(num_machines_used)
    replication_factors.reverse()
    for i in range(len(splits)):
        time = 0.0
        if prev_split > 0:
            time = states[splits[i]-1].compute_time - states[prev_split-1].compute_time
        else:
            time = states[splits[i]-1].compute_time
        if print_configuration:
            print(prev_split, splits[i], time, replication_factors[i])
        if splits[i] < len(states):
            predecessors = gr.all_predecessors(states[splits[i]-1].antichain)
            for predecessor in predecessors:
                if predecessor.stage_id is None:
                    predecessor.set_stage_id(i)
        prev_split = splits[i]
    for node in gr.nodes.values():
        if node.stage_id is None:
            node.set_stage_id(len(splits)-1)
    if output_directory is not None:
        gr.to_dot(os.path.join(output_directory, "gpus=%d" % num_machines))
        gr_str = str(gr)
        with open(os.path.join(output_directory, "gpus=%d.txt" % num_machines), 'w') as f:
            f.write(gr_str)

    total_time = states[-1].compute_time
    total_parameter_size = states[-1].parameter_size
    pipeline_parallel_total_time = A[len(states)-1][num_machines-1][0]
    pipeline_parallel_latency = A[len(states)-1][num_machines-1][1]

    if verbose:
        print()
        print("Time taken by single-stage pipeline:", total_time)
        print("Total latency:", pipeline_parallel_latency)
        print("Time per stage in pipeline:", pipeline_parallel_total_time)
        print("Throughput increase (compared to single machine):",
            total_time / pipeline_parallel_total_time)
        print("[Note that single-machine and %d-machine DP might not fit given memory constraints]")
    return pipeline_parallel_total_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run PipeDream's optimizer for replicated settings")
    )
    parser.add_argument('-n', "--num_machines", required=True, type=int,
                        help="Number of machines available")
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Profile filename")
    parser.add_argument('-b', "--network_bandwidth", type=float, default=1000000000,
                        help="Available network bandwidth in bytes/sec")
    parser.add_argument('-m', "--num_machines_in_first_level", type=int, default=None,
                        help="Number of machines in first level")
    parser.add_argument('-s', "--memory_size", type=float, default=16000000000,
                        help="Amount of memory available on each machine")
    parser.add_argument("--straight_pipeline", action='store_true',
                        help="No replication across stages")
    parser.add_argument('-o', "--output_directory", default=None, type=str,
                        help="Output directory to dump processed graph")
    parser.add_argument("--use_memory_constraint", action='store_true',
                        help="Enforce memory constraint per machine")
    parser.add_argument("--use_fewer_machines", action='store_true',
                        help="Use fewer machines, if possible")
    parser.add_argument("--activation_compression", action='store_true',
                        help="Compress activations")
    parser.add_argument('-t', "--time_between_inputs", required=True, type=float,
                        help="Time between inputs")

    args = parser.parse_args()
    args = vars(args)

    num_machines = args["num_machines"]
    profile_filename = args["profile_filename"]
    network_bandwidth = args["network_bandwidth"]
    memory_size = args["memory_size"]
    num_machines_in_first_level = args["num_machines_in_first_level"]
    straight_pipeline = args["straight_pipeline"]
    output_directory = args["output_directory"]
    use_memory_constraint = args["use_memory_constraint"]
    use_fewer_machines = args["use_fewer_machines"]
    activation_compression = args["activation_compression"]
    time_between_inputs = args["time_between_inputs"]

    main(num_machines, profile_filename, time_between_inputs, network_bandwidth, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression, output_directory, num_machines_in_first_level=num_machines_in_first_level,
         verbose=True)
