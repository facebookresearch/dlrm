# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import csv
import math

import utils
try:
    # Python 2
    xrange
except NameError:
    # Python 3
    xrange = range

def main(num_machines, profile_filename, network_bandwidth, memory_size,
         num_machines_in_first_level=None, verbose=False):
    if (num_machines_in_first_level is not None and
        num_machines_in_first_level > num_machines):
        raise Exception("num_machines_in_first_level has to less than num_machines!")

    profile_data = utils.parse_profile_file(profile_filename)

    A = []
    for i in xrange(len(profile_data)):
        row_A = []
        for j in xrange(num_machines):
            row_A.append(None)
        A.append(row_A)

    cum_sum = 0.0
    cum_activation_size = 0.0
    cum_parameter_size = 0.0
    for i in xrange(len(profile_data)):
        cum_sum += profile_data[i][0]
        cum_activation_size += profile_data[i][1]
        cum_parameter_size += profile_data[i][2]
        for j in xrange(num_machines):
            stashed_data_size = math.ceil((num_machines - (j+1)) / (j+1)) * cum_activation_size
            stashed_data_size += cum_parameter_size
            if stashed_data_size > memory_size:
                A[i][j] = (None, None)
                continue
            data_parallel_communication_time = (4 * j * cum_parameter_size) / (network_bandwidth * (j+1))
            if num_machines_in_first_level is not None and j != (num_machines_in_first_level - 1):
                A[i][j] = (None, None)
            else:
                A[i][j] = (max(cum_sum, data_parallel_communication_time) / (j+1), None)

    min_machines = 1 if num_machines_in_first_level is None else num_machines_in_first_level
    cum_times = []
    cum_activation_sizes = []
    cum_parameter_sizes = []
    for i in xrange(len(profile_data)):
        if i == 0:
            cum_times.append(profile_data[i][0])
            cum_activation_sizes.append(profile_data[i][1])
            cum_parameter_sizes.append(profile_data[i][2])
        else:
            cum_times.append(cum_times[-1] + profile_data[i][0])
            cum_activation_sizes.append(cum_activation_sizes[-1] + profile_data[i][1])
            cum_parameter_sizes.append(cum_parameter_sizes[-1] + profile_data[i][2])

    for m in xrange(min_machines, num_machines):
        for i in xrange(1, len(profile_data)):
            (min_pipeline_time, optimal_split) = A[i][m]
            for j in xrange(i):
                for m_prime in xrange(1, m+1):
                    input_transfer_time = (2.0 * profile_data[j][1]) / (network_bandwidth * m_prime)
                    output_transfer_time = None
                    if i < len(profile_data) -1:
                        output_transfer_time = (2.0 * profile_data[i-1][1]) / (network_bandwidth * m_prime)

                    last_stage_time = cum_times[i] - cum_times[j]
                    last_stage_parameter_size = cum_parameter_sizes[i] - cum_parameter_sizes[j]
                    stashed_data_size = (cum_activation_sizes[i] - cum_activation_sizes[j])
                    stashed_data_size *= math.ceil((num_machines - (m+1)) / m_prime)
                    stashed_data_size += last_stage_parameter_size
                    if stashed_data_size > memory_size:
                        continue
                    last_stage_time = max(last_stage_time,
                                          ((4 * (m_prime - 1) * last_stage_parameter_size) / (network_bandwidth * m_prime)))
                    last_stage_time /= m_prime

                    if A[j][m-m_prime][0] is None:
                        continue
                    pipeline_time = max(A[j][m-m_prime][0], last_stage_time, input_transfer_time)
                    if output_transfer_time is not None:
                        pipeline_time = max(pipeline_time, output_transfer_time)
                    if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                        optimal_split = (j, m-m_prime)
                        min_pipeline_time = pipeline_time
            A[i][m] = (min_pipeline_time, optimal_split)

    metadata = A[len(profile_data)-1][num_machines-1]
    next_split = metadata[1]
    remaining_machines_left = num_machines
    splits = []
    replication_factors = []
    prev_split = len(profile_data)
    while next_split is not None:
        num_machines_used = (remaining_machines_left - next_split[1] - 1)
        if verbose:
            print("Number of machines used: %d..." % num_machines_used)
            print("Split between layers %d and %d..." % (next_split[0], next_split[0] + 1))
        splits.append(next_split[0]+1)
        compute_time = 0.0
        parameter_size = 0.0
        for i in range(next_split[0]+1, prev_split):
            compute_time += profile_data[i][0]
            parameter_size += profile_data[i][2]
        dp_communication_time = (4 * (num_machines_used - 1) * parameter_size) / (network_bandwidth * num_machines_used)
        pp_communication_time_input = (profile_data[next_split[0]][1] * (1.0 / float(num_machines_used))) / network_bandwidth
        pp_communication_time_output = (profile_data[prev_split-1][1] * (1.0 / float(num_machines_used))) / network_bandwidth
        compute_time /= num_machines_used
        dp_communication_time /= num_machines_used
        if verbose:
            print("Compute time = %f, Data-parallel communication time = %f, Pipeline-parallel communication time = %f..." % (
                compute_time, dp_communication_time, max(pp_communication_time_input, pp_communication_time_output)))
        prev_split = splits[-1]
        metadata = A[next_split[0]][next_split[1]]
        next_split = metadata[1]
        replication_factors.append(num_machines_used)
        remaining_machines_left -= num_machines_used
    if verbose:
        print("Number of machines used: %d..." % remaining_machines_left)
    num_machines_used = remaining_machines_left
    compute_time = 0.0
    parameter_size = 0.0
    for i in range(prev_split):
        compute_time += profile_data[i][0]
        parameter_size += profile_data[i][2]
    dp_communication_time = (4 * (num_machines_used - 1) * parameter_size) / (network_bandwidth * num_machines_used)
    compute_time /= num_machines_used
    dp_communication_time /= num_machines_used

    if verbose:
        print("Compute time = %f, Data-parallel communication time = %f..." % (compute_time, dp_communication_time))
        print()
        print("(Split start, split end) / time taken per stage / replication factor per stage:")

    prev_split = 0
    splits.reverse()
    splits.append(len(profile_data))
    replication_factors.append(remaining_machines_left)
    replication_factors.reverse()

    for i in xrange(len(splits)):
        time = 0.0
        if verbose:
            print((prev_split, splits[i]),)
        for j in xrange(prev_split, splits[i]):
            time += profile_data[j][0]
        if verbose:
            print(time, replication_factors[i])
        prev_split = splits[i]

    total_time = 0.0
    total_parameter_size = 0.0
    for i in xrange(len(profile_data)):
        total_time += profile_data[i][0]
        total_parameter_size += profile_data[i][2]
    data_parallel_communication_time = ((4 * (num_machines - 1) * total_parameter_size) /
                                        (network_bandwidth * num_machines))
    data_parallel_total_time = max(total_time, data_parallel_communication_time) / num_machines
    pipeline_parallel_total_time = A[len(profile_data)-1][num_machines-1][0]

    if verbose:
        print()
        print("Time taken by single-stage pipeline:", total_time)
        print("Time per stage in pipeline:", pipeline_parallel_total_time)
        print("Throughput increase (compared to single machine):", \
            total_time / pipeline_parallel_total_time)
        print("[Note that single-machine and %d-machine DP might not fit given memory constraints]")
        print("Throughput increase of %d-machine DP compared to single machine:" % num_machines, \
            total_time / data_parallel_total_time)
        print("Throughput increase (compared to %d-machine DP):" % num_machines, \
            data_parallel_total_time / pipeline_parallel_total_time)
        print("Number of images that need to be admitted:", int(math.ceil(
            float(num_machines) / replication_factors[0]) * replication_factors[0]))
    return pipeline_parallel_total_time, data_parallel_total_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run PipeDream's optimizer for replicated settings")
    )
    parser.add_argument('-n', "--num_machines", required=True, type=int,
                        help="Number of machines available")
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Filename of CSV with profile")
    parser.add_argument('-b', "--network_bandwidth", type=float, default=1000000000,
                        help="Available network bandwidth in bytes/sec")
    parser.add_argument('-m', "--num_machines_in_first_level", type=int, default=None,
                        help="Number of machines in first level")
    parser.add_argument('-s', "--memory_size", type=float, default=16000000000,
                        help="Amount of memory available on each machine")

    args = parser.parse_args()
    args = vars(args)

    num_machines = args["num_machines"]
    profile_filename = args["profile_filename"]
    network_bandwidth = args["network_bandwidth"]
    memory_size = args["memory_size"]
    num_machines_in_first_level = args["num_machines_in_first_level"]

    print("=========================")
    print("Without layer splitting")
    print("=========================")
    main(num_machines, profile_filename, network_bandwidth, memory_size,
         num_machines_in_first_level=num_machines_in_first_level,
         verbose=True)
