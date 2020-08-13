# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os


def print_usage():
    print ("python process_output.py <output.log>")


def resumed_from_checkpoint(first_line, second_line):
    if len(first_line.split("=> loading checkpoint")) > 1:
        starting_epoch = int(second_line.split("epoch ")[1].split(")")[0])
        return True, starting_epoch
    return False, 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()
        sys.exit()

    log_file_path = sys.argv[1]
    assert os.path.isfile(log_file_path), log_file_path

    # Read log output file
    log_output = [line.rstrip('\n') for line in open(log_file_path)]

    epoch_time = []
    top_1_accuracies = []
    top_5_accuracies = []
    end_of_epoch = False

    # Check if loading checkpoint (runs a validation at start)
    resume, epoch_idx = resumed_from_checkpoint(
        first_line=log_output[0], second_line=log_output[1])

    if resume:
        end_of_epoch = True
        epoch_time.append(0.0)

    for line in log_output:
        if not end_of_epoch:
            key = "Epoch %d: " % epoch_idx
            line = line.split(key)
            if len(line) > 1:
                line = line[1].split(" ")
                epoch_time.append(float(line[0]) / 60.0 / 60.0)
                end_of_epoch = True
                epoch_idx += 1
        else:
            key = " * "
            line = line.split(key)
            if len(line) > 1:
                line = line[1].split(" ")
                top_1_accuracies.append(float(line[1]))
                top_5_accuracies.append(float(line[3]))
                end_of_epoch = False

    run_time = 0
    print ("Epoch#\tRuntime\tTop-1\tTop-5\tEpoch")
    for i in range(len(epoch_time)):
        run_time += epoch_time[i]
        print("%d\t%.3f\t%.3f\t%.3f\t%.3f" % (
            i+1, run_time, top_1_accuracies[i], top_5_accuracies[i], epoch_time[i]))