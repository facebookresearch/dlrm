# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class RuntimeStats:
    def __init__(self, forward):
        self.stats = {
            'compute_time': 0.0,
            'send_tensors': 0.0,
            'send_tensors_size': 0,
            'receive_tensors': 0.0,
            'receive_tensors_size': 0,
        }
        self.forward = forward

    def print_stats(self):
        if self.forward:
            print("Forward Stats:")
        else:
            print("Backward Stats:")
        for i in sorted(self.stats):
            units = 'seconds'
            if i == 'receive_tensors_size' or i == 'send_tensors_size':
                units = 'bytes'
            print("\t %s %.3f %s" % (i, self.stats[i], units))

    def reset_stats(self):
        for i in self.stats.keys():
            self.stats[i] = 0.0