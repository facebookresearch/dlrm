# from PipeDream with some modifications.

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import graphviz
import os


class Graph(object):
    def __init__(self, node=None):
        self.nodes = {}
        if node is not None:
            self.nodes[node.node_id] = node
        self.edges = {}
        self.in_edges = {}

        self._predecessors = {}
        self._successors = {}
        self._augmented_antichains = {}
        self._deaugmented_augmented_antichains = {}
        self._next_antichains = {}
        self._antichain_dag = None

        self._colors = ['lightblue', 'green', 'grey', 'firebrick1',
                        'gold', 'chocolate1', 'beige']

        if node is not None:
            self.in_edges[node.node_id] = list()

    def copy(self):
        gr = Graph()
        for node_id in self.in_edges:
            for node2 in self.in_edges[node_id]:
                gr.add_edge(node2, self.nodes[node_id])
        return gr

    def sources(self):
        sources = []
        for node_id in self.nodes:
            if node_id not in self.in_edges or len(self.in_edges[node_id]) == 0:
                sources.append(self.nodes[node_id])
        return sources

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def remove_node(self, node):
        del self.nodes[node.node_id]
        if node.node_id in self.edges:
            out_nodes = self.edges[node.node_id]
            del self.edges[node.node_id]
            for out_node in out_nodes:
                self.in_edges[out_node.node_id].remove(node)
        if node.node_id in self.in_edges:
            in_nodes = self.in_edges[node.node_id]
            del self.in_edges[node.node_id]
            for in_node in in_nodes:
                self.edges[in_node.node_id].remove(node)

    def sinks(self):
        sinks = []
        for node_id in self.nodes:
            if node_id not in self.edges or len(self.edges[node_id]) == 0:
                sinks.append(self.nodes[node_id])
        return sinks

    def reset(self):
        self._predecessors = {}
        self._successors = {}

    def add_edge(self, node1, node2):
        if node1.node_id == node2.node_id:
            raise Exception("self circle on %s" % node1.node_id)
        if node1.node_id not in self.nodes:
            self.nodes[node1.node_id] = node1
        if node2.node_id not in self.nodes:
            self.nodes[node2.node_id] = node2

        if node2.node_id not in self.in_edges:
            self.in_edges[node2.node_id] = list()
        self.in_edges[node2.node_id].append(node1)
        if node1.node_id not in self.edges:
            self.edges[node1.node_id] = list()
        self.edges[node1.node_id].append(node2)

    def remove_edge(self, node1, node2):
        self.edges[node1.node_id].remove(node2)
        self.in_edges[node2.node_id].remove(node1)

    def populate_depths(self):
        # Helper method that annotates each node in the graph with its depth from the sink.
        sources = self.sources()
        sources[0].depth = 1
        queue = [sources[0]]
        while len(queue) > 0:
            node = queue.pop(-1)
            if node.node_id not in self.edges: continue
            for out_node in self.edges[node.node_id]:
                if out_node.depth is None or out_node.depth < (node.depth + 1):
                    out_node.depth = node.depth + 1
                queue.append(out_node)

    def populate_heights(self):
        # Helper method that annotates each node in the graph with its height from the further
        # away sink.
        sinks = self.sinks()
        for sink in sinks: sink.height = 1
        queue = sinks
        visited = set()
        while len(queue) > 0:
            node = queue.pop(-1)
            visited.add(node.node_id)
            if node.node_id not in self.in_edges: continue
            for in_node in self.in_edges[node.node_id]:
                if in_node.height is None or in_node.height < (node.height + 1):
                    in_node.height = node.height + 1
                if in_node.node_id not in visited:
                    queue.append(in_node)

    def partition_graph(self):
        stage_ids = set()
        for node_id in self.nodes:
            stage_ids.add(self.nodes[node_id].stage_id)
        if len(stage_ids) == 1:
            return [self.copy()]
        subgraphs = []
        for stage_id in stage_ids:
            subgraphs.append(self.partition_graph_helper(stage_id))
        return subgraphs

    def partition_graph_helper(self, stage_id):
        subgraph = Graph()
        for node1_id in self.nodes:
            if self.nodes[node1_id].stage_id == stage_id:
                subgraph.add_node(self.nodes[node1_id])
                if node1_id not in self.edges: continue
                for node2 in self.edges[node1_id]:
                    if node2.stage_id == stage_id:
                        subgraph.add_edge(self.nodes[node1_id], node2)
        return subgraph

    def compress_branch_helper(self, node, new_node_id):
        if len(self.in_edges[node.node_id]) > 1:
            return None, node
        new_node = Node("compressed_node%d" % new_node_id,
                        node_desc=("Branch %d" % new_node_id))
        chain_length = 0
        # Assumption here is that node has edges coming into it, since this is how
        # compress_branch_helper was called on it.
        while (len(self.in_edges[node.node_id]) == 1 and node.node_id in self.edges
               and len(self.edges[node.node_id]) == 1):
            chain_length += 1
            next_node = self.edges[node.node_id][0] # Since node has a single out-neighbor.
            # Compute time and parameter size are added; latest node's activation_size is used.
            new_node.forward_compute_time += node.forward_compute_time
            new_node.backward_compute_time += node.backward_compute_time
            new_node.activation_size = node.activation_size
            new_node.parameter_size += node.parameter_size
            # If next_node has more than one predecessor, then can't continue merging
            # next_node into new_node.
            if len(self.in_edges[next_node.node_id]) > 1:
                break
            node = next_node
            if node.node_id not in self.edges:
               return new_node, node
        if chain_length == 0:
            return node, node
        if chain_length == 1:
            new_node.node_desc = node.node_desc

        # If node can't be compressed into `new_node` because it has multiple
        # out-neighbors, make sure to compress `node` into `new_node` as well.
        if node.node_id in self.edges and len(self.edges[node.node_id]) > 1:
            new_node.forward_compute_time += node.forward_compute_time
            new_node.backward_compute_time += node.backward_compute_time
            new_node.activation_size = node.activation_size
            new_node.parameter_size += node.parameter_size

        # Return the new_node along with the last merged-in node which is now
        # effectively replaced in the input graph.
        return new_node, node

    def compress_branches(self):
        nodes = self.sources() # Start exploration with the input graph's source node.
        new_gr = Graph() # Create new graph, that will be returned.
        i = 0
        seen_node_ids = set()
        new_node_mapping = dict() # Map old nodes to the new compressed nodes.
        while len(nodes) > 0:
            node = nodes.pop(0)
            if node.node_id in seen_node_ids:
                continue
            if node.node_id in self.edges and len(self.edges[node.node_id]) > 1:
                for out_node in self.edges[node.node_id]:
                    # Each out_node is now a branch that needs to be compressed.
                    compressed_node, old_node = self.compress_branch_helper(
                        out_node, i)
                    i += 1
                    if compressed_node is None:
                        # Now, add an edge between `node` (or the node that replaces `node`)
                        # and `out_node`, since node compression didn't take place.
                        if node.node_id in new_node_mapping:
                            new_gr.add_edge(new_node_mapping[node.node_id], out_node)
                        else:
                            new_gr.add_edge(node, out_node)
                    else:
                        new_node_mapping[old_node.node_id] = compressed_node
                        # Add an edge between `node` (or the node that replaces `node`)
                        # and `compressed_node`.
                        if node.node_id in new_node_mapping:
                            new_gr.add_edge(new_node_mapping[node.node_id], compressed_node)
                        else:
                            new_gr.add_edge(node, compressed_node)
                    if old_node.node_id not in seen_node_ids:
                        nodes.append(old_node)
            else:
                # No branching -- copy graph to output graph.
                if node.node_id in self.edges:
                    for out_node in self.edges[node.node_id]:
                        in_node = node
                        if node.node_id in new_node_mapping:
                             in_node = new_node_mapping[node.node_id]
                        if out_node.node_id in new_node_mapping:
                            new_gr.add_edge(in_node, new_node_mapping[out_node.node_id])
                        else:
                            new_gr.add_edge(in_node, out_node)
                        if out_node.node_id not in seen_node_ids:
                            nodes.append(out_node)
            seen_node_ids.add(node.node_id)
        return new_gr

    def is_series_parallel(self, arch):
        gr_copy = self.copy()
        chain_nodes = gr_copy.chain_nodes()
        while len(chain_nodes) > 0:
            node = chain_nodes[0]
            predecessor = next(iter(gr_copy.in_edges[node.node_id]))
            successor = next(iter(gr_copy.edges[node.node_id]))
            if successor not in gr_copy.edges[predecessor.node_id]:
                gr_copy.add_edge(predecessor, successor)
            del gr_copy.nodes[node.node_id]
            gr_copy.remove_edge(node, successor)
            gr_copy.remove_edge(predecessor, node)
            chain_nodes = gr_copy.chain_nodes()
        gr_copy.to_dot("%s/%s" % (arch, arch))
        return len(gr_copy.nodes) == 2

    def chain_nodes(self):
        chain_nodes = list()
        for node in self.nodes.values():
            if node.node_id in self.edges and len(self.edges[node.node_id]) == 1 \
                and node.node_id in self.in_edges and len(self.in_edges[node.node_id]) == 1:
                chain_nodes.append(node)
        return chain_nodes

    def aggregate(self, sum_activations=False):
        forward_compute_time = 0.0
        backward_compute_time = 0.0
        parameter_size = 0.0
        activation_size = 0.0
        for node in self.nodes.values():
           forward_compute_time += node.forward_compute_time
           backward_compute_time += node.backward_compute_time
           parameter_size += node.parameter_size
           if sum_activations:
               activation_size += node.activation_size
           else:
               if node.node_id not in self.in_edges or len(self.in_edges[node.node_id]) == 0:
                   activation_size += node.activation_size
        return [forward_compute_time, backward_compute_time, parameter_size, activation_size]

    def check_fidelity(self, other):
        self_aggregate = self.aggregate()
        other_aggregate = other.aggregate()
        for i in range(len(self_aggregate)):
            if other_aggregate[i] != 0:
                assert(0.9999 <= (self_aggregate[i] / other_aggregate[i]) <= 1.0001)

    def check_isomorphism(self, other):
        # Hack to check for isomorphism (break ties when exploring out-neighbors with "height"
        # [longest path from one of the sinks]).
        self.populate_heights()
        other.populate_heights()
        self_topological_sort = self.topological_sort()
        other_topological_sort = other.topological_sort()
        assert(len(self_topological_sort) == len(other_topological_sort))

        for (self_node, other_node) in zip(self_topological_sort, other_topological_sort):
            assert(self_node.node_desc == other_node.node_desc)
            if self_node.node_id in self.edges:
                assert(len(self.edges[self_node.node_id]) == len(other.edges[other_node.node_id]))
            if self_node.node_id in self.in_edges:
                assert(len(self.in_edges[self_node.node_id]) == len(other.in_edges[other_node.node_id]))

    def topological_sort(self):
        # Algorithm from https://en.wikipedia.org/wiki/Topological_sorting
        self.sorted_nodes = []
        self.marked_nodes = set()
        self.temporarily_marked_nodes = set()
        nodes = list(self.nodes.values())
        nodes.sort(key=lambda x: x.node_desc)
        for node in nodes:
            if node.node_id in self.marked_nodes:
                continue
            self.topological_sort_helper(node.node_id)
        return [self.nodes[node_id] for node_id in self.sorted_nodes]

    def topological_sort_helper(self, node_id):
        if node_id in self.marked_nodes:
            return
        if node_id in self.temporarily_marked_nodes:
            raise Exception("Graph has a cycle")
        self.temporarily_marked_nodes.add(node_id)
        if node_id in self.edges:
            out_nodes = list(self.edges[node_id])
            out_nodes.sort(key=lambda x: (x.node_desc, x.height))
            for out_node in out_nodes:
                self.topological_sort_helper(out_node.node_id)
        self.marked_nodes.add(node_id)
        self.temporarily_marked_nodes.remove(node_id)
        self.sorted_nodes.insert(0, node_id)

    def predecessors(self, node):
        if node in self._predecessors:
            return self._predecessors[node]
        predecessors = set()
        if node not in self.in_edges:  # Source node
            return predecessors
        for in_node in self.in_edges[node]:
            predecessors.add(in_node)
            predecessors.update(self.predecessors(in_node.node_id))
        self._predecessors[node] = predecessors
        return self._predecessors[node]

    def all_predecessors(self, antichain):
        all_predecessors = set()
        for antichain_node in antichain:
            all_predecessors.update(self.predecessors(antichain_node))
            all_predecessors.add(self.nodes[antichain_node])
        return all_predecessors

    def successors(self, node):
        if node in self._successors:
            return self._successors[node]
        successors = set()
        if not node in self.edges:  # Sink node
            return successors
        for out_node in self.edges[node]:
            successors.add(out_node)
            successors.update(self.successors(out_node.node_id))
        self._successors[node] = successors
        return self._successors[node]

    def augment_antichain(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._augmented_antichains:
            return self._augmented_antichains[antichain_key]
        extra_nodes = set()
        all_predecessors = set()
        for antichain_node in antichain:
            predecessors = self.predecessors(antichain_node)
            all_predecessors = all_predecessors.union(predecessors)
        for antichain_node in antichain:
            predecessors = self.predecessors(antichain_node)
            for predecessor in predecessors:
                for out_node in self.edges[predecessor.node_id]:
                    if out_node not in predecessors and out_node.node_id != antichain_node:
                        extra_nodes.add(predecessor.node_id)
        self._augmented_antichains[antichain_key] = list(extra_nodes) + antichain
        return self._augmented_antichains[antichain_key]

    def deaugment_augmented_antichain(self, augmented_antichain):
        augmented_antichain_key = tuple(sorted(augmented_antichain))
        if augmented_antichain_key in self._deaugmented_augmented_antichains:
            return self._deaugmented_augmented_antichains[augmented_antichain_key]
        nodes_to_remove = set()
        all_successors = set()
        for augmented_antichain_node in augmented_antichain:
            successors = self.successors(augmented_antichain_node)
            for augmented_antichain_node_prime in augmented_antichain:
                if self.nodes[augmented_antichain_node_prime] in successors:
                    nodes_to_remove.add(augmented_antichain_node)
        antichain = list()
        for augmented_antichain_node in augmented_antichain:
            if (augmented_antichain_node not in nodes_to_remove and \
                augmented_antichain_node not in antichain):
                antichain.append(augmented_antichain_node)
        self._deaugmented_augmented_antichains[augmented_antichain_key] = antichain
        return self._deaugmented_augmented_antichains[augmented_antichain_key]

    def is_next_antichain(self, augmented_antichain, new_node):
        successors = self.successors(new_node)
        augmented_antichain_set = set(augmented_antichain)
        for successor in successors:
            if successor.node_id in augmented_antichain_set:
                return False
        return True

    def construct_antichain(self, augmented_antichain, old_node, new_node):
        new_antichain = [x if x != old_node else new_node for x in augmented_antichain]
        return self.deaugment_augmented_antichain(new_antichain)

    def next_antichains(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._next_antichains:
            return self._next_antichains[antichain_key]

        next_antichains = []
        antichain_set = set(antichain)
        augmented_antichain = self.augment_antichain(antichain)
        for augmented_antichain_node in augmented_antichain:
            next_nodes = self.edges[augmented_antichain_node] if augmented_antichain_node in self.edges else []
            for next_node in next_nodes:
                if next_node.node_id in antichain_set:
                    continue
                if self.is_next_antichain(augmented_antichain, next_node.node_id):
                    next_antichain = self.construct_antichain(augmented_antichain,
                                                              augmented_antichain_node,
                                                              next_node.node_id)
                    next_antichains.append(next_antichain)
        self._next_antichains[antichain_key] = next_antichains
        return self._next_antichains[antichain_key]

    def antichain_dag(self):
        if self._antichain_dag is not None:
            return self._antichain_dag

        antichain_dag = Graph()
        antichain_id = 0
        antichain = [self.sources()[0].node_id]
        source_node = AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(antichain))
        antichain_dag.source = source_node
        antichain_queue = [antichain]
        antichain_mapping = {tuple(sorted(antichain)): source_node}

        while len(antichain_queue) > 0:
            antichain = antichain_queue.pop(0)
            antichain_key = tuple(sorted(antichain))
            if antichain_key in self._next_antichains:
                continue
            next_antichains = self.next_antichains(antichain)
            for next_antichain in next_antichains:
                next_antichain_key = tuple(sorted(next_antichain))
                if next_antichain_key not in antichain_mapping:
                    antichain_id += 1
                    next_antichain_node = AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(next_antichain))
                    antichain_mapping[next_antichain_key] = next_antichain_node
                antichain_dag.add_edge(antichain_mapping[antichain_key],
                                       antichain_mapping[next_antichain_key])
                antichain_queue.append(next_antichain)

        self._antichain_dag = antichain_dag
        return antichain_dag

    def __str__(self):
        strs = []
        for node in self.nodes.values():
            strs.append(str(node))
        for node in self.nodes.values():
            if node.node_id not in self.in_edges:
                continue
            for in_node in self.in_edges[node.node_id]:
                strs.append("\t%s -- %s" % (in_node.node_id, node.node_id))
        return "\n".join(strs)

    @staticmethod
    def from_str(graph_str):
        gr = Graph()
        graph_str_lines = graph_str.strip().split('\n')
        for graph_str_line in graph_str_lines:
            if not graph_str_line.startswith('\t'):
                node = Node.from_str(graph_str_line.strip())
                gr.nodes[node.node_id] = node
            else:
                [in_node_id, node_id] = graph_str_line.strip().split(" -- ")
                if node_id not in gr.in_edges:
                    gr.in_edges[node_id] = [gr.nodes[in_node_id]]
                else:
                    gr.in_edges[node_id].append(gr.nodes[in_node_id])
                if in_node_id not in gr.edges:
                    gr.edges[in_node_id] = [gr.nodes[node_id]]
                else:
                    gr.edges[in_node_id].append(gr.nodes[node_id])
        return gr

    def to_dot(self, arch):
        dot = graphviz.Digraph()
        for node in self.nodes.values():
            node_desc = "%s\n[forward_compute_time=%.3f,backward_compute_time=%.3f,activation_size=%s,parameter_size=%.1f]" % (
                node.node_desc, node.forward_compute_time, node.backward_compute_time,
                node.activation_size, node.parameter_size)
            if node.stage_id is not None:
                color = self._colors[node.stage_id % len(self._colors)]
                dot.node(node.node_id, node_desc,
                   color=color, style='filled')
            else:
                dot.node(node.node_id, node_desc)
        for node in self.nodes.values():
            if node.node_id not in self.edges:
                continue
            for out_node in self.edges[node.node_id]:
                dot.edge(node.node_id, out_node.node_id)
        dot.render(arch)

    def plot_cdfs(self, cdfs, output_directory):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import seaborn as sns
        matplotlib.rc('text', usetex=True)
        sns.set_style('ticks')
        sns.set_style({'font.family':'sans-serif'})
        flatui = ['#002A5E', '#FD151B', '#8EBA42', '#348ABD', '#988ED5', '#777777', '#8EBA42', '#FFB5B8']
        sns.set_palette(flatui)
        paper_rc = {'lines.linewidth': 2, 'lines.markersize': 10}
        sns.set_context("paper", font_scale=3,  rc=paper_rc)
        current_palette = sns.color_palette()

        plt.figure(figsize=(10, 4))
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

        labels = ["Compute", "Activations", "Parameters"]
        for i in range(3):
            cdf = [cdfs[j][i] for j in range(len(cdfs))]
            ax.plot(range(len(cdfs)), cdf,  label=labels[i],
                    linewidth=2)
        ax.set_xlim([0, None])
        ax.set_ylim([0, 100])

        ax.set_xlabel("Layer ID")
        ax.set_ylabel("CDF (\%)")
        plt.legend()

        with PdfPages(os.path.join(output_directory, "cdf.pdf")) as pdf:
            pdf.savefig(bbox_inches='tight')

    def plot_bar_graph(self, all_values, ylabel, legend, output_template, output_directory):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import seaborn as sns
        matplotlib.rc('text', usetex=True)
        sns.set_style('ticks')
        sns.set_style({'font.family':'sans-serif'})
        flatui = ['#002A5E', '#FD151B', '#8EBA42', '#348ABD', '#988ED5', '#777777', '#8EBA42', '#FFB5B8']
        sns.set_palette(flatui)
        paper_rc = {'lines.linewidth': 2, 'lines.markersize': 10}
        sns.set_context("paper", font_scale=3,  rc=paper_rc)
        current_palette = sns.color_palette()

        labels = ["Compute_times", "Activations", "Parameters"]
        ylabels = ["Compute time\n(milliseconds)", "Activation size\n(bytes)", "Parameter size\n(bytes)"]
        for i in range(3):
            plt.figure(figsize=(10, 4))
            ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

            values_sum = sum([all_values[j][i] for j in range(len(all_values))])
            # Truncate the number of values plotted, since bars become very thin otherwise.
            values = [all_values[j][i] for j in range(len(all_values))][:400]
            if legend:
                ax.bar(range(len(values)), values, label="Sum: %.1f" % values_sum)
            else:
                ax.bar(range(len(values)), values)
            ax.set_xlim([0, None])
            ax.set_ylim([0, None])

            ax.set_xlabel("Layer ID")
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel(ylabels[i])
            if legend:
                plt.legend()

            with PdfPages(os.path.join(output_directory,
                          (output_template % labels[i].lower()))) as pdf:
                pdf.savefig(bbox_inches='tight')

    def render_bar_graphs_and_cdfs(self, output_directory):
        topological_ordering = self.topological_sort()[1:]  # Skip input node.
        cdfs = []
        raw_values = []
        pdfs = []
        for node in topological_ordering:
            activation_size = node.activation_size
            if isinstance(activation_size, list):
                activation_size = sum(activation_size)
            if len(cdfs) == 0:
                cdfs.append([node.forward_compute_time + node.backward_compute_time,
                             activation_size, node.parameter_size])
            else:
                cdfs.append([cdfs[-1][0] + node.forward_compute_time + node.backward_compute_time,
                             cdfs[-1][1] + activation_size,
                             cdfs[-1][2] + node.parameter_size])

        for node in topological_ordering:
            activation_size = node.activation_size
            if isinstance(activation_size, list):
                activation_size = sum(activation_size)
            raw_values.append((node.forward_compute_time + node.backward_compute_time,
                               activation_size, node.parameter_size))
        self.plot_bar_graph(raw_values, None, True, "%s.pdf", output_directory)

        for node in topological_ordering:
            activation_size = node.activation_size
            if isinstance(activation_size, list):
                activation_size = sum(activation_size)
            pdfs.append(((node.forward_compute_time + node.backward_compute_time) / (cdfs[-1][0] / 100.0),
                         activation_size / (cdfs[-1][1] / 100.0),
                         node.parameter_size / (cdfs[-1][2] / 100.0)))
        self.plot_bar_graph(pdfs, "PDF (\%)", False, "%s_pdf.pdf", output_directory)

        for i in range(len(cdfs)):
            cdfs[i][0] /= (cdfs[-1][0] / 100.0)
            cdfs[i][1] /= (cdfs[-1][1] / 100.0)
            cdfs[i][2] /= (cdfs[-1][2] / 100.0)
        self.plot_cdfs(cdfs, output_directory)


class Node(object):
    def __init__(self, node_id, node_desc="", forward_compute_time=0.0,
                 backward_compute_time=0.0, activation_size=0.0, parameter_size=0.0,
                 stage_id=None):
        self.node_id = node_id
        self.node_desc = node_desc
        self.forward_compute_time = forward_compute_time
        self.backward_compute_time = backward_compute_time
        self.activation_size = activation_size
        self.parameter_size = parameter_size
        self.stage_id = stage_id
        self.depth = None
        self.height = None

    def set_stage_id(self, stage_id):
        self.stage_id = stage_id

    def __str__(self):
        stage_id_str = " -- stage_id=%d" % self.stage_id if self.stage_id is not None else ""
        node_desc = self.node_desc.replace('\n', "")
        activation_size = ("%s" % self.activation_size).replace(", ", "; ")
        return "%s -- %s -- forward_compute_time=%.3f, backward_compute_time=%.3f, activation_size=%s, parameter_size=%.3f%s" % (
            self.node_id, node_desc, self.forward_compute_time, self.backward_compute_time,
            activation_size, self.parameter_size, stage_id_str)

    @staticmethod
    def from_str(node_str):
        node_str_tokens = node_str.strip().split(" -- ")
        node_id = node_str_tokens[0]
        node_desc = node_str_tokens[1]
        node_metadata = node_str_tokens[2]
        stage_id = None
        if len(node_str_tokens) > 3:
            stage_id = int(node_str_tokens[3].split("=")[1])
        [forward_compute_time, backward_compute_time, activation_size, parameter_size] = node_metadata.split(", ")
        forward_compute_time = float(forward_compute_time.split("=")[1])
        backward_compute_time = float(backward_compute_time.split("=")[1])
        if "[" in activation_size:
            activation_size = activation_size.split("=")[1]
            activation_size = sum([float(x) for x in activation_size.lstrip("[").rstrip("]").split("; ")])
        else:
            activation_size = float(activation_size.split("=")[1])
        parameter_size = float(parameter_size.split("=")[1])
        return Node(node_id, node_desc, forward_compute_time=forward_compute_time,
                    backward_compute_time=backward_compute_time, activation_size=activation_size,
                    parameter_size=parameter_size, stage_id=stage_id)

class AntichainNode(Node):
    def __init__(self, node_id, antichain, node_desc=""):
        self.antichain = antichain
        self.output_activation_size = 0.0
        super(AntichainNode, self).__init__(node_id, node_desc)

    def __str__(self):
        return "%s -- %s" % (self.node_id, self.antichain)
