"""
This code is adapted to this project based on the answer of user
"LetterRip" (https://stackoverflow.com/users/2384638/letterrip) in the following thread of stackoverflow:
https://stackoverflow.com/questions/12367801/finding-all-cycles-in-undirected-graphs
"""

cycles = []


def find_the_backbone_cycle_of_robot_network(graph):
    cycles = find_cycles_of_robot_network(graph)
    return max(cycles, key=len)

def find_cycles_of_robot_network(graph):
    for edge in graph:
        for node in edge:
            findNewCycles([node], graph)
    return cycles

def findNewCycles(path, graph):
    start_node = path[0]
    next_node= None
    sub = []

    #visit each edge and each node of each edge
    for edge in graph:
        node1, node2 = edge
        if start_node in edge:
                if node1 == start_node:
                    next_node = node2
                else:
                    next_node = node1
                if not visited(next_node, path):
                        # neighbor node not on path yet
                        sub = [next_node]
                        sub.extend(path)
                        # explore extended path
                        findNewCycles(sub, graph);
                elif len(path) > 2  and next_node == path[-1]:
                        # cycle found
                        p = rotate_to_smallest(path);
                        inv = invert(p)
                        if isNew(p) and isNew(inv):
                            cycles.append(p)

def invert(path):
    return rotate_to_smallest(path[::-1])

#  rotate cycle path such that it begins with the smallest node
def rotate_to_smallest(path):
    n = path.index(min(path))
    return path[n:]+path[:n]

def isNew(path):
    return not path in cycles

def visited(node, path):
    return node in path