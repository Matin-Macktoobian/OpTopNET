import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import itertools
from collections import Counter
import csv
from datetime import datetime

from TopoNET_CycleFinder import find_the_backbone_cycle_of_robot_network

"""
Hyper-parameters

number_of_configurations: the number of the records of the dataset
number_of_robots:         the number of robots existing in the network                
zone_range:               the length of each dimension of the network's field           
connectivity_threshold:   the maximum allowed distance between robots violation of which disconnects the network
epsilon:                  the width of the tension bound 
scale_factor:             the factor to further control the connectivity distribution of robots
"""
number_of_configurations = 10
number_of_robots = 10
zone_range = 1
connectivity_threshold = 0.5
epsilon = 0.1
scale_factor = 0.1



"""
This class creates robot instances supported by the native computational functions of the class.
"""
class Robot():
    id = itertools.count(start=1)

    def __init__(self, location = None, reliable_id_set = [], critical_id_set = [], degree = 0):
        self.id = next(Robot.id)
        self.location = np.random.uniform(0, zone_range, size=(1, 2)).tolist()[0]
        self.reliable_id_set = reliable_id_set
        self.critical_id_set = critical_id_set
        self.degree = degree


    """
    This function makes the instances of Robot class iterable.
    """
    def __iter__(self):
        for _ in self.__dict__.values():
            yield _


    """
    This function computes the distance between two robots.
    """
    def compute_distance(self, other_robot):
        return sqrt((self.location[0] - other_robot.location[0]) ** 2 + (self.location[1] - other_robot.location[1]) ** 2)


    """
    This function checks whether new_robot is connected to the network of the previously-available robots.
    """
    def check_connectivity(self, robot_network):
        if (len(robot_network) == 0):
            return True
        else:
            return any([robot for robot in robot_network if self.compute_distance(robot) < scale_factor*(connectivity_threshold + epsilon)])


    """
    This function computes the reliable sets associated with the robots of a network.
    """
    def compute_reliable_set(self):
        other_robots = robot_network.copy()
        other_robots.remove(self)
        self.reliable_id_set = [other_robot.id for other_robot in other_robots
                             if self.compute_distance(other_robot) <= scale_factor*connectivity_threshold]


    """
    This function computes the critical sets associated with the robots of a network.
    """
    def compute_critical_set(self):
        other_robots = robot_network.copy()
        other_robots.remove(self)
        self.critical_id_set = [other_robot.id for other_robot in other_robots
                                if ((self.compute_distance(other_robot) > scale_factor*connectivity_threshold) and
                                    self.compute_distance(other_robot) <= scale_factor*(connectivity_threshold + epsilon))]


    """
    This function sets the computed relibale and critical set of a robot network. 
    """
    def set_reliable_and_critical_sets(self):
        self.compute_reliable_set()
        self.compute_critical_set()


    """
    This overridden function defines the measure of comparability between two instances of Robot class.
    """
    def __lt__(self, other):
        if (len(self.reliable_id_set) < len(other.reliable_id_set)):
            return True
        elif (len(self.reliable_id_set) == len(other.reliable_id_set)):
            return len(self.critical_id_set) < len(other.critical_id_set)


    """
    This overridden function provides a formatted output for the objects of Robot class. 
    """
    def __str__(self):
        return "id: {!s:<5}, location: {!s:<45}, reliable set: {!s:<22}, critical set: {!s:<30}, degree: {}".format(
            self.id, self.location, self.reliable_id_set, self.critical_id_set, self.degree)

#############################################################################################################

"""
This class represents cycle topology objects each of which includes a backbone cycle and a set of clusters associated
with those robots which does not reside on the backbone cycle.
"""
class CycleTopology():
    def __init__(self, backbone_cycle = [], clusters = []):
        #backbone_cycle is the list of the robot constructing the backbone cycle.
        self.backbone_cycle = backbone_cycle
        #clusters include [a, b] lists in which orphan robot 'a' is connected to backbone robot 'b'.
        self.clusters = clusters


    """
    This overridden function makes the objects of the class iterable.
    """
    def __iter__(self):
        for _ in self.__dict__.values():
            yield _


    """
    This overridden function provides a formatted output for the objects of CycleTopology class. 
    """
    def __str__(self):
        return "backbone cycle: {!s:<5}, clusters: {}".format(self.backbone_cycle, self.clusters)

#############################################################################################################

"""
This function plot a robot network configuration.
"""
def plot_locations_of_robot_network(robot_network):
    ### removing the extra nesting of the robot's list
    formatted_robot_network = [[attribute for attribute in my_class] for my_class in robot_network]
    locations = [[subitem for subitem in item[1]] for item in formatted_robot_network]
    plt.scatter(np.asarray(locations)[:, 0], np.asarray(locations)[:, 1])
    plt.xlim([0, zone_range])
    plt.ylim([0, zone_range])
    plt.show()


"""
This function create a robot network and computes all of the properties of its robot members.
"""
def create_robot_network():
    robot_network = []
    first_robot = Robot()
    robot_network.append(first_robot)
    counter = 1
    while counter != number_of_robots:
        next_robot = Robot()
        if next_robot.check_connectivity(robot_network):
            robot_network.append(next_robot)
            counter += 1
        else:
            del next_robot
            Robot.id = itertools.count(counter + 1)
    return robot_network


"""
This function sort robots based on the criticality of their connectedness. The larger the cardinality of a reliable set, 
the higher its priority. If reliable sets' cardinalities are equal, then those of the critical sets decide, say, the larger
one will be prioritized. This function plays a crucial rule in the expected functionality of the 
synthesize_ring_topology_for_robot_network(robot_network) function. For more information, please refer to that function's 
comments.
"""
def sort_robot_network(robot_network):
    sorted_robot_network = sorted(robot_network, reverse=True)
    return sorted_robot_network


"""
This function generates the backbone cycle of a robot network. 
"""
def generate_backbone_cycle_links(robot_network):
    backbone_cycle_links = []
    for robot in robot_network:
        for item in robot.reliable_id_set:
            if [item, robot.id] not in backbone_cycle_links:
                backbone_cycle_links.append([robot.id, item])
    return backbone_cycle_links


"""
This function initializes the degree values of the links of a cycle backbone.
"""
def update_degrees_of_robot_network(robot_network, links):
    flat_links = [item for sublist in links for item in sublist]
    counter = Counter(flat_links)
    for robot in robot_network:
        for key, value in counter.items():
            if robot.id == key:
                robot.degree = value


"""
This function returns the relibale peers (of an orphan robot) on the backbone cycle of their topology. 
"""
def get_reliable_peers_in_cycle(orphan_robot, backbone_cycle):
    reliable_id_list = []
    for id in orphan_robot.reliable_id_set:
        if id in backbone_cycle:
            reliable_id_list.append(id)
    return reliable_id_list


"""
This function returns the critical peers (of an orphan robot) on the backbone cycle of their topology. 
"""
def get_critical_peers_in_cycle(orphan_robot, backbone_cycle):
    critical_id_list = []
    for id in orphan_robot.critical_id_set:
        if id in backbone_cycle:
            critical_id_list.append(id)
    return critical_id_list


"""
This function employs a heuristic estimation for the case none of the peers of an orphan robot has been already assigned to 
a backbone robot. Using this heuristics, the nearest backbone robot to an orphan robot is found.
"""
def find_the_nearest_backbone_robot(orphan_robot, robot_network, backbone_cycle):
    backbone_robots = [robot for robot in robot_network for id in backbone_cycle if robot.id == id]
    nearest_backbone_robot_id = min(robot.id for robot in backbone_robots
                                    if orphan_robot.compute_distance(robot))
    return nearest_backbone_robot_id


"""
This function finds the nearesr least-connected peer of a orphan robot residing on the backbone cycle of their topology. 
"""
def get_the_nearest_least_connected_peer_in_cycle(robot_network, id_list, orphan_robot):
    minimum_degree = min(robot.degree for robot in robot_network for id in id_list if robot.id == id)
    robots_cycle_with_minimum_degree = [robot for robot in robot_network if robot.degree == minimum_degree]
    # If the minimum-degree robot is unique, so the distance constraint may be ignored.
    if (len(robots_cycle_with_minimum_degree) == 1):
        return robots_cycle_with_minimum_degree[0].id
    # else if there are more than one robot with the minimum degree, pick the nearest one to the orphan robot.
    else:
        minimum_distance = min(orphan_robot.compute_distance(robot) for robot in robots_cycle_with_minimum_degree)
        nearest_robots_cycle_with_minimum_degree = [robot for robot in robots_cycle_with_minimum_degree
                                                             if orphan_robot.compute_distance(robot) == minimum_distance]
        return nearest_robots_cycle_with_minimum_degree[0].id


"""
The function below computes the cycle topology of a robot network. Please refer to the internal comments to better understand
the functionality of this routine. (Note that the robot network is sorted in a descending order first by the cardinality of 
its reliable_id_set and then by that of its critical_id_set. So, this order is also conserved while processing of orphan robots
to connect them to the backbone cycle of the robot network's topology. Accordingly, the earlier orphan robots are closer to \
the backbone cycle compared to those which come later to be processed.)
"""
def synthesize_cycle_topology_for_robot_network(robot_network):
    cycle_topology = CycleTopology([], [])

    #List of the the links of the backbone cycle
    backbone_cycle_links = generate_backbone_cycle_links(robot_network)

    update_degrees_of_robot_network(robot_network, backbone_cycle_links)

    cycle_topology.backbone_cycle = find_the_backbone_cycle_of_robot_network(backbone_cycle_links)

    #Add each item of backbone cycle as its own cluster to the clusters list
    for item in cycle_topology.backbone_cycle:
        cycle_topology.clusters.append([item, item])

    #Orphan robots are those which do not belong to the robot network cycle.
    orphan_robots = [robot for robot in robot_network if (robot.id not in cycle_topology.backbone_cycle)]
    for orphan_robot in orphan_robots:
        if any(orphan_robot.reliable_id_set):
            # If there is any of the reliable peers of the orphan robot which belong to the cycle
            reliable_temp = get_reliable_peers_in_cycle(orphan_robot, cycle_topology.backbone_cycle)
            if any(reliable_temp):
                # the id captured below is indeed the cluster id of the orphan robot
                id_of_the_nearest_reliable_robot_cycle_with_minimum_degree = get_the_nearest_least_connected_peer_in_cycle(
                    robot_network, reliable_temp, orphan_robot)
                # connect the orphan robot to the reliable peer of which owns the least degree
                cycle_topology.clusters.append(
                    [orphan_robot.id, id_of_the_nearest_reliable_robot_cycle_with_minimum_degree])
            # In this case, none of the reliable peers of the orphan robot belong to the cycle
            else:
                #Does the orphan robot have any reliable peer which has been already assigned to a cluster
                temp = [cluster[1] for id in orphan_robot.reliable_id_set for cluster in cycle_topology.clusters if
                        cluster[0] == id]
                # If so, get the list of clusters associated with those peers and assign the orphan robot to the first one of them
                if any(temp):
                    cycle_topology.clusters.append([orphan_robot.id, temp[0]])
                # Otherwise, Do the same this time for the critical peers of it
                else:
                    temp1 = [cluster[1] for id in orphan_robot.critical_id_set for cluster in cycle_topology.clusters if
                            cluster[0] == id]
                    if any(temp1):
                        cycle_topology.clusters.append([orphan_robot.id, temp1[0]])
                    else:
                        #If that set is empty, then use the heurisics below, say assign the orphan robot to its nearest backbone
                        #robot
                        nearest_backbone_robot_id = find_the_nearest_backbone_robot(
                            orphan_robot, robot_network, cycle_topology.backbone_cycle)
                        cycle_topology.clusters.append([orphan_robot.id, nearest_backbone_robot_id])
        elif any(orphan_robot.critical_id_set):
            critical_temp = get_critical_peers_in_cycle(orphan_robot, cycle_topology.backbone_cycle)
            # If there is any of the critical peers of the orphan robot which belong to the cycle
            if any(critical_temp):
                # the id captured below is indeed the cluster id of the orphan robot
                id_of_the_nearest_critical_robot_cycle_with_minimum_degree = get_the_nearest_least_connected_peer_in_cycle(
                    robot_network, critical_temp, orphan_robot)
                # connect the orphan robot to the critical peer of which owns the least degree
                cycle_topology.clusters.append(
                    [orphan_robot.id, id_of_the_nearest_critical_robot_cycle_with_minimum_degree])
            # In this case, none of the critical peers of the orphan robot belong to the cycle
            else:
                #Get its critical peers which been already assigned to clusters
                temp = [cluster[1] for id in orphan_robot.critical_id_set for cluster in cycle_topology.clusters if cluster[0] == id]
                #Assign the orphan robot to the cluster
                cycle_topology.clusters.append([orphan_robot.id, temp[0]])
    cycle_topology.clusters = sorted(cycle_topology.clusters, key=lambda id_and_name: id_and_name[0])
    return cycle_topology


"""
This function extracts the locations of the robots of the network.
"""
def extract_robot_locations(robot_network):
    raw_locations = [robot.location for robot in robot_network]
    flat_list = [item for sublist in raw_locations for item in sublist]
    return flat_list


"""
This function extracts the cycle clusters of the topology of a robot network.
"""
def extract_cycle_clusters(cycle_topology):
    return [item[1] for item in cycle_topology.clusters]


"""
This helper function creates the header of the dataset csv fine.
"""
def create_header(number_of_robots):
    header = []
    for i in range(1, number_of_robots + 1):
        header.append("X" + str(i))
        header.append("Y" + str(i))
    for i in range(1, number_of_robots + 1):
        header.append("C" + str(i))
    return header


"""
This function wraps the data into a csv dataset file/
"""
def create_cycle_Topo_dataset(robots_locations, cycles_clusters):
    header = create_header(len(robots_locations[0])//2)

    with open("cycle_Topo_dataset_"+str(len(robots_locations[0])//2)+"___"+datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(header)
        for robot_location, cycle_cluster in zip(robots_locations, cycles_clusters):
            writer.writerow(robot_location + cycle_cluster)


"""
This function logs the progression of the accumulation of records in the csv dataset.
"""
def log_the_dataset_creation_process(counter):
    if counter%10 == 0 and counter > 0:
        print("The configuration number {}/{} is just added to the dataset.".format(counter, number_of_configurations))

###################################################################################################################
###################################################################################################################
###################################################################################################################

"""
Here is the main function of this data generator.
"""
if __name__ == "__main__":

    robots_locations = []
    cycles_clusters = []

    counter = 1

    while counter < number_of_configurations + 1:

        robot_network = create_robot_network()

        for robot in robot_network:
            robot.set_reliable_and_critical_sets()

        sorted_robot_network = sort_robot_network(robot_network)

        cycle_topology = synthesize_cycle_topology_for_robot_network(sorted_robot_network)

        robot_location = extract_robot_locations(sorted_robot_network)
        cycle_cluster = extract_cycle_clusters(cycle_topology)

        robots_locations.append(robot_location)
        cycles_clusters.append(cycle_cluster)

        Robot.id = itertools.count(1)

        log_the_dataset_creation_process(counter)

        counter = counter + 1

    create_cycle_Topo_dataset(robots_locations, cycles_clusters)