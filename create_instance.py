import json

instance_file = open('instances/10OSC10_1_HD1.txt', 'r')  # the instance we want to use
lines = instance_file.readlines()
instance_file.close()

# completing the parameters given by the text file
nb_machine = int(lines[1][0])
nb_job = int(lines[2][0])
int_lines = []
for line in lines[1:]:
    int_lines.append(list(map(int, line.split())))  # convert string into list of int
nb_machine = int_lines[0][0]
nb_job = int_lines[1][0]
processing_times = int_lines[2:2 + nb_machine]
Adj_matrix = int_lines[2 + nb_machine:]

# completing the remaining parameters
rand = False  # if you want to use a random Erdos-Reyni Graph
mutation_probability = 0.2
edge_probability = 0.5

# the complete instance we store in a json file
instance = {
    "nb_schedule": 30,
    "nb_machine": nb_machine,
    "nb_job": nb_job,
    "processing_times": processing_times,
    "mutation_probability": mutation_probability,
    "Graph": {"rand": False, "edge_probability": edge_probability, "Adj_matrix": Adj_matrix}
}

with open('instance.txt', 'w') as outfile:
    json.dump(instance, outfile)
