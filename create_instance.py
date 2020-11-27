import json

instance_file = open('instances/20OSC20_1_MD1.txt', 'r')  # the instance we want to use
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
with open('parameters.json') as parameters:
    param = json.load(parameters)

# the complete instance we store in a json file
instance = {
    "nb_schedule": param["nb_schedule"],
    "nb_machine": nb_machine,
    "nb_job": nb_job,
    "processing_times": processing_times,
    "mutation_probability": param["mutation_probability"],
    "Graph": {"rand": param["rand_graph"], "edge_probability": param["edge_probability"], "Adj_matrix": Adj_matrix}
}
with open('instance.json', 'w') as outfile:
    json.dump(instance, outfile)
