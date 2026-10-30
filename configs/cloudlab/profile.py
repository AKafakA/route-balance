"""The CloudLab profile of the paper's cluster (geni-lib): 13 bare-metal nodes at CloudLab Wisconsin on one LAN,
3 c4130 (four V100 each), 8 d7525 (one A30 each) and 2 d8545 (four A100 each), the paper's 28 GPUs, each booting the
cluster's default Ubuntu image. Create a profile from this file on the CloudLab portal and instantiate it; every node
then runs scripts/setup.sh (stage 1 of docs/reproduce/).
"""
import random

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg

# Create a portal context.
pc = portal.Context()

executor_nodes_mapping = {
    "c4130": 3,
    "d7525": 8,
    "d8545": 2,
}

num_nodes = sum(executor_nodes_mapping.values())

request = pc.makeRequestRSpec()

# Add a raw PC to the request.
link = request.Link()
link._best_effort = True

for i in range(0, num_nodes):
    node = request.RawPC("node" + str(i))
    hardware_type = random.choice(list(executor_nodes_mapping.keys()))
    while executor_nodes_mapping[hardware_type] <= 0:
        hardware_type = random.choice(list(executor_nodes_mapping.keys()))
    node.hardware_type = hardware_type
    executor_nodes_mapping[hardware_type] -= 1
    link.addNode(node)

pc.printRequestRSpec(request)
