#!/bin/bash

#!/bin/bash

# Update package list and install required packages
NUM_NODES=$1
NUM_SCHEDULER_DATASTORE=1
chmod -R +rwx /users/${USER}
cd /users/${USER}

echo "Installing required packages..."
apt update
echo y | sudo apt install python3-pip thrift-compiler stress openjdk-17-jdk openjdk-17-jre vim maven

# Install Python package
echo "Installing python package..."
pip install optparse-pretty

# Clone the Git repository
echo "Cloning the dodoor repository..."
git clone https://github.com/AKafakA/dodoor.git
git config --global --add safe.directory /users/${USER}/dodoor

# Checkout the specific branch and rebuild
cd dodoor
echo "Checking out the exp branch and rebuilding..."
git checkout exp
sh rebuild.sh

# Run the configuration generator script
#echo "Running the configuration generator script..."
#python3 deploy/python/scripts/config_generator.py --num-nodes $NUM_NODES --scheduler-ports 20503,20504,20505,20506 --cores 8 --memory 61440

echo "Setup completed successfully!"
