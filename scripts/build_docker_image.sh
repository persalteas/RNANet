#!/bin/bash

# echo "WARNING: The purpose of this file is to document how the docker image was built.";
# echo "You cannot execute it directly, because of licensing reasons. Please get your own";
# echo "DSSR 2.0 executable at http://innovation.columbia.edu/technologies/CU20391";
# echo "and place it in this folder.";
# exit 0;

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

####################################################### Dependencies ##############################################################

# The $THISDIR folder is supposed to contain the x3dna-dssr executable
cp `which x3dna-dssr` $THISDIR

######################################################## Build Docker image ######################################################
# Execute the Dockerfile and build the image
docker build -t rnanet:latest ..

############################################################## Cleaning ##########################################################
rm x3dna-dssr

# to run, use something like:
# docker run -v /home/persalteas/Data/RNA/3D/:/3D -v /home/persalteas/Data/RNA/sequences/:/sequences -v /home/persalteas/labo/:/runDir persalteas/rnanet [ additional options here ]
# Without additional options, this runs a standard pass with known issues support, log output, and no statistics. The default resolution threshold is 4.0 Angstroms.

