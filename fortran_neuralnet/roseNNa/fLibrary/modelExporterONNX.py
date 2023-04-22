# Import packages
import torch
import torch.nn as nn
import sys
import argparse
import os
import timeit
import numpy as np
import pathlib
import getopt

# Import user-defined architectures here
from architectures import *

# TODO
# - get activation function
# - use other architectures conveniently
# - add more documentation
# - improve arg parse command line
# - improve paths with OS

#############
# CONSTANTS #
#############

# Define global constants for preprocessing
FILEPATH = "../goldenFiles/"
# ^ where the files will get located, by default same folder
NAME  = ""
# ^ the name of the network and its folder within goldenfiles
VERBOSE = True
# ^ whether we print or not during preprocessing
INPUT_SIZE = 0
# ^ number of input nodes of the network
produce = True
# ^ TODO - what precisely?

#######################
# AUXILIARY FUNCTIONS #
#######################

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()

#####################
# READING ARGUMENTS #
#####################

# Command line arguments for processing
parser = argparse.ArgumentParser()
parser.add_argument('--name',"-n", required=True, help="Please provide name of your pretrained model (must also be goldenFiles dir name).")
parser.add_argument('--input-size',"-i", help="Please provide the size of the input of the network")

# Parse them
args = parser.parse_args()

# Get them
NAME = args.name.strip()
INPUT_SIZE = int(args.input_size)
# Update the filepath
FILEPATH = FILEPATH + "/" + NAME + "/"
if VERBOSE:
    print(f"Reading model from and saving onnx to: {FILEPATH}")
# Try to load PyTorch object
try:
    model = torch.load(FILEPATH + NAME + ".pt")
except AttributeError as error:
    print(error)
    # code to handle the AttributeError
    print("An AttributeError occurred - are you sure that your code for your PyTorch model is in architectures.py and loaded in modelExporterONNX?")
    exit()


#################
# Preprocessing #
#################

# Get input
print("Observed input")
inp = torch.ones(1, INPUT_SIZE)
print(inp)

if produce:
    with open("inputs.fpp",'w') as f:
        inputs = inp.flatten().tolist()
        inpShapeDict = {'inputs': list(inp.shape)}
        inpDict = {'inputs':inputs}
        f.write(f"""#:set inpShape = {inpShapeDict}""")
        f.write("\n")
        f.write(f"""#:set arrs = {inpDict}""")
        f.write("\n")
        f.write("a")

# Get outputs
logits = model(inp)
# Print
if VERBOSE:
    print("Observed output")
    print(logits.flatten().tolist())

with open(FILEPATH + NAME + ".txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))

###############
# ONNX EXPORT #
###############

# Finally, export the model and weights as ONNX files
torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  FILEPATH + NAME + ".onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  FILEPATH + NAME + "_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )