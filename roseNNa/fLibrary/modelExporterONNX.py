# Import packages
import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
import pathlib
import sys, getopt

# Import user-defined architectures here
from architectures import FeedForwardNetwork

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

# # Two auxiliary functions are needed to load the weights from the state dict into a fresh network
# def get_hidden_size_from_state_dict(state_dict):
#     """
#     Finds the sizes of the two hidden layers of our 2-layer architecture given a state dict.
#     :param state_dict: State dict of saved parameters
#     :return: h, sizes of hidden layers
#     """
#
#     # Return variable is a list
#     h = []
#     # Iterate over the state dict, look at weight matrices and save their second shape param (input)
#     for key in state_dict:
#         if key[-6:] == "weight":
#             print(state_dict[key].shape)
#             shape = state_dict[key].shape
#             h.append(shape[-1])
#
#     h.append(shape[0])
#
#     # We discard the first one, as that corresponds to the input
#     return h

#
# def create_nn(state_dict):
#     """
#     Create a NeuralNetwork object if given a dictionary of the weights, with correct sizes for hidden layers.
#     :param state_dict: State dictionary containing the weights of the neural network
#     :return:
#     """
#     h = get_hidden_size_from_state_dict(state_dict)
#     print(h)
#     activation_func = nn.Sigmoid
#     model = FeedForwardNetwork(h=h, activation_function=activation_func)
#     model.load_state_dict(state_dict)
#
#     return model

#####################
# READING ARGUMENTS #
#####################

# Check command line arguments for processing
if len(sys.argv) > 1:

    # First, read the options
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("-") or arg.startswith("--"):
            if arg == "-h" or arg == "--help":
                print("Usage: python modelExporterONNX.py [options]")
                print("First argument must be PyTorch model saved as .pt")
                print("Options:")
                print("  -h,  --help         Show this help message and exit")
                print("  -v,  --verbose      Enable verbose mode")
                # print("  -fn, --filename     Specify output filenames")
            elif arg == "-v" or arg == "--verbose":
                VERBOSE = True
            # elif arg == "-fn" or arg == "--filename":
            #     FILENAME = sys.argv[i + 1]
            elif arg == "-n":
                produce = False

    # Check default first argument: the first argument has to be the name of the model = name of dir inside goldenfiles
    NAME = sys.argv[1].strip()
    # Update the target directory using the filename
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

else:
    print("modelExporterONNX takes at least name of PyTorch model! Exiting program")
    exit()

#################
# Preprocessing #
#################

# Get input
print("Observed input")
# The number of input nodes is is model.h[0]
inp = torch.ones(1, model.h[0])
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