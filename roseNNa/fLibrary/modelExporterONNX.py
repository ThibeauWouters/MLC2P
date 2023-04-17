# Import packages
import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
import pathlib
import sys, getopt
import pickle
import FeedForwardNetwork

#############
# CONSTANTS #
#############

# Define global constants for preprocessing
FILEPATH = os.getcwd()
# ^ where the files will get located, by default same folder
FILENAME  = "mynet"
# ^ the name you want to give to the files to recognize it
VERBOSE = True
# ^ whether we print or not during preprocessing
produce = True
# ^ TODO

#######################
# AUXILIARY FUNCTIONS #
#######################

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()

# Two auxiliary functions are needed to load the weights from the state dict into a fresh network
def get_hidden_size_from_state_dict(state_dict):
    """
    Finds the sizes of the two hidden layers of our 2-layer architecture given a state dict.
    :param state_dict: State dict of saved parameters
    :return: h, sizes of hidden layers
    """

    # Return variable is a list
    h = []
    # Iterate over the state dict, look at weight matrices and save their second shape param (input)
    for key in state_dict:
        if key[-6:] == "weight":
            print(state_dict[key].shape)
            shape = state_dict[key].shape
            h.append(shape[-1])

    h.append(shape[0])

    # We discard the first one, as that corresponds to the input
    return h


### TODO get activation function and change it!!!
# def get_activation_func(state_dict):
#
#     return state_dict["activation1"]

def create_nn(state_dict):
    """
    Create a NeuralNetwork object if given a dictionary of the weights, with correct sizes for hidden layers.
    :param state_dict: State dictionary containing the weights of the neural network
    :return:
    """
    h = get_hidden_size_from_state_dict(state_dict)
    print(h)
    activation_func = nn.Sigmoid
    model = FeedForwardNetwork.FeedForwardNetwork(h=h, activation_function=activation_func)
    model.load_state_dict(state_dict)

    return model

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
                print("First argument must be PyTorch model saved to Pickle file")
                print("Options:")
                print("  -h,  --help         Show this help message and exit")
                print("  -v,  --verbose      Enable verbose mode")
                print("  -fn, --filename     Specify output filenames")
            elif arg == "-v" or arg == "--verbose":
                VERBOSE = True
            elif arg == "-fn" or arg == "--filename":
                FILENAME = sys.argv[i + 1]
            elif arg == "-n":
                produce = False

    # Check default first argument: the first argument has to be the model
    MODEL_FILENAME = sys.argv[1].strip()
    if MODEL_FILENAME[-4:] != ".pth":
        print("Model filename must be .pth file. Exiting program.")
        exit()
    # Open the file in binary mode
    state_dict = torch.load(MODEL_FILENAME)
    model = create_nn(state_dict)
        # Call load method to deserialze
        # model = pickle.load(file)
        # if VERBOSE:
        #     print(f"Reading pickled model from {MODEL_FILENAME}")
        #     print(model)

else:
    print("modelExporterONNX takes at least filename of pickled PyTorch model! Exiting program")
    exit()

#################
# Preprocessing #
#################

# Get input
print("Observed input")
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

with open(FILEPATH + FILENAME + ".txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))

###############
# ONNX EXPORT #
###############

# Finally, export the model and weights as ONNX files
torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  FILEPATH + FILENAME + ".onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  FILEPATH + FILENAME + "_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )