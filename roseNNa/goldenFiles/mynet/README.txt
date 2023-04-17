Hey there! This directory stores our own neural network. The state dict is one for the C2P network for the ideal gas law, i.e. the one trained first for the master thesis. This is already a pruned network that should have a slightly higher performance compared to the first one. The architecture looks like:
```
NeuralNetwork(
  (linear1): Linear(in_features=3, out_features=504, bias=True)
  (linear2): Linear(in_features=504, out_features=127, bias=True)
  (linear3): Linear(in_features=127, out_features=1, bias=True)
)
```
There are sigmoid layers in between. The code for this network is in NNC2P.ipynb, see https://github.com/ThibeauWouters/master-thesis-AI