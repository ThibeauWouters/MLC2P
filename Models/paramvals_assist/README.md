This is a neural network for the GRHD assist. That is, this network takes three inputs: D, S, tau and generates a single output, mu, which is an approximation of the root that is used within Kastaun's C2P scheme. The model has two layers with each 20 neurons and uses sigmoid activation functions.


The info below is no longer valid.


**NOTE**: We performed a normalization on the training data. We did a minmaxscale (see sklearn MinMaxScaler for background). 
- min values are: [1.76215788e-05 7.86319124e-06 2.95001570e-05]
- scale values are: [14.51535662 64.16326322 61.74312749]

**NOTE**: We also tested this network on samples to estimate the error we get on the estimate, since this has to be used with e.g. Brent's method which uses bracketing for the rootfinding. Based on the $\ell_1$-norm, we get:
- mean error: 0.0008978864974869628
- std error: 0.0019794488050689546

We used a $\mu \pm 3\sigma$ interval for the estimate. Hence, the width of the root is 0.006836232912693827.
.
