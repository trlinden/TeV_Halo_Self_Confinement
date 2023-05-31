# TeV_Halo_Self_Confinement

There are two codes that are included here, one is a 1D particle transport code, and the 2nd is a spherically symmetric 3D particle transport code. The differences are in the r^2 volume factor of the 3D code, which causes the CR gradient to fall off quicker.

The 3D version includes a contribution from a SNR by default -- the easiest way to disable this contribution is to set the SNR power to be some small and inconsequential number.

There are many options that can be set within the code, both through command line arguments and through switchable options within the code.

The associated paper with this code can be found at: https://arxiv.org/abs/2111.01143.
