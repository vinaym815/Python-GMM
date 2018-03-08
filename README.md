# Python-GMM

This code can be utilized N gaussians to D dimensional data. It 
was inspired by video of GMM by siraj available
on https://www.youtube.com/watch?v=JNlEIEwe-Cg&t=603s

The algorithm does not include computation of log likelihood
and uses number of iteration as a termination criterion

The algorithm also doesn't include a safe check to prevent the
collase of a gaussian to a single point.
