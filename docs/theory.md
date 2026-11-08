# Theory and assumptions

## Reduced-unit Lennard-Jones cluster

For a cluster of N identical particles with reduced coordinates (length in sigma), the total potential is

U = 4 sum_{i<j} [(1/r_ij)^12 - (1/r_ij)^6].

Energy is in epsilon, force-gradient components are in epsilon/sigma, and temperature in the Metropolis rule is in epsilon/k_B (the code's reduced temperature is numerically k_B T / epsilon). No dimensional parameters are needed. The radial pair minimum is at r = 2^(1/6) with pair energy -1. The analytic Cartesian gradient for atom i is the sum over j of 24[(1/r_ij)^6 - 2(1/r_ij)^12](r_i-r_j)/r_ij^2.

## Basin hopping

Each trial perturbs every Cartesian coordinate with independent Gaussian noise, recenters the cluster, and locally minimizes with SciPy L-BFGS-B. A proposed local minimum with energy change Delta U is accepted if it is lower, or with probability exp(-Delta U / T) otherwise. The lowest energy found is retained independently of the current state. A NumPy generator seeded by the caller makes the search reproducible.

## Validation target and limitations

This project uses N=13, whose reported global minimum is an icosahedral structure with energy -44.326801 epsilon. The example starts from a seeded random geometry; tests also start from a displaced icosahedral seed to verify the known minimum and deterministic repeatability. A finite run is a heuristic search and cannot prove global optimality for arbitrary initial conditions.

This is a classical pair-potential model. It omits electronic structure, many-body interactions, temperature-dependent thermodynamics, and dimensional parameter mapping. There is no periodic boundary condition or cutoff. Reduced units are used throughout rather than SI units.

## References

- D. J. Wales and J. P. K. Doye, "Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms," *J. Phys. Chem. A* 101, 5111-5116 (1997), https://doi.org/10.1021/jp970984n.
- Cambridge Cluster Database, Lennard-Jones cluster global minima: https://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html.
