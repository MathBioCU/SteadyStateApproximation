## Steady-State Approximation

This software package is distributed in the hope that it will be useful for the asymptotic analysis
of size-structured population models. The theoretical framework for this software has been developed in Mirzaev I., and Bortz D.M., (2016), where we 
developed a numerical framework for computing approximations to stationary solutions of general evolution equations. This software can also be used 
to produce existence and stability regions for steady states of size-structured population models. 

**Two particular applications are given:**

1. Population balance equations (see Mirzaev, I., & Bortz, D. M. (2015). *arXiv:1507.07127* )
2. Sinko-Streifer size-structured population model (see Sinko, J. W., and Streifer, W. (1967), *Ecology, 48(6):910-918.* ) 


## Dependencies
The program is written purely in Python 2.7. It depends on some famous libraries: *SciPy, NumPy, matplotlib, multiprocessing, etc*.
All the dependencies can be solved by installing *Anaconda* software package. Installation instructions can be found at
*https://www.continuum.io/downloads*


## Basic Usage

For example to generate existence and stability regions for the population balance equations:

1. Navigate to the folder, where you have extracted this software package.
2. Change model rates in **pbe_model_rates.py** file as you desire (instructions available in the python file).
3. Run the following command in your terminal
```
 python pbe_exixtence_region.py 
``` 
Generated existence and stability regions can be found in **images** folder. Similarly, run programs **pbe_jacobian_eigenvalue_plots.py** and **pbe_stability_plots.py** 

Note that to simulate Sinko-Streifer model. Model rates should be updated in **sinko_model_rates.py** and the program **sinko_existence_region.py** should be run according to above instructions. 



## Citation
If you use this program to do research that leads to publication, we ask that you acknowledge use of this program by citing the following in your publication::


Cite the paper: 

```
Mirzaev I., Bortz D. M. (2016) A numerical framework for computing steady states of size-structured population models and their stability , arXiv:1602.07033
```

Cite the software package:

```
Mirzaev, I. (2016). Steady state approximation. https://github.com/MathBioCU/SteadyStateApproximation.
```

## More Information

For further information please contact me at mirzaev@colorado.edu


