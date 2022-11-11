# Differential Evolution (DE) and Cross Entropy Method (CEM)
We install Differential Evolution (DE) and Cross Entropy Method (CEM)
Improved version to optimize the following objective functions:
1. f1: Sphere (d = 2 variables and 10 variables).
2. f2: Zakharov (d = 2 variables and 10 variables).
3. f3: Rosenbrock (d = 2 variables and 10 variables).
4. f4: Michalewicz (d = 2 variables and 10 variables).
5. f5: Ackley (d = 2 variables and 10 variables).

Information about these functions: formula of objective function, domain of value (search domain), global minima, you can refer to the following link:
[here](https://www.sfu.ca/~ssurjano/optimization.html)

### You do the following experiment:
- Population size N (or ) = 32, 64, 128, 256, 512, 1024
- For each case (f,d,N), we need to run the experiment 10 times, using different random seeds
- Each time the experiment runs, the algorithm stops immediately after 100,000 times (for d=2)
or 1,000,000 times (for d=10) the fitness evaluation function is called.
You save information about **the best solution $best_i$ and the value of the objective function( $𝑏𝑒𝑠𝑡_{i}$ )**
that DE and CEM find **at each generation i and the number of evaluation function calls
(num_of_evaluations)** used from the beginning of the experiment to the end of the generation this thing
- For each combination (f, d), draw a convergence graph: the vertical axis of the function value.
the target 𝑓𝑖𝑡𝑛𝑒𝑠( $𝑏𝑒𝑠𝑡_i$) saved as above, the horizontal axis is the number of calls to the evaluation function (from 0 to
100,000 or 1,000,000) like the following illustration. Note: just choose 2
case N=128 and 1024 of DE and CEM for easy viewing. The line in the middle is
average value over 10 runs, and error bars are +/- 1 std. Can draw
error bar with fill_between.
![image]

