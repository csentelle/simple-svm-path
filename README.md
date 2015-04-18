# simplesvmpath
This is a simple implementation of the SVM regularization path following algorithm. Implementation is in MATLAB. 
This method allows use of semi-definite kernels as well as duplicate data points. Details are found in the publication: 

Sentelle, Christopher, Anagnostopoulos, Georgios C., Georgiopoulos, Michael, "A Simple SVM Path Algorithm 
   for Semi-Definite Kernels," IEEE Transactions on Neural Networks and Learning Systems, Submitted. 
   

The regularization path following algorithm explores all solutions to the SVM problem at all values of the parameter, $C$. 
The problem is first reposed to use $\lambda$ instead of $C$ where $\lambda = \frac{1}{C}$. The algorithm starts at a 
large value of $\lambda$ (small value of $C$) where the it is found that the solution essentially does not change with
any larger value (the set of support vectors does not change). From there, the algorithm finds the next value of $\lambda$ 
smaller than the current value where the set of support vectors change. This is referred to as a "breakpoint". At the
breakpoint, the active set (set of support vectors) are changed and the process continues until reaching the pre-specified
minimum value for $\lambda$. In essence, the algorithm finds all of the solutions at all values of $\lambda$ and can do so 
in many cases in not much more time, on average, than it would take to solve for a single value of the regularization parameter.

This works extends the basic premise introduced by Hastie et al. to work efficiently with semi-definite kernels. The method, 
here, also avoids cases where a portion of the regularization path might be skipped (a set of solutions are not identified or 
a range of regularization values are skipped). We also introduce a fully self-contained initialization for finding the 
starting value of $\lambda$ for the case of unequal class sizes. Therefore, the algorithm is entirely self-contained, not 
requiring an external solver, and is designed to work for any practical dataset or kernel type.

Note that this code is currently for demonstration purposes only. That is, while the entire regularization path is solved, 
the algorithm does not currently return the solutions. It is straight-forward, for example, to add code for performing 
cross-validation or evaluate performance on a validation data set at each solution and return the solution yielding the 
best validation performance. If there is sufficient future demand, I plan to provide this type of functionality as well as a 
C++ and/or python implementation.





