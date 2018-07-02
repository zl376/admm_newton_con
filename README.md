# Newton ADMM method with linear inequality constraints

## Note:
* Use **penalty indicator** for inequality constraint
* **Split** (canonical) loss and penalty with equality constraint
* Solve equality constrained problem using **ADMM** (alternating direction method of multipliers)
* Primal sub-problem is solved using **Newton** method (one step)
* Newton inversion can be done using **direct** inversion or **CG (conjugate gradient)**

## Require:
* (Canonical) loss has explicit gradient and hessian.

## An example for logistic regression
![alt text](https://github.com/zl376/host_media/blob/master/formula/10.png "Comparison of un-constrained (left) and positivity-constrained (right) logistic regression")
