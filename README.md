# tobit_wCI
Tobit Model based on StatsModel with Confidence Interval in Python 

---

## Developer:

| Developer  | e-mail |
| ------------- | ------------- |
| Walter Martins Filho  | walterwsmf@outlook.com |

---

## Credits:

I use the tobit likelihood and its seconde derivate from James Jensen, https://github.com/jamesdj/tobit, and just re-write the part for which package will implement the linear regression. In this case, I used the OLS routine from Statsmodel package. 

Also, I add the routines to obtain the confidence interval from the Hessian matrix that came from scipy.optimize, which gave to us the maximum and minimum limits for our model.

---
## Problems:

- tobit does not understand the values to censur when use apply_model from use.py

- it seems to have a problem how I include some dummie function on the problem.

---

### Debug

Some warnings appears when we try to check errors on this code. But, it is only warnings:

1. warning in "tobit.py": **No name 'log_ndtr' in module 'scipy.special'**
    
    - **Source**: https://github.com/PyCQA/pylint/issues/2742