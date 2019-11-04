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
## License:

