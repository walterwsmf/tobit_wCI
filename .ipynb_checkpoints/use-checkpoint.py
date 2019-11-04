import warnings
with warnings.catch_warnings():
    #warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action = "ignore", category = RuntimeWarning)
    warnings.simplefilter(action = "ignore", category = FutureWarning)
    #system
    import os
    import glob
    import itertools
    #data
    import numpy as np
    import pandas as pd
    #OLS STATSMODELS
    import statsmodels.regression.linear_model as sm
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    #Tobit Model
    from tobit import TobitModel 

def permut2lists(a,b):
    """
    Permuta os elementos de duas listas, 'a' e 'b', criando uma lista das listas de combinacoes possiveis.
    """
    solucao = []
    for i,j in  itertools.product(a,b):
        solucao.append([i,j])
    return solucao

def permut_from_dict(var_dict):
    """
    Permuta todos os elementos que tiverem nas chaves de um dicionario dado de entrada
    """
    elements_toPermut = [tuple(var_dict[l]) for l in var_dict.keys()]
    solucao = []
    for i in itertools.product(*elements_toPermut):
        solucao.append(i)
    return solucao

def confidence_interval(X_fit,res,N,alpha = 1.96, show=False):
    """
    X: independent variables pandas.DataFrame
    res: object, output from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    
    Solution found from https://stackoverflow.com/questions/43593592/errors-to-fit-parameters-of-scipy-optimize
    """
    ftol = 1.0e-09
    lower_coef, upper_coef = [],[]
    
    for i in range(res.x.shape[0]-1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            uncertainty_i = np.sqrt(max(1, abs(res.fun))*ftol*res.hess_inv[i])
        if show:
            print(res.x[i], uncertainty_i[i])
        lower_coef.append(res.x[i] - (alpha/np.sqrt(N)) * uncertainty_i[i])#/np.sqrt(data.shape[0]))
        upper_coef.append(res.x[i] + (alpha/np.sqrt(N)) * uncertainty_i[i])#/np.sqrt(data.shape[0]))
    
    lower_limit = np.dot(X_fit,lower_coef)
    upper_limit = np.dot(X_fit,upper_coef)
    
    return (lower_coef, upper_coef), (lower_limit, upper_limit), (res.x,uncertainty_i)

class apply_models(object):
    
    def __init__(self,
                data,
                column_y,
                numeric1_min,
                numeric1_max,
                cols_numeric=["numeric1"],
                cols_dummies=["dummie1"],
                col_censura = 'censura',
                limit_inf=1000,show=False):
        
        self.data = data
        self.column_y = column_y
        self.numeric1_min = numeric1_min
        self.numeric1_max = numeric1_max
        self.cols_numeric = cols_numeric
        self.cols_dummies = cols_dummies
        self.censura = col_censura
        self.limit_inferior = limit_inf
        self.show=show
        
    def create_X_variables(self,):
        X = self.data[self.cols_numeric+self.cols_dummies]
        dummies = pd.get_dummies(X[self.cols_dummies])
        self.X = pd.concat([X[self.cols_numeric],dummies],axis=1)
        self.X = self.X.assign(cte=0.01+np.zeros(X.shape[0]))
        
    def create_X_to_predict(self,):
        X = self.data[self.cols_numeric + self.cols_dummies]
        
        #salvando a informacao das variaveis dummies e numericas
        if not hasattr(self, 'numeric'):
            self.numeric = X[self.cols_numeric].drop_duplicates().sort_values(by=self.cols_numeric)
        if not hasattr(self, 'dummies'):
            self.dummies = X[self.cols_dummies].drop_duplicates().sort_values(by=self.cols_dummies)

        self.quant_per_dummies = [len(X[i].unique()) for i in self.cols_dummies]
        self.amostra_final = int(self.numeric1_max+1.-self.numeric1_min)*int(np.product(self.quant_per_dummies))
        if self.show:
            print('Sample = ',self.amostra_final)

        numeric = pd.concat([pd.Series(np.arange(self.numeric1_min,self.numeric1_max+1.,1))],ignore_index=True)
        numeric = numeric.apply(lambda x: int(x))
        numeric = list(numeric.values)
    
        #criando dicionario de variaveis unicas
        X2fit = {}        
        X2fit[self.cols_numeric[0]] = numeric
#         for i in cols_numeric:
#             X2fit[i] = list(X[i].unique())
        for i in self.cols_dummies:
            X2fit[i] = list(X[i].unique())
        #salvando a informacao de dicionario no objeto
        self.dict_valores_permutacao = X2fit
        
        if self.show:
            print('Dict. Possible Solutions = ',X2fit)
        
        #criando permutacoes
        self.X2predict = permut_from_dict(X2fit)
        self.X2predict = pd.DataFrame(self.X2predict)
        self.X2predict.columns = self.cols_numeric+self.cols_dummies
        
        self.X2_apply_model = pd.get_dummies(self.X2predict, columns=self.cols_dummies)
        self.X2_apply_model = self.X2_apply_model.assign(cte=0.01+np.zeros(self.X2_apply_model.shape[0]))
    
        #return X2predict, X2_apply_model
        
        
    def apply_tobit(self):
        if not hasattr(self, 'X'):
            self.create_X_variables()
            
        model = TobitModel()
        
        y_fit = np.log(self.data[self.column_y])
        
        #garantir que sejam floats
        self.data[self.column_y] = pd.to_numeric(self.data[self.column_y],errors='coerce')
        self.data[self.censura] = pd.to_numeric(self.data[self.censura],errors='coerce')
        self.limit_inferior = float(self.limit_inferior)

        censura_sup = self.data[self.column_y] > self.data[self.censura]
        censura_inf = self.data[self.column_y] < self.limit_inferior
        
        y_cens = np.zeros(self.data.shape[0])
        y_cens[censura_sup] =  1
        y_cens[censura_inf] = -1
        self.data['censura'] = y_cens
        
        result = model.fit( x=self.X.astype(float), y=y_fit, cens=self.data['censura'], verbose=False)
        
        self.fit_tobit_predictions = result.predict(self.X.astype(float))

        return result
    
    def apply_OLS(self,):
        
        y_fit = np.log(self.data[self.column_y])
        
        if not hasattr(self, 'X'):
            self.create_X_variables()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = sm.OLS(y_fit, self.X.astype(float)).fit()
            predictions = model.predict(self.X.astype(float))
            print_model = model.summary()
            prstd, iv_l, iv_u = wls_prediction_std(model)
            
            # salvando informacoes do OLS
            self.fit_ols_predictions = predictions
            self.fit_ols_stats = print_model
            self.fit_ols_iv_l = iv_l
            self.fit_ols_iv_u = iv_u
            self.fit_ols_stats = prstd

        return model
    
    def predict_values(self, mode='ols', show=False):
        
        if mode == 'ols':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                modelo = self.apply_OLS()
            
        elif mode =='tobit':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                modelo = self.apply_tobit()
        
        if not hasattr(self, 'X2_apply_model'):
            self.create_X_to_predict()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            solucao = modelo.predict(self.X2_apply_model)
        
        if mode == 'ols':
            self.solucao_ols = pd.Series(solucao)
            prstd, iv_l, iv_u = wls_prediction_std(modelo,exog=self.X2_apply_model)
            del prstd
            self.solucao_ols_iv_l = pd.Series(iv_l)
            self.solucao_ols_iv_u = pd.Series(iv_u)
            self.resultado_ols = pd.concat([self.X2predict,self.solucao_ols_iv_l,self.solucao_ols,self.solucao_ols_iv_u],axis=1)
            self.resultado_ols.columns = self.cols_numeric+self.cols_dummies+['ref_ols_min', 'ref_ols', 'ref_ols_max']
            self.resultado_ols["ref_ols_min"] = self.resultado_ols["ref_ols_min"].apply(lambda x: np.exp(x))
            self.resultado_ols["ref_ols"] = self.resultado_ols["ref_ols"].apply(lambda x: np.exp(x))
            self.resultado_ols["ref_ols_max"] = self.resultado_ols["ref_ols_max"].apply(lambda x: np.exp(x))
            self.resultado_ols["amplitude_ols"] = (self.resultado_ols["ref_ols_max"] - self.resultado_ols["ref_ols_min"])/self.resultado_ols["ref_ols_min"]

        elif mode == 'tobit':
            self.solucao_tobit = pd.Series(solucao)
            coef_limits, limits, coef_uncertains = confidence_interval(self.X2_apply_model,modelo.result,self.data.shape[0],alpha = 2.5, show=self.show)
            self.solucao_tobit_coef_l = pd.Series(coef_limits[0])
            self.solucao_tobit_coef_u = pd.Series(coef_limits[1])
            self.solucao_tobit_iv_l = pd.Series(limits[0])
            self.solucao_tobit_iv_u = pd.Series(limits[1])
            self.fit_tobit_coef = pd.Series(coef_uncertains[0])
            self.fit_tobit_coef_err = pd.Series(coef_uncertains[1])
            self.resultado_tobit = pd.concat([self.X2predict,self.solucao_tobit_iv_l,self.solucao_tobit,self.solucao_tobit_iv_u],axis=1)
            self.resultado_tobit.columns = self.cols_numeric+self.cols_dummies+['ref_tobit_min', 'ref_tobit', 'ref_tobit_max']
            self.resultado_tobit["ref_tobit_min"] = self.resultado_tobit["ref_tobit_min"].apply(lambda x: np.exp(x))
            self.resultado_tobit["ref_tobit"] = self.resultado_tobit["ref_tobit"].apply(lambda x: np.exp(x))
            self.resultado_tobit["ref_tobit_max"] = self.resultado_tobit["ref_tobit_max"].apply(lambda x: np.exp(x))
            self.resultado_tobit["amplitude_tobit"] = (self.resultado_tobit["ref_tobit_max"] - self.resultado_tobit["ref_tobit_min"])/self.resultado_tobit["ref_tobit_min"]