import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from tqdm import tqdm

def ReLU(x):
    return x * (x > 0)

def worst_case_risk(mu, eta, alpha, h, losses):
    return ReLU(mu-eta) / (1-alpha) + eta + h * (losses-mu) / (1-alpha)

class ConditionalShiftRisk:
    '''
    Based on Subbaswamy, Adarsh, Roy Adams, and Suchi Saria. "Evaluating model
    robustness and stability to dataset shift." International Conference on Artificial
    Intelligence and Statistics. PMLR, 2021.
    '''
    def __init__(self, alpha, cv, loss_model=None, quantile_model=None):
        self.quantile_model = quantile_model
        self.loss_model = loss_model
        self.alpha = alpha
        self.cv = cv
        if loss_model is None:
            self.loss_model = GradientBoostingRegressor()
        if quantile_model is None:
            self.quantile_model = GradientBoostingRegressor(loss='quantile', alpha=alpha)

    def fit(self, X_val, mutable_columns, immutable_columns, losses):
        _val = X_val[mutable_columns + immutable_columns]
        self.h = np.zeros_like(losses).astype('bool')
        self.eta = np.zeros_like(losses)
        self.mu = np.zeros_like(losses)
        for train, test in tqdm(KFold(n_splits=self.cv).split(_val), total=self.cv):
            self.loss_model.fit(_val.iloc[train], losses.iloc[train])
            mu_train = self.loss_model.predict(_val.iloc[train])
            mu_test = self.loss_model.predict(_val.iloc[test])
            self.mu[test] = mu_test
            self.quantile_model.fit(_val.iloc[train][immutable_columns], mu_train)
            eta_test = self.quantile_model.predict(_val.iloc[test][immutable_columns])
            self.eta[test] = eta_test
            self.h[test] = mu_test > eta_test
        risk = worst_case_risk(self.mu, self.eta, self.alpha, self.h, losses)
        self.risk = risk.mean()
        self.ub_risk = self.risk + 1.96 * risk.std() / np.sqrt(losses.shape[0])
        self.lb_risk = self.risk - 1.96 * risk.std() / np.sqrt(losses.shape[0])

def calculate_worst_case_risk(error, data, immutable, mutable):

    mn_log = []
    lb_log = []
    ub_log = []

    for i in range(9):
        alpha = (i + 1) / 10
        shift_model = ConditionalShiftRisk(
            alpha=alpha,
            cv=5,
            loss_model=LinearRegression(),
            quantile_model=GradientBoostingRegressor(loss='quantile', alpha=alpha),
        )

        losses = pd.Series(error)

        shift_model.fit(
            data,
            mutable,
            immutable,
            losses,
        )

        mn_log.append(shift_model.risk)
        lb_log.append(shift_model.lb_risk)
        ub_log.append(shift_model.ub_risk)
    
    return mn_log, lb_log, ub_log
