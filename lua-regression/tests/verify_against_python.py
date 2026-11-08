#!/usr/bin/env python3
"""
Verification of lua-regression against standard python libraries.
fmodel equations/coefficients, R^2 / pseudo-R^2, overall p-values, and coefficient p-values directly.
"""

import subprocess
import sys
import numpy as np
import pandas as pd
import scipy.special as sp
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def load_mpg():
    df = pd.read_csv("lua-regression/doc/example/seaborn-data/mpg.csv").dropna(
        subset=["horsepower", "weight"]
    )
    return df["horsepower"].to_numpy(dtype=float), df["weight"].to_numpy(dtype=float)


def verify_special_functions():
    lua_code = r"""
package.path = "lua-regression/?.lua;" .. package.path
require("core.statistics")
local S = luareg_stats
print(string.format("%.16g\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g",
    S.log_gamma(12.5),
    S.normal_cdf(1.96),
    S.normal_inv_cdf(0.975),
    S.t_two_sided_p(2.2281388519649385, 10),
    S.f_upper_tail(4.964602743730711, 1, 10),
    S.chi2_upper_tail(3.841458820694124, 1),
    S.betai(2.5, 3.5, 0.4)
))
"""
    out = subprocess.check_output(["lua", "-e", lua_code], text=True).strip()
    vals = [float(v) for v in out.split("\t")]
    refs = [
        float(sp.gammaln(12.5)),
        float(stats.norm.cdf(1.96)),
        float(stats.norm.ppf(0.975)),
        float(2.0 * stats.t.sf(2.2281388519649385, df=10)),
        float(stats.f.sf(4.964602743730711, dfn=1, dfd=10)),
        float(stats.chi2.sf(3.841458820694124, df=1)),
        float(sp.betainc(2.5, 3.5, 0.4)),
    ]
    names = [
        "gammaln(12.5)",
        "norm.cdf(1.96)",
        "norm.ppf(0.975)",
        "2*t.sf(2.2281, df=10)",
        "f.sf(4.9646, 1, 10)",
        "chi2.sf(3.8415, df=1)",
        "betainc(2.5, 3.5, 0.4)",
    ]
    return list(zip(names, vals, refs))


def python_library_models(x: np.ndarray, y: np.ndarray):
    n = len(x)
    X1 = np.column_stack([x, np.ones(n)])
    X2 = np.column_stack([x ** 2, x, np.ones(n)])
    X_lnx = np.column_stack([np.log(x), np.ones(n)])
    results = {}

    # 1. Linear Regression: statsmodels.api.OLS
    ols1 = sm.OLS(y, X1).fit()
    results["Linear"] = {
        "coeffs": [float(c) for c in ols1.params],
        "r2": float(ols1.rsquared),
        "overall_p": float(ols1.f_pvalue),
        "pvals": [float(p) for p in ols1.pvalues],
    }

    # 2. Quadratic Regression: statsmodels.api.OLS
    ols2 = sm.OLS(y, X2).fit()
    results["Quadratic"] = {
        "coeffs": [float(c) for c in ols2.params],
        "r2": float(ols2.rsquared),
        "overall_p": float(ols2.f_pvalue),
        "pvals": [float(p) for p in ols2.pvalues],
    }

    # 3. Power Regression: statsmodels.api.OLS on log-log + sklearn.metrics.r2_score
    ols_pow = sm.OLS(np.log(y), X_lnx).fit()
    scale_pow, exp_pow = float(np.exp(ols_pow.params[1])), float(ols_pow.params[0])
    results["Power"] = {
        "coeffs": [scale_pow, exp_pow],
        "r2": float(r2_score(y, scale_pow * (x ** exp_pow))),
        "overall_p": float(ols_pow.f_pvalue),
        "pvals": [float(p) for p in ols_pow.pvalues],
    }

    # 4. Logarithmic Regression: statsmodels.api.OLS
    ols_log = sm.OLS(y, X_lnx).fit()
    results["Logarithmic"] = {
        "coeffs": [float(c) for c in ols_log.params],
        "r2": float(ols_log.rsquared),
        "overall_p": float(ols_log.f_pvalue),
        "pvals": [float(p) for p in ols_log.pvalues],
    }

    # 5. Ridge Regression: scikit-learn StandardScaler + Ridge + scipy.stats
    scaler = StandardScaler()
    Z = scaler.fit_transform(x.reshape(-1, 1))
    ridge = Ridge(alpha=1.0, solver="svd").fit(Z, y)
    slope_r = float(ridge.coef_[0] / scaler.scale_[0])
    inter_r = float(ridge.intercept_ - slope_r * scaler.mean_[0])
    r2_r = float(ridge.score(Z, y))
    # Effective degrees of freedom and sandwich covariance (Cule et al. 2011) evaluated via scipy.stats
    G = Z.T @ Z
    A = np.linalg.inv(G + 1.0 * np.eye(1))
    M = A @ G
    df_m_r = float(np.trace(M))
    df_r_r = float(n - 1 - np.trace(2 * M - M @ M))
    rss_r = float(np.sum((y - ridge.predict(Z)) ** 2))
    tss_y = float(np.sum((y - np.mean(y)) ** 2))
    sig2_r = rss_r / df_r_r
    var_slope_r = float(sig2_r * (A @ G @ A)[0, 0] / (scaler.scale_[0] ** 2))
    var_inter_r = float(sig2_r / n + (scaler.mean_[0] ** 2) * var_slope_r)
    se_r = [np.sqrt(var_slope_r), np.sqrt(var_inter_r)]
    c_r = [slope_r, inter_r]
    results["Ridge"] = {
        "coeffs": c_r,
        "r2": r2_r,
        "overall_p": float(stats.f.sf(((tss_y - rss_r) / df_m_r) / sig2_r, dfn=df_m_r, dfd=df_r_r)),
        "pvals": [float(2.0 * stats.t.sf(abs(c_r[i]) / se_r[i], df=df_r_r)) for i in range(2)],
    }

    # 6. Quantile Regression: statsmodels.api.QuantReg (vcov="iid")
    qreg = sm.QuantReg(y, X1).fit(q=0.5, vcov="iid")
    q_ftest = qreg.f_test(np.eye(1, 2))
    results["Quantile"] = {
        "coeffs": [float(c) for c in qreg.params],
        "r2": float(qreg.prsquared),
        "overall_p": float(q_ftest.pvalue),
        "pvals": [float(p) for p in qreg.pvalues],
    }

    # 7. Poisson Regression: statsmodels.api.GLM (Poisson family, scale="X2", use_t=True) + scipy.stats.f
    pois = sm.GLM(y, X1, family=sm.families.Poisson()).fit(scale="X2", use_t=True)
    d_null, d_model = float(pois.null_deviance), float(pois.deviance)
    f_pois = ((d_null - d_model) / 1.0) / (d_model / float(pois.df_resid))
    results["Poisson"] = {
        "coeffs": [float(c) for c in pois.params],
        "r2": float(1.0 - d_model / d_null),
        "overall_p": float(stats.f.sf(f_pois, dfn=1, dfd=pois.df_resid)),
        "pvals": [float(p) for p in pois.pvalues],
    }

    # 8. Lasso Regression: scikit-learn StandardScaler + Lasso + statsmodels/scipy.stats
    # sklearn Lasso minimizes (1/(2n))||y - Xw||_2^2 + alpha||w||_1, so alpha = lambda / (2n)
    lasso = Lasso(alpha=1.0 / (2.0 * n), tol=1e-12, max_iter=10000).fit(Z, y)
    slope_l = float(lasso.coef_[0] / scaler.scale_[0])
    inter_l = float(lasso.intercept_ - slope_l * scaler.mean_[0])
    r2_l = float(lasso.score(Z, y))
    rss_l = float(np.sum((y - lasso.predict(Z)) ** 2))
    sig2_l = rss_l / (n - 2)
    se_l = np.sqrt(np.diag(sig2_l * np.linalg.inv(X1.T @ X1)))
    c_l = [slope_l, inter_l]
    results["Lasso"] = {
        "coeffs": c_l,
        "r2": r2_l,
        "overall_p": float(stats.f.sf(((tss_y - rss_l) / 1.0) / sig2_l, dfn=1, dfd=n - 2)),
        "pvals": [float(2.0 * stats.t.sf(abs(c_l[i]) / se_l[i], df=n - 2)) for i in range(2)],
    }

    # 9. Exponential Regression: statsmodels.api.OLS on ln(y) + sklearn.metrics.r2_score
    ols_exp = sm.OLS(np.log(y), X1).fit()
    scale_exp, rate_exp = float(np.exp(ols_exp.params[1])), float(ols_exp.params[0])
    results["Exponential"] = {
        "coeffs": [scale_exp, rate_exp],
        "r2": float(r2_score(y, scale_exp * np.exp(rate_exp * x))),
        "overall_p": float(ols_exp.f_pvalue),
        "pvals": [float(p) for p in ols_exp.pvalues],
    }

    return results


def lua_models():
    lua_code = r"""
package.path = "lua-regression/?.lua;" .. package.path
tex = { sprint = function() end }
texio = { write = function() end }
require("core.data")
require("core.poly")
require("core.statistics")
require("models.regression")
require("core.pipeline")

local data = read_csv_to_table("lua-regression/doc/example/seaborn-data/mpg.csv", 4, 5, {ignore_nan=true, ignore_missing=true})
local cases = {
    {name="Linear",      opts={type="polynomial", order=1}},
    {name="Quadratic",   opts={type="polynomial", order=2}},
    {name="Power",       opts={type="power",      order=1}},
    {name="Logarithmic", opts={type="log",        order=1}},
    {name="Ridge",       opts={type="ridge",      order=1, lambda=1.0}},
    {name="Quantile",    opts={type="quantile",   order=1, quantile=0.5}},
    {name="Poisson",     opts={type="poisson",    order=1}},
    {name="Lasso",       opts={type="lasso",      order=1, lambda=1.0}},
    {name="Exponential", opts={type="exp",        order=1}},
}
for _, c in ipairs(cases) do
    local m = fit_model(data, c.opts)
    local pvals, overall_p = m.pvalues, m.overall_p
    if c.opts.type == "polynomial" and not pvals then
        _, pvals = regression_pvalues(data, m.coeffs, c.opts.order)
        overall_p = regression_overall_pvalue(data, m.coeffs, c.opts.order)
    end
    local r2 = m.r_squared
    if r2 == nil then r2 = calculate_r_squared(data, m.predict) end
    local c_str, p_str = {}, {}
    for _, v in ipairs(m.coeffs) do c_str[#c_str+1] = string.format("%.15g", v) end
    for _, v in ipairs(pvals or {}) do p_str[#p_str+1] = string.format("%.15g", v) end
    print(string.format("%s\t%s\t%.15g\t%.15g\t%s\t%s",
        c.name, table.concat(c_str, ","), r2, overall_p or -1, table.concat(p_str, ","), m.display_equation()))
end
"""
    out = subprocess.check_output(["lua", "-e", lua_code], text=True)
    res = {}
    for line in out.strip().splitlines():
        name, coeffs_s, r2_s, ov_p_s, pvals_s, eq_s = line.split("\t")
        res[name] = {
            "coeffs": [float(v) for v in coeffs_s.split(",")],
            "r2": float(r2_s),
            "overall_p": float(ov_p_s),
            "pvals": [float(v) for v in pvals_s.split(",") if v],
            "eq": eq_s,
        }
    return res


def rel_err(a: float, b: float) -> float:
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def main():
    failures = 0
    total = 0

    def check(label: str, got: float, want: float, tol: float = 1e-5):
        nonlocal failures, total
        total += 1
        err = rel_err(got, want)
        if err > tol:
            failures += 1
            print(f"FAIL {label:38s} | Lua={got:.8e} | Py={want:.8e} | rel_err={err:.2e}")
        else:
            print(f"OK   {label:38s} | Lua={got:.8e} | Py={want:.8e} | rel_err={err:.2e}")

    print("=== Special Functions vs scipy.special / scipy.stats ===")
    for name, got, want in verify_special_functions():
        check(name, got, want, 1e-10)

    x, y = load_mpg()
    py_res = python_library_models(x, y)
    lu_res = lua_models()

    for name in [
        "Linear",
        "Quadratic",
        "Power",
        "Logarithmic",
        "Ridge",
        "Quantile",
        "Poisson",
        "Lasso",
        "Exponential",
    ]:
        p = py_res[name]
        l = lu_res[name]
        print(f"\n--- {name} (LaTeX eq: y = {l['eq']}) ---")
        tol = 5e-2 if name == "Quantile" else 1e-5
        for idx, (lc, pc) in enumerate(zip(l["coeffs"], p["coeffs"]), 1):
            check(f"{name} coeff[{idx}]", lc, pc, 1e-5)
        check(f"{name} R^2", l["r2"], p["r2"], 1e-5)
        check(f"{name} overall_p", l["overall_p"], p["overall_p"], tol)
        for idx, (lp, pp) in enumerate(zip(l["pvals"], p["pvals"]), 1):
            check(f"{name} pval[{idx}]", lp, pp, tol)

    print(f"\nSummary: {total - failures}/{total} checks matched against scipy/statsmodels/sklearn.")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
