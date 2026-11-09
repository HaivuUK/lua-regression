#!/usr/bin/env python3
"""
Verification of lua-regression against standard python libraries.
fmodel equations/coefficients, R^2 / pseudo-R^2, overall p-values, and coefficient p-values directly.
"""

import glob
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import scipy.special as sp
import scipy.stats as st
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in (os.getcwd(), os.path.join(os.path.dirname(__file__), "..", ".."))
            if os.path.isfile(os.path.join(p, "lua-regression", "core", "data.lua")))
ROOT = os.path.abspath(ROOT)
DATA = os.path.join(ROOT, "lua-regression", "doc", "example", "seaborn-data")

# Sensible (x, y) pairs; any other CSV falls back to its first two numeric columns.
PAIRS = {
    "anagrams": ("num1", "num2"), "anscombe": ("x", "y"), "attention": ("solutions", "score"),
    "car_crashes": ("speeding", "total"), "dots": ("time", "firing_rate"), "exercise": ("id", "pulse"), 
    "flights": ("year", "passengers"), "fmri": ("timepoint", "signal"), "geyser": ("duration", "waiting"),
    "glue": ("Year", "Score"), "healthexp": ("Spending_USD", "Life_Expectancy"),
    "iris": ("petal_length", "petal_width"), "mpg": ("horsepower", "weight"), 
    "penguins": ("flipper_length_mm", "body_mass_g"), "planets": ("orbital_period", "mass"), 
    "taxis": ("distance", "fare"), "titanic": ("age", "fare"),
}

# name -> (Lua options, applicable(x, y), tolerance)
MODELS = {
    "Linear":      ('type="polynomial", order=1', lambda x, y: True, 1e-4),
    "Quadratic":   ('type="polynomial", order=2', lambda x, y: True, 1e-4),
    "Logarithmic": ('type="log"', lambda x, y: (x > 0).all(), 1e-4),
    "Power":       ('type="power"', lambda x, y: (x > 0).all() and (y > 0).all(), 1e-4),
    "Exponential": ('type="exp"', lambda x, y: (y > 0).all(), 1e-4),
    "Ridge":       ('type="ridge", lambda=1.0', lambda x, y: True, 1e-4),
    "Lasso":       ('type="lasso", lambda=1.0', lambda x, y: True, 1e-3),
    "Quantile":    ('type="quantile", quantile=0.5', lambda x, y: True, 5e-2),
    "Poisson":     ('type="poisson"', lambda x, y: (y >= 0).all() and y.sum() > 0, 5e-2),
}

LUA_HEAD = f"""
package.path = "{ROOT}/lua-regression/?.lua;" .. package.path
tex = {{sprint = function() end}}
texio = {{write = function() end}}
for _, m in ipairs{{"data", "poly", "statistics"}} do require("core." .. m) end
require("models.regression")
"""

def run_lua(body):
    return subprocess.check_output(["lua", "-e", LUA_HEAD + body], text=True).strip()

def rel_err(a, b):
    """Relative error, or absolute error when both values are ~0."""
    if a == b:
        return 0.0
    if abs(a) < 1e-4 and abs(b) < 1e-4:
        return abs(a - b)
    return abs(a - b) / max(abs(a), abs(b))

# --- special functions -------------------------------------------------------
SPECIAL = [  # (label, Lua expression, scipy reference)
    ("gammaln(12.5)", "S.log_gamma(12.5)", sp.gammaln(12.5)),
    ("norm.cdf(1.96)", "S.normal_cdf(1.96)", st.norm.cdf(1.96)),
    ("norm.ppf(0.975)", "S.normal_inv_cdf(0.975)", st.norm.ppf(0.975)),
    ("2*t.sf(2.2281, 10)", "S.t_two_sided_p(2.2281388519649385, 10)", 2 * st.t.sf(2.2281388519649385, 10)),
    ("f.sf(4.9646, 1, 10)", "S.f_upper_tail(4.964602743730711, 1, 10)", st.f.sf(4.964602743730711, 1, 10)),
    ("chi2.sf(3.8415, 1)", "S.chi2_upper_tail(3.841458820694124, 1)", st.chi2.sf(3.841458820694124, 1)),
    ("betainc(2.5, 3.5, 0.4)", "S.betai(2.5, 3.5, 0.4)", sp.betainc(2.5, 3.5, 0.4)),
]

def special_checks():
    exprs = ", ".join(e for _, e, _ in SPECIAL)
    out = run_lua(f'local S = luareg_stats; print(string.format("{"%.16g " * len(SPECIAL)}", {exprs}))')
    for (label, _, ref), got in zip(SPECIAL, out.split()):
        yield label, float(got), float(ref), 1e-10

# --- model fitting -----------------------------------------------------------
def fit_lua(path, xc, yc, names):
    cases = ", ".join(f'{{"{n}", {{{MODELS[n][0]}}}}}' for n in names)
    out = run_lua(f"""
local d = read_csv_to_table("{path}", "{xc}", "{yc}", {{ignore_nan=true, ignore_missing=true}})
local function join(t) local s = {{}} for i, v in ipairs(t or {{}}) do s[i] = string.format("%.15g", v) end return table.concat(s, ",") end
for _, c in ipairs{{{cases}}} do
    local m = fit_model(d, c[2])
    local p, op = m.pvalues, m.overall_p
    if not p then _, p = regression_pvalues(d, m.coeffs, c[2].order); op = regression_overall_pvalue(d, m.coeffs, c[2].order) end
    print(c[1], join(m.coeffs), string.format("%.15g", m.r_squared or calculate_r_squared(d, m.predict)), op or -1, join(p))
end""")
    res = {}
    for line in out.splitlines():
        n, c, r2, op, p = line.split("\t")
        res[n] = dict(coeffs=[float(v) for v in c.split(",")], r2=float(r2), overall_p=float(op),
                      pvals=[float(v) for v in p.split(",") if v])
    return res

def summarize(coeffs, r2, f_p, pvals):
    return dict(coeffs=[float(c) for c in coeffs], r2=float(r2), overall_p=float(f_p),
                pvals=[float(p) for p in pvals])

def shrunk(model, x, y, scaler, Z, df_m, df_r, cov_slope):
    """Shared F / t inference for the penalised (ridge, lasso) fits on raw-scale coefficients."""
    n = len(x)
    slope = model.coef_[0] / scaler.scale_[0]
    coeffs = [slope, model.intercept_ - slope * scaler.mean_[0]]
    rss = np.sum((y - model.predict(Z)) ** 2)
    sig2 = rss / df_r
    var_slope = sig2 * cov_slope / scaler.scale_[0] ** 2
    se = np.sqrt([var_slope, sig2 / n + scaler.mean_[0] ** 2 * var_slope])
    f_p = st.f.sf((np.sum((y - y.mean()) ** 2) - rss) / df_m / sig2, df_m, df_r)
    return summarize(coeffs, model.score(Z, y), f_p, 2 * st.t.sf(np.abs(coeffs) / se, df_r))

def fit_python(name, x, y):
    n = len(x)
    X1, X2, Xl = np.c_[x, np.ones(n)], np.c_[x ** 2, x, np.ones(n)], None
    if (x > 0).all():
        Xl = np.c_[np.log(x), np.ones(n)]

    if name in ("Linear", "Quadratic", "Logarithmic"):
        m = sm.OLS(y, {"Linear": X1, "Quadratic": X2, "Logarithmic": Xl}[name]).fit()
        return summarize(m.params, m.rsquared, m.f_pvalue, m.pvalues)
    if name in ("Power", "Exponential"):  # OLS on ln(y); R^2 on the original scale
        m = sm.OLS(np.log(y), Xl if name == "Power" else X1).fit()
        a, b = np.exp(m.params[1]), m.params[0]
        return summarize([a, b], r2_score(y, np.exp(m.fittedvalues)), m.f_pvalue, m.pvalues)
    if name in ("Ridge", "Lasso"):
        scaler = StandardScaler()
        Z = scaler.fit_transform(x.reshape(-1, 1))
        if name == "Lasso":  # sklearn alpha = lambda / (2n)
            return shrunk(Lasso(alpha=1 / (2 * n), tol=1e-12, max_iter=10000).fit(Z, y), x, y, scaler, Z,
                          1, n - 2, np.linalg.inv(X1.T @ X1)[0, 0] * scaler.scale_[0] ** 2)
        # Ridge: effective degrees of freedom and sandwich covariance (Cule & De Iorio 2012)
        G = Z.T @ Z
        A = np.linalg.inv(G + np.eye(1))
        M = A @ G
        return shrunk(Ridge(alpha=1.0, solver="svd").fit(Z, y), x, y, scaler, Z,
                      np.trace(M), n - 1 - np.trace(2 * M - M @ M), (A @ G @ A)[0, 0])
    if name == "Quantile":
        m = sm.QuantReg(y, X1).fit(q=0.5, vcov="iid")
        return summarize(m.params, m.prsquared, m.f_test(np.eye(1, 2)).pvalue, m.pvalues)
    if name == "Poisson":
        fam = sm.families.Poisson()
        m = sm.GLM(y, X1, family=fam).fit()  # unit dispersion: Wald z-tests, chi-square LR test
        dev_drop, r2 = m.null_deviance - m.deviance, 1 - m.deviance / m.null_deviance
        if m.deviance / m.df_resid <= 1:
            return summarize(m.params, r2, st.chi2.sf(dev_drop, 1), m.pvalues)
        m = sm.GLM(y, X1, family=fam).fit(scale="X2", use_t=True)  # overdispersed: quasi-Poisson
        f = dev_drop / (m.deviance / m.df_resid)
        return summarize(m.params, r2, st.f.sf(f, 1, m.df_resid), m.pvalues)

# --- driver ------------------------------------------------------------------
def dataset_columns(path):
    stem = os.path.basename(path)[:-4]
    if stem == "brain_networks" or stem == "diamonds" or stem == "tips":  # multi-row header or " " surrounded names, not a flat table
        return None
    if stem in PAIRS:
        return PAIRS[stem]
    df = pd.read_csv(path).apply(pd.to_numeric, errors="coerce")
    num = [c for c in df if df[c].notna().sum() >= 5 and df[c].nunique() > 2]
    return tuple(num[:2]) if len(num) >= 2 else None

def main():
    total = failed = 0

    def check(label, got, want, tol):
        nonlocal total, failed
        total += 1
        err = rel_err(got, want)
        failed += err > tol
        print(f"  {'FAIL' if err > tol else 'OK  '} {label:40s} Lua={got:.8e} Py={want:.8e} err={err:.1e}")

    print("== Special functions vs scipy ==")
    for item in special_checks():
        check(*item)

    tested = 0
    for path in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        name = os.path.basename(path)
        cols = dataset_columns(path)
        if not cols:
            print(f"\n-- {name}: skipped (no two numeric columns)")
            continue
        df = pd.read_csv(path)[list(cols)].apply(pd.to_numeric, errors="coerce").dropna()
        x, y = df[cols[0]].to_numpy(float), df[cols[1]].to_numpy(float)
        if len(x) < 5:
            continue
        names = [m for m, (_, ok, _) in MODELS.items() if ok(x, y)]
        lua = fit_lua(path, *cols, names)
        tested += 1
        print(f"\n== {name} (x={cols[0]}, y={cols[1]}, n={len(x)}) ==")
        for m in names:
            tol, L, P = MODELS[m][2], lua[m], fit_python(m, x, y)
            rows = [(f"c{i}", a, b) for i, (a, b) in enumerate(zip(L["coeffs"], P["coeffs"]), 1)]
            rows += [("R^2", L["r2"], P["r2"])]
            rows += [("overall_p", L["overall_p"], P["overall_p"])] * (L["overall_p"] >= 0)
            rows += [(f"p{i}", a, b) for i, (a, b) in enumerate(zip(L["pvals"], P["pvals"]), 1)]
            for label, a, b in rows:
                check(f"{name[:-4]} {m} {label}", a, b, tol)

    print(f"\n{total - failed}/{total} checks passed across {tested} datasets.")
    sys.exit(failed > 0)

if __name__ == "__main__":
    main()
