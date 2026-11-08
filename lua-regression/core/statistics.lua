-- Copyright (c) 2025 George Allison
--
-- This work may be distributed and/or modified under the
-- conditions of the LaTeX Project Public License, either version 1.3
-- of this license or any later version.
-- The latest version of this license is in
--   http://www.latex-project.org/lppl.txt
-- and version 1.3 or later is part of all distributions of LaTeX
-- version 2005/12/01 or later.
--
-- This work has the LPPL maintenance status `maintained'.
--
-- The Current Maintainer of this work is George Allison.
--
-- This work consists of the files lua-regression.sty and lua-regression.lua
--
-- This software is provided 'as is', without warranty of any kind,
-- either expressed or implied, including, but not limited to, the
-- implied warranties of merchantability and fitness for a
-- particular purpose.

-- Special functions and significance tests.
-- Everything here is local except the `luareg_stats` table and the two
-- regression_* entry points used by the pipeline.

local huge = math.huge
local EPS = 1e-15            -- convergence tolerance (double precision)
local FPMIN = 1e-300
local MAX_ITER = 500
local HALF_LOG_TWO_PI = 0.5 * math.log(2 * math.pi) -- 0.91893853320467274178 
local DBL_MAX_LIMIT = 171.6243769563027 -- still safe for exp(log_gamma(x)) to avoid overflow

-- Lanczos approximation (g = 7, n = 9), relative error about 1e-15
-- https://www.mrob.com/pub/ries/lanczos-gamma.html
local lanczos = {
    0.99999999999980993227684700473478,
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,
    -176.61502916214059906584551354,
    12.507343278686904814458936853,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7,
}

local function log_gamma(x)
    assert(x > 0, "Invalid input for log_gamma function: " .. tostring(x))
    if x < 0.5 then
        -- reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        return math.log(math.pi / math.sin(math.pi * x)) - log_gamma(1 - x)
    end
    x = x - 1
    local a = lanczos[1]
    local t = x + 7.5
    for i = 2, 9 do a = a + lanczos[i] / (x + i - 1) end
    return HALF_LOG_TWO_PI + (x + 0.5) * math.log(t) - t + math.log(a)
end

local function gamma(x)
    assert(x > 0, "Invalid input for gamma function: " .. tostring(x))
    if x > DBL_MAX_LIMIT then return huge end
    return math.exp(log_gamma(x))
end

-- Regularised incomplete gamma Q(a, x) = Gamma(a, x) / Gamma(a)
-- https://functions.wolfram.com/GammaBetaErf/GammaRegularized/introductions/Gammas/ShowAll.html
local function gamma_q(a, x)
    if x <= 0 then return 1 end
    local lg = log_gamma(a)
    if x < a + 1 then
        -- series for P(a, x)
        local ap, sum, del = a, 1 / a, 1 / a
        for _ = 1, MAX_ITER do
            ap = ap + 1
            del = del * x / ap
            sum = sum + del
            if math.abs(del) < math.abs(sum) * EPS then break end
        end
        return 1 - sum * math.exp(-x + a * math.log(x) - lg)
    end
    -- continued fraction (modified Lentz) for Q(a, x)
    local b = x + 1 - a
    local c = 1 / FPMIN
    local d = 1 / b
    local h = d
    for i = 1, MAX_ITER do
        local an = -i * (i - a)
        b = b + 2
        d = an * d + b
        if math.abs(d) < FPMIN then d = FPMIN end
        c = b + an / c
        if math.abs(c) < FPMIN then c = FPMIN end
        d = 1 / d
        local del = d * c
        h = h * del
        if math.abs(del - 1) < EPS then break end
    end
    return math.exp(-x + a * math.log(x) - lg) * h
end

-- Complementary error function, erfc(x) = Q(1/2, x^2) for x >= 0
local function erfc(x)
    if x >= 0 then return gamma_q(0.5, x * x) end
    return 2 - gamma_q(0.5, x * x)
end

-- Standard normal CDF, accurate in both tails
local function normal_cdf(z)
    return 0.5 * erfc(-z / math.sqrt(2))
end

-- Inverse standard normal CDF Phi^{-1}(p) for p in (0, 1).
-- Acklam's rational approximation refined with two Halley/Newton steps.
-- https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
local function normal_inv_cdf(p)
    assert(p > 0 and p < 1, "Invalid input for normal_inv_cdf: " .. tostring(p))
    local a1, a2, a3, a4, a5, a6 =
        -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00
    local b1, b2, b3, b4, b5 =
        -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01
    local c1, c2, c3, c4, c5, c6 =
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00
    local d1, d2, d3, d4 =
         7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00
    local p_low, p_high = 0.02425, 1 - 0.02425
    local z
    if p < p_low then
        local q = math.sqrt(-2 * math.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    elseif p <= p_high then
        local q = p - 0.5
        local r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    else
        local q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    end
    -- Refine with Newton steps against normal_cdf
    local inv_sqrt_2pi = 0.39894228040143267794
    for _ = 1, 2 do
        local err = normal_cdf(z) - p
        local pdf = inv_sqrt_2pi * math.exp(-0.5 * z * z)
        if pdf > 0 then z = z - err / pdf end
    end
    return z
end

-- Continued fraction for the incomplete beta function (modified Lentz)
local function betacf(a, b, x)
    local qab, qap, qam = a + b, a + 1, a - 1
    local c = 1
    local d = 1 - qab * x / qap
    if math.abs(d) < FPMIN then d = FPMIN end
    d = 1 / d
    local h = d
    for m = 1, MAX_ITER do
        local m2 = 2 * m
        local aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if math.abs(d) < FPMIN then d = FPMIN end
        c = 1 + aa / c
        if math.abs(c) < FPMIN then c = FPMIN end
        d = 1 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if math.abs(d) < FPMIN then d = FPMIN end
        c = 1 + aa / c
        if math.abs(c) < FPMIN then c = FPMIN end
        d = 1 / d
        local del = d * c
        h = h * del
        if math.abs(del - 1) < EPS then break end
    end
    return h
end

-- Regularised incomplete beta function I_x(a, b)
-- https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainc.html
local function betai(a, b, x)
    if x <= 0 then return 0 end
    if x >= 1 then return 1 end
    local ln_bt = log_gamma(a + b) - log_gamma(a) - log_gamma(b)
                  + a * math.log(x) + b * math.log(1 - x)
    local bt = math.exp(ln_bt)
    if x < (a + 1) / (a + b + 2) then
        return bt * betacf(a, b, x) / a
    end
    return 1 - bt * betacf(b, a, 1 - x) / b
end

-- Two-sided p-value of Student's t statistic with `dof` degrees of freedom
local function t_two_sided_p(t, dof)
    if t ~= t then return nil end
    if t == 0 then return 1 end
    return betai(dof / 2, 0.5, dof / (dof + t * t))
end

-- Upper-tail probability P(F > f) of the F distribution.
-- Evaluated directly (not as 1 - CDF) so tiny p-values keep their precision.
local function f_upper_tail(f, dfn, dfd)
    if f <= 0 then return 1 end
    if f == huge then return 0 end
    return betai(dfd / 2, dfn / 2, dfd / (dfd + dfn * f))
end

-- Upper-tail probability P(Chi^2 > x) with `df` degrees of freedom
local function chi2_upper_tail(x, df)
    if df <= 0 then return nil end
    if x <= 0 then return 1 end
    if x == huge then return 0 end
    return gamma_q(0.5 * df, 0.5 * x)
end

luareg_stats = {
    gamma = gamma,
    log_gamma = log_gamma,
    gamma_q = gamma_q,
    erfc = erfc,
    normal_cdf = normal_cdf,
    normal_inv_cdf = normal_inv_cdf,
    betai = betai,
    t_two_sided_p = t_two_sided_p,
    f_upper_tail = f_upper_tail,
    chi2_upper_tail = chi2_upper_tail,
}

local function horner(coeffs, x)
    local y = 0
    for i = 1, #coeffs do y = y * x + coeffs[i] end
    return y
end

-- Standard errors and two-sided t-test p-values of OLS polynomial coefficients.
-- The covariance comes from the QR fit on scaled x and is mapped back to the
-- original coefficient basis, so it stays well conditioned for large x.
function regression_pvalues(data, coeffs, order)
    local n = #data
    local p = order + 1
    local dof = n - p
    if dof <= 0 then return nil end

    local _, cov = polynomial_regression(data, order, true)
    if not cov then return nil end

    local rss = 0
    for i = 1, n do
        rss = rss + (data[i][2] - horner(coeffs, data[i][1]))^2
    end
    local sigma2 = rss / dof

    local std_errs, p_values = {}, {}
    for i = 1, p do
        local var = sigma2 * cov[i][i]
        std_errs[i] = math.sqrt(math.max(var, 0))
        if std_errs[i] == 0 then
            p_values[i] = (coeffs[i] == 0) and 1 or 0
        else
            p_values[i] = t_two_sided_p(coeffs[i] / std_errs[i], dof)
        end
    end
    return std_errs, p_values
end

-- Overall F-test of regression significance (upper tail)
function regression_overall_pvalue(data, coeffs, order)
    local n = #data
    local df_model = order
    local df_resid = n - (order + 1)
    if df_model < 1 or df_resid <= 0 then return nil end

    local y_mean = 0
    for i = 1, n do y_mean = y_mean + data[i][2] end
    y_mean = y_mean / n

    local tss, rss = 0, 0
    for i = 1, n do
        tss = tss + (data[i][2] - y_mean)^2
        rss = rss + (data[i][2] - horner(coeffs, data[i][1]))^2
    end
    if tss == 0 then return nil end
    if rss == 0 then return 0 end

    local f_stat = math.max(tss - rss, 0) / df_model / (rss / df_resid)
    return f_upper_tail(f_stat, df_model, df_resid)
end

-- Asymptotic normal Wald standard errors and two-sided p-values from an
-- estimated parameter covariance matrix
function wald_normal_pvalues(coeffs, cov)
    if not coeffs or not cov then return nil end
    local std_errs, p_values = {}, {}
    for i = 1, #coeffs do
        local var = cov[i] and cov[i][i] or 0
        local se = math.sqrt(math.max(var, 0))
        std_errs[i] = se
        if se == 0 then
            p_values[i] = (coeffs[i] == 0) and 1 or 0
        else
            local z = math.abs(coeffs[i] / se)
            p_values[i] = erfc(z / math.sqrt(2))
        end
    end
    return std_errs, p_values
end

-- Residual and null deviances for a Poisson GLM
function poisson_deviances(data, predict_func)
    local n = #data
    if n < 2 then return nil, nil end
    local sum_y = 0
    for i = 1, n do sum_y = sum_y + data[i][2] end
    local mean_y = sum_y / n
    if mean_y <= 0 then return nil, nil end

    local function unit_dev(y, mu)
        mu = math.max(mu, 1e-15)
        local term = (y > 0) and (y * math.log(y / mu)) or 0
        return 2 * (term - (y - mu))
    end

    local d_model, d_null = 0, 0
    for i = 1, n do
        local y = data[i][2]
        local mu = predict_func(data[i][1])
        d_model = d_model + unit_dev(y, mu)
        d_null = d_null + unit_dev(y, mean_y)
    end
    return math.max(d_model, 0), math.max(d_null, 0)
end

-- Deviance pseudo-R^2 = 1 - D(model) / D(null) for Poisson regression
function poisson_pseudo_r2(data, predict_func)
    local d_model, d_null = poisson_deviances(data, predict_func)
    if not d_null or d_null == 0 then return nil end
    return 1 - d_model / d_null
end

-- Likelihood-ratio / deviance F-test and coefficient p-values for a Poisson GLM.
-- Uses unit dispersion (phi = 1) when residual deviance <= df_resid (or when
-- fixed_dispersion == 1), and quasi-Poisson dispersion adjustment when overdispersed.
function poisson_inference(data, coeffs, cov_unscaled, predict_func, df_model, fixed_dispersion)
    local n = #data
    local df_resid = n - (df_model + 1)
    if not df_model or df_model < 1 or df_resid <= 0 then return nil, nil end
    local d_model, d_null = poisson_deviances(data, predict_func)
    if not d_null then return nil, nil end

    local pearson_chi2 = 0
    for i = 1, n do
        local mu = math.max(predict_func(data[i][1]), 1e-15)
        pearson_chi2 = pearson_chi2 + (data[i][2] - mu) ^ 2 / mu
    end
    local phi_dev = d_model / df_resid
    local phi_pearson = pearson_chi2 / df_resid

    local use_unit = (fixed_dispersion == 1) or (phi_dev <= 1.0)
    local pvals, overall_p
    if use_unit then
        _, pvals = wald_normal_pvalues(coeffs, cov_unscaled)
        overall_p = chi2_upper_tail(math.max(d_null - d_model, 0), df_model)
    else
        if cov_unscaled then
            pvals = {}
            for i = 1, #coeffs do
                local var = phi_pearson * (cov_unscaled[i] and cov_unscaled[i][i] or 0)
                local se = math.sqrt(math.max(var, 0))
                if se == 0 then
                    pvals[i] = (coeffs[i] == 0) and 1 or 0
                else
                    pvals[i] = t_two_sided_p(coeffs[i] / se, df_resid)
                end
            end
        end
        local f_stat = (math.max(d_null - d_model, 0) / df_model) / phi_dev
        overall_p = f_upper_tail(f_stat, df_model, df_resid)
    end
    return pvals, overall_p
end

function poisson_overall_pvalue(data, predict_func, df_model)
    local _, overall_p = poisson_inference(data, nil, nil, predict_func, df_model, 1)
    return overall_p
end

local function sample_quantile_sorted(sorted_vals, q)
    local n = #sorted_vals
    local pos = 1 + (n - 1) * math.min(math.max(q, 0), 1)
    local lo = math.floor(pos)
    local hi = math.ceil(pos)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])
end

-- Koenker & Machado (1999) R^1(tau) check-loss pseudo-R^2 for quantile regression
function quantile_pseudo_r2(data, predict_func, tau)
    local n = #data
    if n < 2 then return nil end
    tau = tau or 0.5
    local ys = {}
    for i = 1, n do ys[i] = data[i][2] end
    table.sort(ys)
    local q_tau = sample_quantile_sorted(ys, tau)

    local function rho(u)
        return u * (tau - ((u < 0) and 1 or 0))
    end

    local v_model, v_null = 0, 0
    for i = 1, n do
        local y = data[i][2]
        v_model = v_model + rho(y - predict_func(data[i][1]))
        v_null = v_null + rho(y - q_tau)
    end
    if v_null == 0 then return nil end
    return 1 - v_model / v_null, v_model, v_null
end

-- Asymptotic standard errors, Student-t coefficient p-values, and Wald F-test
-- overall p-value for quantile regression (matching statsmodels QuantReg vcov="iid":
-- Hall-Sheather 1988 bandwidth with Epanechnikov kernel density at 0).
function quantile_inference(data, coeffs, cov_x_unscaled, predict_func, order, tau)
    local n = #data
    local df_resid = n - (order + 1)
    if order < 1 or df_resid <= 0 then return nil, nil end
    tau = math.min(math.max(tau or 0.5, 1e-4), 1 - 1e-4)

    local resids, yhat = {}, {}
    local y_sum, yhat_sum = 0, 0
    for i = 1, n do
        local yi = data[i][2]
        local pred = predict_func(data[i][1])
        yhat[i] = pred
        resids[i] = yi - pred
        y_sum = y_sum + yi
        yhat_sum = yhat_sum + pred
    end
    local y_mean = y_sum / n
    local yhat_mean = yhat_sum / n
    local y_var = 0
    for i = 1, n do y_var = y_var + (data[i][2] - y_mean) ^ 2 end
    local std_y = math.sqrt(y_var / n)

    local sorted_e = {}
    for i = 1, n do sorted_e[i] = resids[i] end
    table.sort(sorted_e)
    local iqre = sample_quantile_sorted(sorted_e, 0.75) - sample_quantile_sorted(sorted_e, 0.25)

    -- Hall-Sheather (1988) bandwidth in probability space (alpha = 0.05)
    local z_tau = normal_inv_cdf(tau)
    local phi_z = 0.39894228040143267794 * math.exp(-0.5 * z_tau * z_tau)
    local z_alpha = 1.959963984540054 -- Phi^{-1}(0.975)
    local h_q = (n ^ (-1 / 3)) * (z_alpha ^ (2 / 3)) * ((1.5 * (phi_z ^ 2) / (2 * z_tau * z_tau + 1)) ^ (1 / 3))
    local q_hi = math.min(1 - 1e-6, tau + h_q)
    local q_lo = math.max(1e-6, tau - h_q)
    local scale_e = math.min(std_y, iqre / 1.34)
    local h = scale_e * (normal_inv_cdf(q_hi) - normal_inv_cdf(q_lo))

    -- Epanechnikov kernel density estimate fhat(0)
    local fhat0 = 0
    if h > 0 then
        local k_sum = 0
        for i = 1, n do
            local u = resids[i] / h
            if math.abs(u) <= 1 then
                k_sum = k_sum + 0.75 * (1 - u * u)
            end
        end
        fhat0 = k_sum / (n * h)
    end
    local sparsity = (fhat0 > 0) and (1 / fhat0) or 0
    local omega2 = tau * (1 - tau) * (sparsity ^ 2)

    local pvals
    if cov_x_unscaled then
        pvals = {}
        for i = 1, #coeffs do
            local var = omega2 * (cov_x_unscaled[i] and cov_x_unscaled[i][i] or 0)
            local se = math.sqrt(math.max(var, 0))
            if se == 0 then
                pvals[i] = (coeffs[i] == 0) and 1 or 0
            else
                pvals[i] = t_two_sided_p(coeffs[i] / se, df_resid)
            end
        end
    end

    local ess_pred = 0
    for i = 1, n do ess_pred = ess_pred + (yhat[i] - yhat_mean) ^ 2 end
    local overall_p
    if omega2 <= 0 then
        overall_p = (ess_pred > 0) and 0 or 1
    else
        local f_stat = (ess_pred / order) / omega2
        overall_p = f_upper_tail(f_stat, order, df_resid)
    end
    return pvals, overall_p
end
