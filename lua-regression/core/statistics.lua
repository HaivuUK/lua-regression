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

-- Distribution funcctions and statistial significance calculations
function gamma(x)
    assert(x > 0, "Invalid input for gamma function: " .. x)
    
    -- Split function domain into three intervals:
    -- (0, 0.001), (0.001, 12), and (12, inf)

    -- First interval (0, 0.001)
    -- For small x we can use the approximation: 1/gamma(x) = x + gamma x^2, where gamma is the Euler-Mascheroni constant

    local gamma = 0.57721566490153286060651209008240243104215933593992 -- Euler-Mascheroni constant

    if x < 0.001 then
        return 1.0 / (x * (1.0 + gamma * x))
    end

    -- Second interval (0.001, 12)
    if x < 12.0 then
        -- approximate gamma over (1,2) and use reduction identities to reduce other cases to this interval

        y = x
        n = 0
        arg_was_less_than_one = (y < 1.0)

        -- add or subtract integers as necessary to bring y into (1,2)
        if arg_was_less_than_one then
            y = y + 1.0
        else
            n = math.floor(y) - 1  -- will use n later
            y = y - n
    end

    -- numerator coefficients for approximation over the interval (1,2)
    p = {
        -1.71618513886549492533811E+0,
         2.47656508055759199108314E+1,
        -3.79804256470945635097577E+2,
         6.29331155312818442661052E+2,
         8.66966202790413211295064E+2,
        -3.14512729688483675254357E+4,
        -3.61444134186911729807069E+4,
         6.64561438202405440627855E+4
    }

    -- denominator coefficients for approximation over interval
    q = {
        -3.08402300119738975254353E+1,
         3.15350626979604161529144E+2,
        -1.01515636749021914166146E+3,
        -3.10777167157231109440444E+3,
         2.25381184209801510330112E+4,
         4.75584627752788110767815E+3,
        -1.34659959864969306392456E+5,
        -1.15132259675553483497211E+5
    }

    num = 0.0
    den = 1.0

    z = y - 1
    for i = 1, 8 do
        num = (num + p[i]) * z
        den = den * z + q[i]
    end
    result = num / den + 1.0

    -- apply correction if argument was not initially in (1,2)
    if arg_was_less_than_one then
        result = result / (y - 1.0)
    else
        for i = 1, n do
            result = result * y
            y = y + 1.0
        end
    end
    return result
end

-- third interval (12, inf)
    if x > 171.624 then
        return 1.0/0.0  -- will overflow
    end

    return math.exp(log_gamma(x))
end

function log_gamma(x)
    assert(x > 0, "Invalid input for log_gamma function: " .. x)
    
    if x < 12.0 then
        return math.log(math.abs(gamma(x)))
    end

    -- Abramowitz and Stegun 6.1.41
    -- Asymptotic series should be good to at least 11 or 12 figures
    -- For error analysis see Whittiker and Watson, modern analaysis page 252

    c = {
        1.0/12.0,
        -1.0/360.0,
        1.0/1260.0,
        -1.0/1680.0,
        1.0/1188.0,
        -691.0/360360.0,
        1.0/156.0,
        -3617.0/122400.0
    }
    z = 1.0/(x*x)
    sum = c[8]
    for i = 7, 1, -1 do
        sum = sum * z + c[i]
    end
    series = sum / x

    halfLogTwoPi = 0.91893853320467274178032973640562
    logGamma = (x - 0.5)*math.log(x) - x + halfLogTwoPi + series
    return logGamma
end

-- Cumulative distribution function for the standard normal distribution
local function normal_cdf(z)
    -- Abramowitz and Stegun approximation for the error function
    local t = 1 / (1 + 0.2316419 * math.abs(z))
    local d = 0.3989422804014337 * math.exp(-z * z / 2)
    local prob = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    if z > 0 then
        return 1 - prob
    else
        return prob
    end
end

-- Beta function for F-distribution CDF approximation
local function betacf(a, b, x)
    local MAX_ITER, EPS, FPMIN = 100, 3.0e-7, 1.0e-30
    local qab, qap, qam = a + b, a + 1.0, a - 1.0
    local c, d, h = 1.0, 1.0 - qab * x / qap, 1.0 - qab * x / qap
    if math.abs(d) < FPMIN then d = FPMIN end
    d = 1.0 / d
    h = d
    for m = 1, MAX_ITER do
        local m2 = 2 * m
        local aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if math.abs(d) < FPMIN then d = FPMIN end
        c = 1.0 + aa / c
        if math.abs(c) < FPMIN then c = FPMIN end
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if math.abs(d) < FPMIN then d = FPMIN end
        c = 1.0 + aa / c
        if math.abs(c) < FPMIN then c = FPMIN end
        d = 1.0 / d
        local del = d * c
        h = h * del
        if math.abs(del - 1.0) < EPS then break end
    end
    return h
end

local function betai(a, b, x)
    if x < 0 or x > 1 then return 0 end
    if x == 0 then return 0 end
    if x == 1 then return 1 end

    -- user serries expansion for small x
    if x < 1e-8 then
        -- first term expansion
        return math.exp(a * math.log(x) - math.log(a) + log_gamma(a + b) - log_gamma(a) - log_gamma(b))
    end

    -- series expansion for small 1-x
    if 1 - x < 1e-8 then
        return 1 - math.exp(b * math.log(1 - x) - math.log(b) + log_gamma(a + b) - log_gamma(a) - log_gamma(b))
    end

    -- else continued fraction
    local log_bt = log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * math.log(x) + b * (math.log(1 - x))
    if x < (a + 1) / (a + b + 2) then
        local frac = betacf(a, b, x) / a
        if frac == 0 then return 0 end
        local log_val = log_bt + math.log(frac)
        if log_val < -700 then
            return 0
        else
            return math.exp(log_val)
        end
    else
        local frac = betacf(b, a, 1 - x) / b
        if frac == 0 then return 1 end
        local log_val = log_bt + math.log(frac)
        if log_val < -700 then
            return 1
        end
        local val = math.exp(log_val)
        if val < 1e-15 then
            return 1 - 1e-15
        else
            return 1 - val
        end
    end
end

local function f_cdf(f, dfn, dfd)
    return betai(0.5 * dfn, 0.5 * dfd, dfn * f / (dfn * f + dfd))
end

-- calcualate the standard error and p-values for regression
function regression_pvalues(data, coeffs, order)
    local n = #data
    if n < order + 1 then return nil end

    -- matrix to store the design matrix
    local X = {}
    for i = 1, n do
        local row = {}
        local x = data[i][1]
        for p = order, 0, -1 do
            table.insert(row, x^p)
        end
        X[i] = row
    end

    -- predicted values and residuals
    local y_pred = {}
    for i = 1, n do
        local pred = 0
        for j = 1, #coeffs do
            pred = pred + coeffs[j] * X[i][j]
        end
        y_pred[i] = pred
    end

    -- residual sum of squares
    local rss = 0
    for i = 1, n do
        local resid = data[i][2] - y_pred[i]
        rss = rss + resid^2
    end

    local dof = n - (#coeffs)
    if dof <= 0 then return nil end

    local sigma_squared = rss / dof

    -- compute (x^T * x)^-1
    local xtx = {}
    for i = 1, #coeffs do
        xtx[i] = {}
        for j = 1, #coeffs do
            local sum = 0
            for k = 1, n do
                sum = sum + X[k][i] * X[k][j]
            end
            xtx[i][j] = sum
        end
    end

    -- invert matrix
    local xtx_inv = improved_invert_matrix(xtx)
    if not xtx_inv then return nil end

    -- Standard errors
    local std_errs = {}
    for i = 1, #coeffs do
        std_errs[i] = math.sqrt(sigma_squared * xtx_inv[i][i])
    end

    -- p-value with normal approximation
    local p_values = {}
    for i = 1, #coeffs do
        local t_stat = coeffs[i] / std_errs[i]
        local p_val = 2 * (1 - normal_cdf(math.abs(t_stat)))
        p_values[i] = p_val
    end

    return std_errs, p_values
end

-- Overall F-test for regression significance
function regression_overall_pvalue(data, coeffs, order)
    local n = #data
    if n < order + 1 then return nil end

    -- matrix to store the design matrix
    local X = {}
    for i = 1, n do
        local row = {}
        local x = data[i][1]
        for p = order, 0, -1 do
            table.insert(row, x^p)
        end
        X[i] = row
    end

    -- predicted values and residuals
    local y_pred = {}
    for i = 1, n do
        local pred = 0
        for j = 1, #coeffs do
            pred = pred + coeffs[j] * X[i][j]
        end
        y_pred[i] = pred
    end

    -- residual sum of squares and total sum of squares
    local y_mean = 0
    for i = 1, n do y_mean = y_mean + data[i][2] end
    y_mean = y_mean / n

    local tss, rss = 0, 0
    for i = 1, n do
        tss = tss + (data[i][2] - y_mean)^2
        rss = rss + (data[i][2] - y_pred[i])^2
    end

    local df_model = order
    local df_resid = n - (order + 1)
    if df_resid <= 0 then return nil end

    local ms_model = (tss - rss) / df_model
    local ms_resid = rss / df_resid
    local f_stat = ms_model / ms_resid

    -- p value
    local p_value = 1 - f_cdf(f_stat, df_model, df_resid)
    return p_value
end