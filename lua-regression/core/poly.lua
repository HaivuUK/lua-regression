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

LUAREG_EPS = 2.220446049250313e-16      -- IEEE double machine epsilon
LUAREG_RANK_TOL = 1e-12                 -- relative tolerance for rank deficiency

-- Householder QR least squares.
-- X is an n-by-p matrix (table of rows), y a table of n numbers.
-- Returns beta (p coefficients) and M = (X^T X)^-1 computed from R^-1 R^-T,
-- or nil if X is (numerically) rank deficient. Avoids forming X^T X, so the
-- condition number is not squared.
function qr_least_squares(X, y)
    local n, p = #X, #X[1]
    if n < p then return nil end

    local A, b = {}, {}
    for i = 1, n do
        A[i] = {}
        for j = 1, p do A[i][j] = X[i][j] end
        b[i] = y[i]
    end

    local max_diag = 0
    for k = 1, p do
        local norm = 0
        for i = k, n do norm = norm + A[i][k] ^ 2 end
        norm = math.sqrt(norm)
        if norm == 0 then return nil end
        local alpha = (A[k][k] > 0) and -norm or norm
        local v = {}
        for i = k, n do v[i] = A[i][k] end
        v[k] = v[k] - alpha
        local vnorm2 = 0
        for i = k, n do vnorm2 = vnorm2 + v[i] ^ 2 end
        if vnorm2 > 0 then
            for j = k, p do
                local dot = 0
                for i = k, n do dot = dot + v[i] * A[i][j] end
                local f = 2 * dot / vnorm2
                for i = k, n do A[i][j] = A[i][j] - f * v[i] end
            end
            local dot = 0
            for i = k, n do dot = dot + v[i] * b[i] end
            local f = 2 * dot / vnorm2
            for i = k, n do b[i] = b[i] - f * v[i] end
        end
        max_diag = math.max(max_diag, math.abs(A[k][k]))
    end

    for k = 1, p do
        if math.abs(A[k][k]) <= LUAREG_RANK_TOL * max_diag then return nil end
    end

    -- Back substitution for beta
    local beta = {}
    for i = p, 1, -1 do
        local s = b[i]
        for j = i + 1, p do s = s - A[i][j] * beta[j] end
        beta[i] = s / A[i][i]
    end

    -- Rinv (upper triangular), then M = Rinv * Rinv^T
    local Rinv = {}
    for i = 1, p do Rinv[i] = {} for j = 1, p do Rinv[i][j] = 0 end end
    for col = 1, p do
        for i = col, 1, -1 do
            local s = (i == col) and 1 or 0
            for j = i + 1, col do s = s - A[i][j] * Rinv[j][col] end
            Rinv[i][col] = s / A[i][i]
        end
    end
    local M = {}
    for i = 1, p do
        M[i] = {}
        for j = 1, p do
            local s = 0
            for k = math.max(i, j), p do s = s + Rinv[i][k] * Rinv[j][k] end
            M[i][j] = s
        end
    end
    return beta, M
end

-- Weighted (and optionally L2-regularised) least squares solved via Householder QR.
-- Avoids forming X^T W X.  `penalize_last` controls whether the last column
-- (the intercept in descending-power bases) is penalised when ridge > 0.
function qr_weighted_least_squares(X, y, weights, ridge, penalize_last)
    local n, p = #X, #X[1]
    local A, b = {}, {}
    for i = 1, n do
        local w = weights and weights[i] or 1
        local sw = math.sqrt(math.max(w, 0))
        local row = {}
        for j = 1, p do row[j] = sw * X[i][j] end
        A[i] = row
        b[i] = sw * y[i]
    end
    if ridge and ridge > 0 then
        local sr = math.sqrt(ridge)
        local last_col = (penalize_last == false) and (p - 1) or p
        for j = 1, last_col do
            local row = {}
            for k = 1, p do row[k] = (k == j) and sr or 0 end
            A[#A + 1] = row
            b[#b + 1] = 0
        end
    end
    return qr_least_squares(A, b)
end

-- Map a covariance matrix M in the normalised u = (x - x_min) / x_range basis
-- (descending powers) to the original x basis via T M T^T.
function denormalize_covariance(M, x_min, x_range)
    local p = #M
    local T = {}
    for i = 1, p do T[i] = {} end
    for j = 1, p do
        local unit = {}
        for i = 1, p do unit[i] = (i == j) and 1 or 0 end
        local col = denormalize_coefficients(unit, x_min, x_range)
        for i = 1, p do T[i][j] = col[i] end
    end
    local TM = {}
    for i = 1, p do
        TM[i] = {}
        for j = 1, p do
            local s = 0
            for k = 1, p do s = s + T[i][k] * M[k][j] end
            TM[i][j] = s
        end
    end
    local cov = {}
    for i = 1, p do
        cov[i] = {}
        for j = 1, p do
            local s = 0
            for k = 1, p do s = s + TM[i][k] * T[j][k] end
            cov[i][j] = s
        end
    end
    return cov
end

-- Build a polynomial design matrix (descending powers) for normalised x
function design_matrix(norm_data, order)
    local X, Y = {}, {}
    for i = 1, #norm_data do
        local row, x = {}, norm_data[i][1]
        for p = order, 0, -1 do row[#row + 1] = x ^ p end
        X[i] = row
        Y[i] = norm_data[i][2]
    end
    return X, Y
end

-- Polynomial regression fitted on x scaled to [0,1] with Householder QR.
-- Returns coefficients (descending powers, original scale) and, optionally,
-- the unscaled covariance matrix (X^T X)^-1 expressed in the original basis.
function polynomial_regression(data, order, want_covariance)
    local n = #data
    if n < order + 1 then
        log_error("\\textbf{Error:} Not enough data points for order " .. order .. " polynomial.\\\\")
        return nil
    end

    local norm_data, x_min, x_range = normalize_data(data)
    local X, Y = design_matrix(norm_data, order)
    local norm_coeff, M = qr_least_squares(X, Y)
    if not norm_coeff then
        log_error("\\textbf{Error:} Design matrix is rank deficient for order " .. order .. ".\\\\")
        return nil
    end

    local coeff = denormalize_coefficients(norm_coeff, x_min, x_range)
    if not want_covariance then return coeff end
    return coeff, denormalize_covariance(M, x_min, x_range)
end

-- Matrix inversion with partial pivoting and a relative singularity test
function improved_invert_matrix(m)
    local n = #m
    local scale = 0
    for i = 1, n do
        for j = 1, n do scale = math.max(scale, math.abs(m[i][j])) end
    end
    if scale == 0 then return nil end

    local aug = {}
    for i = 1, n do
        aug[i] = {}
        for j = 1, 2*n do
            if j <= n then
                aug[i][j] = m[i][j]
            else
                aug[i][j] = (i == j-n) and 1 or 0
            end
        end
    end

    for i = 1, n do
        local max_row, max_val = i, math.abs(aug[i][i])
        for k = i+1, n do
            if math.abs(aug[k][i]) > max_val then
                max_row = k
                max_val = math.abs(aug[k][i])
            end
        end

        -- Relative to the matrix scale, not an absolute constant
        if max_val <= n * LUAREG_EPS * scale then return nil end

        if max_row ~= i then
            aug[i], aug[max_row] = aug[max_row], aug[i]
        end

        local pivot = aug[i][i]
        for j = i, 2*n do aug[i][j] = aug[i][j] / pivot end

        for k = 1, n do
            if k ~= i then
                local factor = aug[k][i]
                for j = i, 2*n do
                    aug[k][j] = aug[k][j] - factor * aug[i][j]
                end
            end
        end
    end

    local inv = {}
    for i = 1, n do
        inv[i] = {}
        for j = 1, n do inv[i][j] = aug[i][j+n] end
    end
    return inv
end

-- Transform normalized coefficients back to original scale
function denormalize_coefficients(norm_coeff, x_min, x_range)
    local order = #norm_coeff - 1
    local coeff = {}
    for i = 1, order + 1 do coeff[i] = 0 end

    for i = 1, order + 1 do
        local power = order + 1 - i
        for j = 0, power do
            local bin_coeff = binomial(power, j)
            local idx = order + 1 - j
            coeff[idx] = coeff[idx] + norm_coeff[i] * bin_coeff * ((-x_min/x_range)^(power-j)) * ((1/x_range)^j)
        end
    end
    return coeff
end

-- Calculate binomial coefficient (n choose k)
function binomial(n, k)
    if k < 0 or k > n then return 0 end
    if k == 0 or k == n then return 1 end
    local result = 1
    for i = 1, k do
        result = result * (n - (i - 1)) / i
    end
    return result
end

-- Number formatting -------------------------------------------------------

-- For pgfplots: enough digits that high-order/small coefficients are not
-- rounded to zero.  Scientific notation (1.2e-06) is understood by PGF math.
function format_plot_number(c)
    local s = string.format("%.12g", c)
    s = s:gsub("e%+0*(%d+)", "e%1"):gsub("e%-0*(%d+)", "e-%1")
    return s
end

-- For typeset output: fixed notation unless the value is small, in which case
-- significant figures are kept and written as m\times10^{e}.
function format_display_number(c, sig)
    sig = sig or 4
    if c == 0 or math.abs(c) >= 0.01 then
        return string.format("%." .. sig .. "f", c)
    end
    local s = string.format("%." .. sig .. "g", c)
    local mant, exp = s:match("^(-?[%d%.]+)e([-+]%d+)$")
    if mant then
        return string.format("%s\\times10^{%d}", mant, tonumber(exp))
    end
    return s
end

local function join_terms(terms)
    local eq = table.concat(terms, " + ")
    eq = eq:gsub("%+ %-", "- ")
    return eq
end

-- Polynomial equation string for pgfplots (descending powers)
function generate_polynomial_equation(coefficients)
    local terms = {}
    for i, coeff in ipairs(coefficients) do
        local power = #coefficients - i
        local num = format_plot_number(coeff)
        if power == 0 then
            terms[#terms + 1] = num
        elseif power == 1 then
            terms[#terms + 1] = num .. " * x"
        else
            terms[#terms + 1] = string.format("%s * x^%d", num, power)
        end
    end
    return "{" .. join_terms(terms) .. "}"
end

-- Equation string for pgfplots legend
function format_equation_plot(coefficients)
    local terms = {}
    for i, coeff in ipairs(coefficients) do
        local power = #coefficients - i
        if coeff ~= 0 then
            local num = format_plot_number(coeff)
            if power == 0 then
                terms[#terms + 1] = num
            elseif power == 1 then
                terms[#terms + 1] = num .. "*x"
            else
                terms[#terms + 1] = string.format("%s*x^(%d)", num, power)
            end
        end
    end
    return join_terms(terms)
end

-- Typeset equation
function format_equation_display(coefficients)
    local terms = {}
    for i, coeff in ipairs(coefficients) do
        local power = #coefficients - i
        if coeff ~= 0 then
            local num = format_display_number(coeff)
            if power == 0 then
                terms[#terms + 1] = num
            elseif power == 1 then
                terms[#terms + 1] = num .. "x"
            else
                terms[#terms + 1] = string.format("%sx^{%d}", num, power)
            end
        end
    end
    if #terms == 0 then terms[1] = "0" end
    return join_terms(terms)
end

-- Coefficient of determination.  Returns nil when it is undefined
-- (fewer than 2 points or zero variance in y).
function calculate_r_squared(data, predict_func)
    local n = #data
    if n < 2 then return nil end
    local sum_y = 0
    for _, point in ipairs(data) do sum_y = sum_y + point[2] end
    local mean_y = sum_y / n
    local ss_tot, ss_res = 0, 0
    for _, point in ipairs(data) do
        local y_pred = predict_func(point[1])
        ss_res = ss_res + (point[2] - y_pred)^2
        ss_tot = ss_tot + (point[2] - mean_y)^2
    end
    if ss_tot == 0 then return nil end
    return 1 - (ss_res / ss_tot)
end

-- Outlier filter using Z-scores (sample standard deviation).
-- A variable with zero spread cannot produce outliers and is ignored.
function filter_outliers(data, threshold)
    if not threshold then return data end
    local n = #data
    if n < 3 then return data end

    local sum_x, sum_y = 0, 0
    for _, pt in ipairs(data) do
        sum_x = sum_x + pt[1]
        sum_y = sum_y + pt[2]
    end
    local mean_x, mean_y = sum_x / n, sum_y / n

    local variance_x, variance_y = 0, 0
    for _, pt in ipairs(data) do
        variance_x = variance_x + (pt[1] - mean_x)^2
        variance_y = variance_y + (pt[2] - mean_y)^2
    end
    local std_x = math.sqrt(variance_x / (n - 1))
    local std_y = math.sqrt(variance_y / (n - 1))

    local filtered = {}
    for _, pt in ipairs(data) do
        local zx = std_x > 0 and math.abs(pt[1] - mean_x) / std_x or 0
        local zy = std_y > 0 and math.abs(pt[2] - mean_y) / std_y or 0
        if zx < threshold and zy < threshold then
            filtered[#filtered + 1] = pt
        end
    end
    return filtered
end
