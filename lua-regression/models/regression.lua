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

local function format_power_equation(scale, exponent)
    local s = string.format("%sx^{%s}", format_display_number(scale), format_display_number(exponent))
    return s:gsub("%+ %-", "- ")
end

local function format_log_equation(slope, intercept)
    local s = string.format("%s \\ln(x) + %s", format_display_number(slope), format_display_number(intercept))
    return s:gsub("%+ %-", "- ")
end

-- Confidence banding
fit_model = nil

function generate_confidence_band(data, order, n_bootstrap, conf_level, ci_truncate, ci_xmin, ci_xmax, fit_options)
    -- Sort original data by x
    table.sort(data, function(a, b) return a[1] < b[1] end)
    local xs = {}
    if not ci_truncate and ci_xmin and ci_xmax then
        -- Use a grid over the specified range
        local n_points = math.max(50, math.min(#data, 200))
        local step = (ci_xmax - ci_xmin) / (n_points - 1)
        for i = 0, n_points - 1 do
            xs[#xs+1] = ci_xmin + i * step
        end
    else
        -- observed values only
        for i, pt in ipairs(data) do
            xs[i] = pt[1]
        end
    end

    local alpha = (1 - conf_level)/2
    n_bootstrap = n_bootstrap or 1000

    -- Store predicted values from each bootstrap
    local predictions = {}
    for i = 1, #xs do
        predictions[i] = {}
    end

    -- Draw bootstrap samples and compute fits
    for b = 1, n_bootstrap do
        -- Sample with replacement
        local sample = {}
        for _ = 1, #data do
            local idx = math.random(#data)
            table.insert(sample, {data[idx][1], data[idx][2]})
        end

        local bootstrap_model = fit_model(sample, fit_options or {order = order})
        if bootstrap_model then
            for i, xval in ipairs(xs) do
                local yval = bootstrap_model.predict(xval)
                if yval and yval == yval and math.abs(yval) < 1e20 then
                    table.insert(predictions[i], yval)
                end
            end
        end
    end

    -- each x sort and pick lower/upper percentiles
    local lower_band, upper_band = {}, {}
    for i, preds in ipairs(predictions) do
        table.sort(preds)
        if #preds > 0 then
            local lower_idx = math.floor(#preds * alpha + 0.5)
            local upper_idx = math.floor(#preds * (1 - alpha) + 0.5)
            lower_idx = math.max(lower_idx, 1)
            upper_idx = math.min(upper_idx, #preds)
            table.insert(lower_band, {xs[i], preds[lower_idx] or preds[1]})
            table.insert(upper_band, {xs[i], preds[upper_idx] or preds[#preds]})
        end
    end

    -- Format for pgfplots
    local function coords(array)
        local parts = {}
        for _, pt in ipairs(array) do
            if pt[1] and pt[2] and pt[1] == pt[1] and pt[2] == pt[2] and math.abs(pt[1]) < 1e15 and math.abs(pt[2]) < 1e15 then
                parts[#parts+1] = string.format("(%.6f,%.6f)", pt[1], pt[2])
            end
        end
        return table.concat(parts, " ")
    end

    return coords(lower_band), coords(upper_band)
end

-- Named numerical constants (each can be overridden via the options table)
LUAREG_IRLS_MAX_ITER  = 100
LUAREG_LASSO_MAX_ITER = 1000
LUAREG_IRLS_TOL       = 1e-8
LUAREG_LASSO_TOL      = 1e-10
LUAREG_POISSON_FLOOR  = 0.1
LUAREG_QUANTILE_EPS   = 1e-6
LUAREG_ETA_MAX        = 50

local function eval_horner(coeffs, x)
    local result = 0
    for i = 1, #coeffs do
        result = result * x + coeffs[i]
    end
    return result
end

-- Regularised polynomial regression (Ridge and Lasso).
-- 1. Scales x to u in [0, 1] so high powers do not overflow or lose precision.
-- 2. Standardises each non-intercept basis column u^p to zero mean and unit
--    standard deviation (unless options["standardize"] == false) and centers y,
--    so lambda is scale-invariant and the intercept is never penalised.
-- 3. Solves Ridge via augmented Householder QR (never forming X^T X) and Lasso
--    via coordinate descent initialized from the QR solution.
-- 4. Unstandardises back to the u-basis and maps to original x-coefficients via
--    denormalize_coefficients (descending powers of x).
local function fit_regularized(data, order, lambda, lasso, options)
    options = options or {}
    local n = #data
    if n < order + 1 then return nil end

    local norm_data, x_min, x_range = normalize_data(data)
    local standardize = (options["standardize"] ~= false)

    -- Build non-intercept polynomial columns in descending power: u^order, ..., u^1
    local F = {}
    local y_sum = 0
    for i = 1, n do
        local u = norm_data[i][1]
        local row = {}
        for p = order, 1, -1 do
            row[#row + 1] = u ^ p
        end
        F[i] = row
        y_sum = y_sum + norm_data[i][2]
    end
    local y_mean = y_sum / n

    local mu, sd = {}, {}
    for k = 1, order do
        if standardize then
            local s = 0
            for i = 1, n do s = s + F[i][k] end
            mu[k] = s / n
            local v = 0
            for i = 1, n do v = v + (F[i][k] - mu[k]) ^ 2 end
            sd[k] = math.sqrt(v / n)
        else
            mu[k] = 0
            sd[k] = 1
        end
    end

    local Z, y_centered = {}, {}
    for i = 1, n do
        local row = {}
        for k = 1, order do
            row[k] = (sd[k] > 0) and ((F[i][k] - mu[k]) / sd[k]) or 0
        end
        Z[i] = row
        y_centered[i] = norm_data[i][2] - y_mean
    end

    local gamma, cov_gamma_inv
    if not lasso then
        gamma, cov_gamma_inv = qr_weighted_least_squares(Z, y_centered, nil, lambda, true)
        if not gamma then return nil end
    else
        gamma = qr_weighted_least_squares(Z, y_centered, nil, math.max(lambda * 1e-4, 1e-8), true)
        if not gamma then
            gamma = {}
            for k = 1, order do gamma[k] = 0 end
        end

        local col_norm2 = {}
        for k = 1, order do
            local s = 0
            for i = 1, n do s = s + Z[i][k] ^ 2 end
            col_norm2[k] = s
        end

        local r = {}
        for i = 1, n do
            local pred = 0
            for k = 1, order do pred = pred + gamma[k] * Z[i][k] end
            r[i] = y_centered[i] - pred
        end

        local max_iter = options["lasso_max_iter"] or options["max_iter"] or LUAREG_LASSO_MAX_ITER
        local tol = options["lasso_tol"] or options["tol"] or LUAREG_LASSO_TOL
        local threshold = lambda / 2

        for _ = 1, max_iter do
            local changed = 0
            for j = 1, order do
                local scale_j = col_norm2[j]
                if scale_j > 0 then
                    local old = gamma[j]
                    local rho = 0
                    for i = 1, n do
                        local r_no_j = r[i] + old * Z[i][j]
                        rho = rho + Z[i][j] * r_no_j
                    end
                    local updated
                    if rho > threshold then
                        updated = (rho - threshold) / scale_j
                    elseif rho < -threshold then
                        updated = (rho + threshold) / scale_j
                    else
                        updated = 0
                    end
                    local delta = updated - old
                    if delta ~= 0 then
                        for i = 1, n do r[i] = r[i] - delta * Z[i][j] end
                        gamma[j] = updated
                        changed = math.max(changed, math.abs(delta))
                    end
                else
                    gamma[j] = 0
                end
            end
            if changed < tol then break end
        end
    end

    -- Unstandardise to descending u-basis coefficients [alpha_order, ..., alpha_1, alpha_0]
    local norm_coeff = {}
    local intercept_u = y_mean
    for k = 1, order do
        local ak = (sd[k] > 0) and (gamma[k] / sd[k]) or 0
        norm_coeff[k] = ak
        intercept_u = intercept_u - ak * mu[k]
    end
    norm_coeff[order + 1] = intercept_u

    local coeffs = denormalize_coefficients(norm_coeff, x_min, x_range)
    local predict = function(self_or_x, maybe_x)
        local x = maybe_x or self_or_x
        return eval_horner(coeffs, x)
    end

    -- Compute effective degrees of freedom, sandwich/active-set covariance,
    -- coefficient Student-t p-values, and overall F-test p-value.
    local rss, tss = 0, 0
    for i = 1, n do
        local yi = data[i][2]
        rss = rss + (yi - predict(data[i][1])) ^ 2
        tss = tss + (yi - y_mean) ^ 2
    end

    local C_gamma, df_model, df_resid
    if not lasso then
        -- G = Z^T Z, A = (Z^T Z + lambda I)^{-1}
        local G = {}
        for j = 1, order do
            G[j] = {}
            for k = 1, order do
                local s = 0
                for i = 1, n do s = s + Z[i][j] * Z[i][k] end
                G[j][k] = s
            end
        end
        if cov_gamma_inv then
            local M = {}
            for j = 1, order do
                M[j] = {}
                for k = 1, order do
                    local s = 0
                    for l = 1, order do s = s + cov_gamma_inv[j][l] * G[l][k] end
                    M[j][k] = s
                end
            end
            local tr_M, tr_2M_M2 = 0, 0
            for j = 1, order do
                tr_M = tr_M + M[j][j]
                local m2_jj = 0
                for k = 1, order do m2_jj = m2_jj + M[j][k] * M[k][j] end
                tr_2M_M2 = tr_2M_M2 + (2 * M[j][j] - m2_jj)
            end
            df_model = math.max(tr_M, 1e-9)
            df_resid = n - 1 - tr_2M_M2
            C_gamma = {}
            for j = 1, order do
                C_gamma[j] = {}
                for k = 1, order do
                    local s = 0
                    for l = 1, order do s = s + M[j][l] * cov_gamma_inv[l][k] end
                    C_gamma[j][k] = s
                end
            end
        end
    else
        -- Zou, Hastie & Tibshirani (2007) active-set degrees of freedom for Lasso
        local active = {}
        for k = 1, order do
            if math.abs(gamma[k]) > 0 then active[#active + 1] = k end
        end
        df_model = #active
        df_resid = n - 1 - df_model
        C_gamma = {}
        for j = 1, order do
            C_gamma[j] = {}
            for k = 1, order do C_gamma[j][k] = 0 end
        end
        if df_model > 0 and df_resid > 0 then
            local Z_act = {}
            for i = 1, n do
                local row = {}
                for idx, k in ipairs(active) do row[idx] = Z[i][k] end
                Z_act[i] = row
            end
            local _, cov_act = qr_least_squares(Z_act, y_centered)
            if cov_act then
                for j_idx, j_col in ipairs(active) do
                    for k_idx, k_col in ipairs(active) do
                        C_gamma[j_col][k_col] = cov_act[j_idx][k_idx]
                    end
                end
            end
        end
    end

    local pvals, overall_p
    if C_gamma and df_resid and df_resid > 0 then
        local C_alpha = {}
        for j = 1, order + 1 do C_alpha[j] = {} end
        for j = 1, order do
            for k = 1, order do
                C_alpha[j][k] = (sd[j] > 0 and sd[k] > 0) and (C_gamma[j][k] / (sd[j] * sd[k])) or 0
            end
        end
        for j = 1, order do
            local s = 0
            for k = 1, order do s = s - mu[k] * C_alpha[j][k] end
            C_alpha[j][order + 1] = s
            C_alpha[order + 1][j] = s
        end
        local intercept_var = 1 / n
        for j = 1, order do
            for k = 1, order do
                intercept_var = intercept_var + mu[j] * mu[k] * C_alpha[j][k]
            end
        end
        C_alpha[order + 1][order + 1] = intercept_var

        local C_x = denormalize_covariance(C_alpha, x_min, x_range)
        local sigma2 = rss / df_resid
        pvals = {}
        for j = 1, order + 1 do
            if lasso and j <= order and math.abs(gamma[j]) == 0 then
                pvals[j] = 1
            else
                local se = math.sqrt(math.max(sigma2 * (C_x[j] and C_x[j][j] or 0), 0))
                if se == 0 then
                    pvals[j] = (coeffs[j] == 0) and 1 or 0
                else
                    pvals[j] = luareg_stats.t_two_sided_p(coeffs[j] / se, df_resid)
                end
            end
        end
        if df_model == 0 then
            overall_p = 1
        elseif tss > 0 then
            if rss == 0 then
                overall_p = 0
            else
                local f_stat = (math.max(tss - rss, 0) / df_model) / sigma2
                overall_p = luareg_stats.f_upper_tail(f_stat, df_model, df_resid)
            end
        end
    end

    return {
        coeffs = coeffs,
        pvalues = pvals,
        overall_p = overall_p,
        predict = predict,
        equation = function() return generate_polynomial_equation(coeffs) end,
        display_equation = function() return format_equation_display(coeffs) end,
        plot_equation = function() return generate_polynomial_equation(coeffs) end
    }
end

fit_model = function(data, options)
    options = options or {}
    local raw_type = options["regression_type"] or options["regression type"] or options["type"] or "polynomial"
    local model_type = string.lower(tostring(raw_type):gsub("%s+", ""))
    local order = math.max(1, math.floor(options["order"] or 1))
    local lambda = math.max(0, options["lambda"] or 1)

    if model_type == "polynomial" then
        local coeffs = polynomial_regression(data, order)
        if not coeffs then return nil end
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return eval_horner(coeffs, x)
        end
        return {
            coeffs = coeffs,
            predict = predict,
            equation = function() return generate_polynomial_equation(coeffs) end,
            display_equation = function() return format_equation_display(coeffs) end,
            plot_equation = function() return generate_polynomial_equation(coeffs) end
        }
    elseif model_type == "ridge" or model_type == "lasso" then
        return fit_regularized(data, order, lambda, model_type == "lasso", options)
    elseif model_type == "power" or model_type == "power-series" or model_type == "powerseries" then
        local transformed = {}
        for _, point in ipairs(data) do
            if point[1] > 0 and point[2] > 0 then
                transformed[#transformed + 1] = {math.log(point[1]), math.log(point[2])}
            end
        end
        if #transformed < 2 then return nil end
        local coeffs = polynomial_regression(transformed, 1)
        if not coeffs then return nil end
        local scale, exponent = math.exp(coeffs[2]), coeffs[1]
        local _, pvals = regression_pvalues(transformed, coeffs, 1)
        local overall_p = regression_overall_pvalue(transformed, coeffs, 1)
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return scale * (x ^ exponent)
        end
        return {
            coeffs = {scale, exponent},
            pvalues = pvals,
            overall_p = overall_p,
            predict = predict,
            equation = function() return string.format("{%s * x^(%s)}", format_plot_number(scale), format_plot_number(exponent)) end,
            display_equation = function() return format_power_equation(scale, exponent) end,
            plot_equation = function() return string.format("{%s*x^(%s)}", format_plot_number(scale), format_plot_number(exponent)) end
        }
    elseif model_type == "logarithmic" or model_type == "log" then
        local transformed = {}
        for _, point in ipairs(data) do
            if point[1] > 0 then
                transformed[#transformed + 1] = {math.log(point[1]), point[2]}
            end
        end
        if #transformed < 2 then return nil end
        local coeffs = polynomial_regression(transformed, 1)
        if not coeffs then return nil end
        local _, pvals = regression_pvalues(transformed, coeffs, 1)
        local overall_p = regression_overall_pvalue(transformed, coeffs, 1)
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return coeffs[1] * math.log(x) + coeffs[2]
        end
        local plot_str = string.format("{%s*ln(x) + %s}", format_plot_number(coeffs[1]), format_plot_number(coeffs[2])):gsub("%+ %-", "- ")
        return {
            coeffs = coeffs,
            pvalues = pvals,
            overall_p = overall_p,
            predict = predict,
            equation = function() return string.format("{%s * ln(x) + %s}", format_plot_number(coeffs[1]), format_plot_number(coeffs[2])):gsub("%+ %-", "- ") end,
            display_equation = function() return format_log_equation(coeffs[1], coeffs[2]) end,
            plot_equation = function() return plot_str end
        }
    elseif model_type == "exponential" or model_type == "exp" then
        local transformed = {}
        for _, point in ipairs(data) do
            if point[2] > 0 then
                transformed[#transformed + 1] = {point[1], math.log(point[2])}
            end
        end
        if #transformed < 2 then return nil end
        local coeffs = polynomial_regression(transformed, 1)
        if not coeffs then return nil end
        local scale, rate = math.exp(coeffs[2]), coeffs[1]
        local _, pvals = regression_pvalues(transformed, coeffs, 1)
        local overall_p = regression_overall_pvalue(transformed, coeffs, 1)
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return scale * math.exp(rate * x)
        end
        return {
            coeffs = {scale, rate},
            pvalues = pvals,
            overall_p = overall_p,
            predict = predict,
            equation = function() return string.format("{%s * exp(%s * x)}", format_plot_number(scale), format_plot_number(rate)) end,
            display_equation = function() return string.format("%se^{%sx}", format_display_number(scale), format_display_number(rate)) end,
            plot_equation = function() return string.format("{%s*exp(%s*x)}", format_plot_number(scale), format_plot_number(rate)) end
        }
    elseif model_type == "poisson" then
        local n = #data
        if n < order + 1 then return nil end
        local poisson_floor = options["poisson_floor"] or LUAREG_POISSON_FLOOR
        local max_iter = options["max_iter"] or LUAREG_IRLS_MAX_ITER
        local tol = options["tol"] or LUAREG_IRLS_TOL
        local eta_max = options["eta_max"] or LUAREG_ETA_MAX

        local norm_data, x_min, x_range = normalize_data(data)
        local X_u = design_matrix(norm_data, order)
        local y_log = {}
        for i = 1, n do
            y_log[i] = math.log(math.max(data[i][2], poisson_floor))
        end

        local alpha, cov_u = qr_least_squares(X_u, y_log)
        if not alpha then return nil end

        for _ = 1, max_iter do
            local weights, working = {}, {}
            for i = 1, n do
                local eta = eval_horner(alpha, norm_data[i][1])
                eta = math.min(math.max(eta, -eta_max), eta_max)
                local mean = math.exp(eta)
                weights[i] = mean
                working[i] = eta + (data[i][2] - mean) / mean
            end
            local updated, updated_cov = qr_weighted_least_squares(X_u, working, weights)
            if not updated then break end
            local change = 0
            for j = 1, #alpha do
                local scale_j = math.max(1, math.abs(alpha[j]))
                change = math.max(change, math.abs(updated[j] - alpha[j]) / scale_j)
            end
            alpha, cov_u = updated, updated_cov
            if change < tol then break end
        end

        local coeffs = denormalize_coefficients(alpha, x_min, x_range)
        local cov = cov_u and denormalize_covariance(cov_u, x_min, x_range) or nil

        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            local eta = eval_horner(coeffs, x)
            return math.exp(math.min(math.max(eta, -eta_max), eta_max))
        end
        local pvals, overall_p = poisson_inference(data, coeffs, cov, predict, order, options["dispersion"])
        local r2_dev = poisson_pseudo_r2(data, predict)

        local inner_plot = generate_polynomial_equation(coeffs):sub(2, -2)
        local inner_disp = format_equation_display(coeffs)
        return {
            coeffs = coeffs,
            pvalues = pvals,
            overall_p = overall_p,
            r_squared = r2_dev,
            predict = predict,
            equation = function() return string.format("{exp(%s)}", inner_plot) end,
            display_equation = function() return string.format("e^{%s}", inner_disp) end,
            plot_equation = function() return string.format("{exp(%s)}", inner_plot) end
        }
    elseif model_type == "quantile" then
        local n = #data
        if n < order + 1 then return nil end
        local tau = math.min(1, math.max(0, options["quantile"] or 0.5))
        local max_iter = options["max_iter"] or LUAREG_IRLS_MAX_ITER
        local tol = options["tol"] or LUAREG_IRLS_TOL
        local q_eps = options["quantile_eps"] or LUAREG_QUANTILE_EPS

        local norm_data, x_min, x_range = normalize_data(data)
        local X_u, y_vec = design_matrix(norm_data, order)
        local alpha, cov_u_ols = qr_least_squares(X_u, y_vec)
        if not alpha then return nil end

        for _ = 1, max_iter do
            local weights = {}
            for i = 1, n do
                local eta = eval_horner(alpha, norm_data[i][1])
                local residual = y_vec[i] - eta
                weights[i] = (residual >= 0 and tau or (1 - tau)) / math.max(math.abs(residual), q_eps)
            end
            local updated = qr_weighted_least_squares(X_u, y_vec, weights)
            if not updated then break end
            local change = 0
            for j = 1, #alpha do
                local scale_j = math.max(1, math.abs(alpha[j]))
                change = math.max(change, math.abs(updated[j] - alpha[j]) / scale_j)
            end
            alpha = updated
            if change < tol then break end
        end

        local coeffs = denormalize_coefficients(alpha, x_min, x_range)
        local cov_x_ols = cov_u_ols and denormalize_covariance(cov_u_ols, x_min, x_range) or nil
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return eval_horner(coeffs, x)
        end
        local r2_pseudo = quantile_pseudo_r2(data, predict, tau)
        local pvals, overall_p = quantile_inference(data, coeffs, cov_x_ols, predict, order, tau)
        return {
            coeffs = coeffs,
            pvalues = pvals,
            overall_p = overall_p,
            r_squared = r2_pseudo,
            predict = predict,
            equation = function() return generate_polynomial_equation(coeffs) end,
            display_equation = function() return format_equation_display(coeffs) end,
            plot_equation = function() return generate_polynomial_equation(coeffs) end
        }
    end
    log_error("\\textbf{Error:} Unsupported regression type: " .. model_type)
    return nil
end