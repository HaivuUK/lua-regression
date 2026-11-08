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
    local s = string.format("%.4fx^{%.4f}", scale, exponent)
    return s:gsub("%+ %-", "- ")
end

local function format_log_equation(slope, intercept)
    local s = string.format("%.4f \\ln(x) + %.4f", slope, intercept)
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
        local n_points = math.max(50, #data)
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
                table.insert(predictions[i], bootstrap_model.predict(xval))
            end
        end
    end

    -- each x sort and pick lower/upper percentiles
    local lower_band, upper_band = {}, {}
    for i, preds in ipairs(predictions) do
        table.sort(preds)
        local lower_idx = math.floor(#preds * alpha + 0.5)
        local upper_idx = math.floor(#preds * (1 - alpha) + 0.5)
        lower_idx = math.max(lower_idx, 1)
        upper_idx = math.min(upper_idx, #preds)
        table.insert(lower_band, {xs[i], preds[lower_idx] or preds[1]})
        table.insert(upper_band, {xs[i], preds[upper_idx] or preds[#preds]})
    end

    -- Format for pgfplots
    local function coords(array)
        local parts = {}
        for _, pt in ipairs(array) do
            parts[#parts+1] = string.format("(%f,%f)", pt[1], pt[2])
        end
        return table.concat(parts, " ")
    end

    return coords(lower_band), coords(upper_band)
end

local function solve_weighted_least_squares(data, basis, weights, ridge)
    local p = #basis(data[1][1]) -- number of basis functions
    local xtx, xty = {}, {}
    for i = 1, p do
        xtx[i], xty[i] = {}, 0
        for j = 1, p do
            xtx[i][j] = 0
        end
    end
    for row, point in ipairs(data) do
        local features = basis(point[1])
        local weight = weights and weights[row] or 1
        for i = 1, p do
            xty[i] = xty[i] + weight * features[i] * point[2]
            for j = 1, p do
                xtx[i][j] = xtx[i][j] + weight * features[i] * features[j]
            end
        end
    end
    for i = 2, p do xtx[i][i] = xtx[i][i] + (ridge or 0) end
    local inverse = improved_invert_matrix(xtx)
    if not inverse then return nil end
    local coeffs = {}
    for i = 1, p do
        coeffs[i] = 0
        for j = 1, p do
            coeffs[i] = coeffs[i] + inverse[i][j] * xty[j]
        end
    end
    return coeffs
end

local function polynomial_basis(order)
    return function(x)
        local result = {1}
        for power = 1, order do result[#result+1] = x ^ power end
        return result
    end
end

local function format_basis_plot(coeffs)
    local terms = {}
    for i = #coeffs, 1, -1 do
        local c = coeffs[i]
        local p = i - 1
        if math.abs(c) > 1e-9 or #coeffs == 1 then
            if p == 0 then
                table.insert(terms, string.format("%.4f", c))
            elseif p == 1 then
                table.insert(terms, string.format("%.4f*x", c))
            else
                table.insert(terms, string.format("%.4f*x^%d", c, p))
            end
        end
    end
    if #terms == 0 then table.insert(terms, "0") end
    local eq = table.concat(terms, " + "):gsub("%+ %-", "- ")
    return "{" .. eq .. "}"
end

local function format_basis_display(coeffs)
    local terms = {}
    for i = #coeffs, 1, -1 do
        local c = coeffs[i]
        local p = i - 1
        if math.abs(c) > 1e-9 or #coeffs == 1 then
            if p == 0 then
                table.insert(terms, string.format("%.4f", c))
            elseif p == 1 then
                table.insert(terms, string.format("%.4fx", c))
            else
                table.insert(terms, string.format("%.4fx^{%d}", c, p))
            end
        end
    end
    if #terms == 0 then table.insert(terms, "0") end
    return table.concat(terms, " + "):gsub("%+ %-", "- ")
end

local function format_basis_equation(coeffs)
    local terms = {}
    for i = #coeffs, 1, -1 do
        local c = coeffs[i]
        local p = i - 1
        if math.abs(c) > 1e-9 or #coeffs == 1 then
            if p == 0 then
                table.insert(terms, string.format("%.9f", c))
            elseif p == 1 then
                table.insert(terms, string.format("%.9f * x", c))
            else
                table.insert(terms, string.format("%.9f * x^%d", c, p))
            end
        end
    end
    if #terms == 0 then table.insert(terms, "0") end
    return "{" .. table.concat(terms, " + "):gsub("%+ %-", "- ") .. "}"
end

local function fit_regularized(data, order, lambda, lasso)
    local basis = polynomial_basis(order)
    local coeffs = solve_weighted_least_squares(data, basis, nil, lasso and 0 or lambda)
    if not coeffs then return nil end
    if lasso then
        for _ = 1, 1000 do
            local changed = 0
            for j = 2, order + 1 do
                local rho, scale = 0, 0
                for _, point in ipairs(data) do
                    local features = basis(point[1])
                    local residual = point[2]
                    for k = 1, order + 1 do
                        if k ~= j then residual = residual - coeffs[k] * features[k] end
                    end
                    rho = rho + features[j] * residual
                    scale = scale + features[j] ^ 2
                end
                local old = coeffs[j]
                local threshold = lambda / 2
                if rho > threshold then
                    coeffs[j] = (rho - threshold) / scale
                elseif rho < -threshold then
                    coeffs[j] = (rho + threshold) / scale
                else
                    coeffs[j] = 0
                end
                changed = math.max(changed, math.abs(old - coeffs[j]))
            end
            local sum_res = 0
            for _, point in ipairs(data) do
                local r = point[2]
                local features = basis(point[1])
                for k = 2, order + 1 do
                    r = r - coeffs[k] * features[k]
                end
                sum_res = sum_res + r
            end
            coeffs[1] = sum_res / #data
            if changed < 1e-10 then break end
        end
    end
    return {
        coeffs = coeffs,
        predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            local features, result = basis(x), 0
            for i = 1, #coeffs do result = result + coeffs[i] * features[i] end
            return result
        end,
        equation = function() return format_basis_equation(coeffs) end,
        display_equation = function() return format_basis_display(coeffs) end,
        plot_equation = function() return format_basis_plot(coeffs) end
    }
end

fit_model = function(data, options)
    local raw_type = options["regression_type"] or options["regression type"] or options["type"] or "polynomial"
    local model_type = string.lower(tostring(raw_type):gsub("%s+", ""))
    local order = math.max(1, math.floor(options["order"] or 1))
    local lambda = math.max(0, options["lambda"] or 1)

    if model_type == "polynomial" then
        local coeffs = polynomial_regression(data, order)
        if not coeffs then return nil end
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            local result = 0
            for i, coeff in ipairs(coeffs) do
                result = result + coeff * (x ^ (#coeffs - i))
            end
            return result
        end
        return {
            coeffs = coeffs,
            predict = predict,
            equation = function() return generate_polynomial_equation(coeffs) end,
            display_equation = function() return format_equation_display(coeffs) end,
            plot_equation = function() return generate_polynomial_equation(coeffs) end
        }
    elseif model_type == "ridge" or model_type == "lasso" then
        return fit_regularized(data, order, lambda, model_type == "lasso")
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
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return scale * (x ^ exponent)
        end
        return {
            coeffs = {scale, exponent},
            predict = predict,
            equation = function() return string.format("{%.9f * x^{%.9f}}", scale, exponent) end,
            display_equation = function() return format_power_equation(scale, exponent) end,
            plot_equation = function() return string.format("{%.4f*x^(%.4f)}", scale, exponent) end
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
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return coeffs[1] * math.log(x) + coeffs[2]
        end
        local plot_str = string.format("{%.4f*ln(x) + %.4f}", coeffs[1], coeffs[2]):gsub("%+ %-", "- ")
        return {
            coeffs = coeffs,
            predict = predict,
            equation = function() return string.format("{%.9f * ln(x) + %.9f}", coeffs[1], coeffs[2]):gsub("%+ %-", "- ") end,
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
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            return scale * math.exp(rate * x)
        end
        return {
            coeffs = {scale, rate},
            predict = predict,
            equation = function() return string.format("{%.9f * exp(%.9f * x)}", scale, rate) end,
            display_equation = function() return string.format("%.4fe^{%.4fx}", scale, rate) end,
            plot_equation = function() return string.format("{%.4f*exp(%.4f*x)}", scale, rate) end
        }
    elseif model_type == "poisson" then
        local basis = polynomial_basis(order)
        local log_data = {}
        for _, pt in ipairs(data) do
            table.insert(log_data, {pt[1], math.log(math.max(pt[2], 0.1))})
        end
        local coeffs = solve_weighted_least_squares(log_data, basis)
        if not coeffs then return nil end
        for _ = 1, 100 do
            local weights, working = {}, {}
            for i, point in ipairs(data) do
                local features, eta = basis(point[1]), 0
                for j = 1, #coeffs do eta = eta + coeffs[j] * features[j] end
                local mean = math.exp(math.min(math.max(eta, -50), 50))
                weights[i] = mean
                working[i] = eta + (point[2] - mean) / mean
            end
            local pseudo_data = {}
            for i, point in ipairs(data) do pseudo_data[i] = {point[1], working[i]} end
            local updated = solve_weighted_least_squares(pseudo_data, basis, weights)
            if not updated then break end
            local change = 0
            for i = 1, #coeffs do change = math.max(change, math.abs(updated[i] - coeffs[i])) end
            coeffs = updated
            if change < 1e-8 then break end
        end
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            local features, eta = basis(x), 0
            for i = 1, #coeffs do eta = eta + coeffs[i] * features[i] end
            return math.exp(math.min(math.max(eta, -50), 50))
        end
        local inner_plot = format_basis_plot(coeffs):sub(2, -2)
        local inner_disp = format_basis_display(coeffs)
        return {
            coeffs = coeffs,
            predict = predict,
            equation = function() return string.format("{exp(%s)}", inner_plot) end,
            display_equation = function() return string.format("e^{%s}", inner_disp) end,
            plot_equation = function() return string.format("{exp(%s)}", inner_plot) end
        }
    elseif model_type == "quantile" then
        local basis = polynomial_basis(order)
        local coeffs = solve_weighted_least_squares(data, basis)
        if not coeffs then return nil end
        local tau = math.min(1, math.max(0, options["quantile"] or 0.5))
        for _ = 1, 100 do
            local weights, working = {}, {}
            for i, point in ipairs(data) do
                local features, eta = basis(point[1]), 0
                for j = 1, #coeffs do eta = eta + coeffs[j] * features[j] end
                local residual = point[2] - eta
                local w = (residual >= 0 and tau or (1 - tau)) / (math.abs(residual) + 1e-6)
                weights[i] = w
                working[i] = eta + residual
            end
            local pseudo_data = {}
            for i, point in ipairs(data) do pseudo_data[i] = {point[1], working[i]} end
            local updated = solve_weighted_least_squares(pseudo_data, basis, weights)
            if not updated then break end
            local change = 0
            for i = 1, #coeffs do change = math.max(change, math.abs(updated[i] - coeffs[i])) end
            coeffs = updated
            if change < 1e-8 then break end
        end
        local predict = function(self_or_x, maybe_x)
            local x = maybe_x or self_or_x
            local features, eta = basis(x), 0
            for i = 1, #coeffs do eta = eta + coeffs[i] * features[i] end
            return eta
        end
        return {
            coeffs = coeffs,
            predict = predict,
            equation = function() return format_basis_equation(coeffs) end,
            display_equation = function() return format_basis_display(coeffs) end,
            plot_equation = function() return format_basis_plot(coeffs) end
        }
    end
    log_error("\\textbf{Error:} Unsupported regression type: " .. model_type)
    return nil
end