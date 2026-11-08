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

-- Polynomial regression with improved numerical stability
function polynomial_regression(data, order)
    local n = #data
    if n < order + 1 then 
        log_error("\\textbf{Error:} Not enough data points for order " .. order .. " polynomial.\\\\")
        return nil 
    end
    
    -- Normalize data for better numerical stability
    local norm_data, x_min, x_range = normalize_data(data)
    
    -- Build design matrix X and vector Y
    local X = {}
    local Y = {}
    for i = 1, n do
        local row = {}
        local x = norm_data[i][1]
        for p = order, 0, -1 do
            table.insert(row, x^p)
        end
        X[i] = row
        Y[i] = { norm_data[i][2] }
    end

    -- Use Kahan summation for matrix products to reduce floating-point errors
    local function kahan_sum(values)
        local sum = 0.0
        local c = 0.0
        for _, v in ipairs(values) do
            local y = v - c
            local t = sum + y
            c = (t - sum) - y
            sum = t
        end
        return sum
    end

    -- Compute X^T * X with improved precision
    local xt = {}
    for i = 1, order+1 do
        xt[i] = {}
        for j = 1, n do
            xt[i][j] = X[j][i]
        end
    end

    local xtx = {}
    for i = 1, order+1 do
        xtx[i] = {}
        for j = 1, order+1 do
            local products = {}
            for k = 1, n do
                table.insert(products, xt[i][k] * X[k][j])
            end
            xtx[i][j] = kahan_sum(products)
        end
    end

    local xty = {}
    for i = 1, order+1 do
        local products = {}
        for k = 1, n do
            table.insert(products, xt[i][k] * Y[k][1])
        end
        xty[i] = { kahan_sum(products) }
    end

    local xtx_inv = improved_invert_matrix(xtx)
    if not xtx_inv then
        log_error("\\textbf{Error:} Matrix is ill-conditioned for order " .. order .. ".\\\\")
        return nil 
    end

    -- Solve for coefficients
    local norm_coeff = {}
    for i = 1, order+1 do
        local sum = 0
        for j = 1, order+1 do
            sum = sum + xtx_inv[i][j] * xty[j][1]
        end
        norm_coeff[i] = sum
    end

    -- Transform coefficcients to original scale
    local coeff = denormalize_coefficients(norm_coeff, x_min, x_range)
    return coeff
end

-- Matrix inversion with partial pivoting
function improved_invert_matrix(m)
    local n = #m
    -- Create augmented matrix [A|I]
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
    
    -- Gaussian elimination with partial pivoting
    for i = 1, n do
        -- Find pivot
        local max_row = i
        local max_val = math.abs(aug[i][i])
        for k = i+1, n do
            if math.abs(aug[k][i]) > max_val then
                max_row = k
                max_val = math.abs(aug[k][i])
            end
        end
        
        -- Check for singular matrix
        if max_val < 1e-14 then return nil end
        
        -- Swap rows if needed
        if max_row ~= i then
            aug[i], aug[max_row] = aug[max_row], aug[i]
        end
        
        -- Scale row
        local pivot = aug[i][i]
        for j = i, 2*n do
            aug[i][j] = aug[i][j] / pivot
        end
        
        -- Eliminate
        for k = 1, n do
            if k ~= i then
                local factor = aug[k][i]
                for j = i, 2*n do
                    aug[k][j] = aug[k][j] - factor * aug[i][j]
                end
            end
        end
    end
    
    -- Extract inverse
    local inv = {}
    for i = 1, n do
        inv[i] = {}
        for j = 1, n do
            inv[i][j] = aug[i][j+n]
        end
    end
    return inv
end

-- Transform normalized coefficients back to original scale
function denormalize_coefficients(norm_coeff, x_min, x_range)
    local order = #norm_coeff - 1
    local coeff = {}
    
    -- Start with a copy of normalized coefficients
    for i=1, order+1 do
        coeff[i] = 0
    end
    
    -- For each normalized coefficient
    for i=1, order+1 do
        local power = order + 1 - i
        
        -- Apply binomial expansion to convert back
        for j=0, power do
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

-- Function to generate the polynomial equation string for pgfplots
function generate_polynomial_equation(coefficients)
    local terms = {}
    for i, coeff in ipairs(coefficients) do
        local power = #coefficients - i -- Reverse the order to start with the highest power
        if power == 0 then
            table.insert(terms, string.format("%.9f", coeff)) -- Constant term
        elseif power == 1 then
            table.insert(terms, string.format("%.9f * x", coeff)) -- Linear term
        else
            table.insert(terms, string.format("%.9f * x^%d", coeff, power)) -- Higher-order terms
        end
    end

    -- Concatenate terms into a single string formatted for pgfplots
    return string.format("{%s}", table.concat(terms, " + "):gsub("%+ %-", "- "))
end

-- Function to generate the linear equation string for pgfplots legend
function format_equation_plot(coefficients)
    local terms = {}
    for i, coeff in ipairs(coefficients) do
        local power = #coefficients - i -- Start with the highest power
        if coeff ~= 0 then -- Skip zero coefficients
            if power == 0 then
                table.insert(terms, string.format("%.4f", coeff)) -- Constant term
            elseif power == 1 then
                table.insert(terms, string.format("%.4f*x", coeff)) -- Linear term
            else
                table.insert(terms, string.format("%.4f*x^(%d)", coeff, power)) -- Higher-order terms
            end
        end
    end

    -- Concatenate terms into a single string formatted for pgfplots
    local equation = table.concat(terms, " + ")

    -- Replace "+ -" with "-" for negative terms
    equation = equation:gsub("%+ %-", "- ")

    return equation
end

-- Format equation display
function format_equation_display(coefficients)
    local terms = {}
    for i, coeff in ipairs(coefficients) do
        local power = #coefficients - i -- Start with the highest power
        if coeff ~= 0 then -- Skip zero coefficients
            if power == 0 then
                terms[#terms + 1] = string.format("%.4f", coeff) -- Constant term
            elseif power == 1 then
                terms[#terms + 1] = string.format("%.4fx", coeff) -- Linear term
            else
                terms[#terms + 1] = string.format("%.4fx^{%d}", coeff, power) -- Higher-order terms
            end
        end
    end
    return table.concat(terms, " + "):gsub("%+ %-", "- ")
end

-- Computes R²
function calculate_r_squared(data, predict_func)
    local n = #data
    if n < 2 then return 0 end
    local sum_y = 0
    for _, point in ipairs(data) do
        sum_y = sum_y + point[2]
    end
    local mean_y = sum_y / n
    local ss_tot = 0
    local ss_res = 0
    for _, point in ipairs(data) do
        local y_obs = point[2]
        local y_pred = predict_func(point[1])
        ss_res = ss_res + (y_obs - y_pred)^2
        ss_tot = ss_tot + (y_obs - mean_y)^2
    end
    return 1 - (ss_res / ss_tot)
end

-- Outlier filter using Z-scores
function filter_outliers(data, threshold)
    if not threshold then return data end

    -- Calculate mean and std
    local n = #data
    local sum_x, sum_y = 0, 0
    for _, pt in ipairs(data) do
        sum_x = sum_x + pt[1]
        sum_y = sum_y + pt[2]
    end
    local mean_x = sum_x / n
    local mean_y = sum_y / n

    local variance_x, variance_y = 0, 0
    for _, pt in ipairs(data) do
        variance_x = variance_x + (pt[1] - mean_x)^2
        variance_y = variance_y + (pt[2] - mean_y)^2
    end
    local std_x = math.sqrt(variance_x / (n - 1))
    local std_y = math.sqrt(variance_y / (n - 1))

    local filtered = {}
    for _, pt in ipairs(data) do
        local zx = math.abs(pt[1] - mean_x) / std_x
        local zy = math.abs(pt[2] - mean_y) / std_y
        if zx < threshold and zy < threshold then
            table.insert(filtered, pt)
        end
    end
    return filtered
end