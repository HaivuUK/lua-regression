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

-- Improved split function that handles empty fields in CSV
function split(str, delim)
    local result = {}
    -- Handle edge case of empty string
    if str == "" then return result end
    
    -- Handle trailing delimiter with a final empty field
    if str:sub(-1) == delim then
        str = str .. " "  -- Add a space to create a final empty field
    end
    
    -- Special case for blank cells - two delimiters in a row
    str = str:gsub(delim..delim, delim.." "..delim)  -- Replace ,, with , ,
    
    -- Split the string using Lua pattern matching
    local pattern = string.format("([^%s]*)", delim)
    for field in str:gmatch(pattern) do
        table.insert(result, field)
    end
    
    return result
end

-- Function to find the index of a column in the header
function find_column_index(header, column_name)
	for i, name in ipairs(header) do
		if name == column_name then
			return i
		end
	end
	return nil -- Column name not found
end

-- Normalize data to improve numerical stability for polynomial regression
function normalize_data(data)
    local x_min, x_max = math.huge, -math.huge
    for _, point in ipairs(data) do
        x_min = math.min(x_min, point[1])
        x_max = math.max(x_max, point[1])
    end
    
    local x_range = x_max - x_min
    local normalized = {}
    for _, point in ipairs(data) do
        -- Normalize to [0,1] range
        local x_norm = (point[1] - x_min) / x_range
        table.insert(normalized, {x_norm, point[2]})
    end
    
    return normalized, x_min, x_range
end

-- Function to read a CSV file and store the data in a table
function read_csv_to_table(file_path, xcol, ycol, options)
    local data = {}
    local file = io.open(file_path, "r")
    if not file then
        if options and options["debug"] then
            log_error("\\textbf{Error:} Unable to open file " .. file_path)
        end
        return nil
    end

    -- Read first line (header) and remove BOM if present
    local header_line = file:read("*l") or ""
    header_line = header_line:gsub("^\239\187\191", "") -- Remove UTF-8 BOM
    header_line = header_line:gsub("^%s*(.-)%s*$", "%1") -- Trim
    local header = split(header_line, ',')

    -- Debug: Print headers
    if options and options["debug"] then
        log_error("\\textbf{Debug:} Headers: " .. table.concat(header, ", ") .. "\\\\")
    end

    -- If xcol / ycol are names, convert them to column indices
    if type(xcol) == "string" then
        local idx = find_column_index(header, xcol)
        if not idx then
            if options and options["debug"] then
                log_error("\\textbf{Error:} Column '" .. xcol .. "' not found in header")
            end
            file:close()
            return nil
        end
        xcol = idx
    end

    if type(ycol) == "string" then
        local idx = find_column_index(header, ycol)
        if not idx then
            if options and options["debug"] then
                log_error("\\textbf{Error:} Column '" .. ycol .. "' not found in header.")
            end
            file:close()
            return nil
        end
        ycol = idx
    end

    -- Read the data lines
    local line_number = 1
    local skipped_lines = 0
    while true do
        local line = file:read("*l")
        if not line then break end

        line = line:gsub("^%s*(.-)%s*$", "%1") -- Trim
        if line ~= "" then
            local values = split(line, ',')

            -- Check if we have enough columns
            if #values < math.max(xcol, ycol) then
                if options and options["debug"] then
                    log_error("\\textbf{Warning:} Line " .. line_number + 1 .. " has too few columns.\\\\")
                end
                skipped_lines = skipped_lines + 1
            elseif values[xcol] == "" or values[ycol] == "" then
                if options and options["debug"] then
                    log_error("\\textbf{Warning:} Empty cell at line " .. line_number + 1 .. " (column " .. xcol .. " or " .. ycol .. ")\\\\")
                end
                skipped_lines = skipped_lines + 1
            else
                local x_val = tonumber(values[xcol])
                local y_val = tonumber(values[ycol])
                if x_val and y_val then
                    table.insert(data, {x_val, y_val})
                else
                    if options and options["debug"] then
                        log_error("\\textbf{Warning:} Invalid data at line " .. line_number + 1 .. " (" .. values[xcol] .. ", " .. values[ycol] .. ")\\\\")
                    end
                    skipped_lines = skipped_lines + 1
                end
            end
        end
        line_number = line_number + 1
    end

    file:close()

    if #data == 0 then
        if options and options["debug"] then
            log_error("\\textbf{Error:} No valid data points found in file.")
        end
        return nil
    end

    if options and options["debug"] then
        log_error("\\textbf{Info:} Read " .. #data .. " data points, skipped " .. skipped_lines .. " lines.\\\\")
    end

    return data
end

-- Approximate significant figures
function sigFig(value, digits)
    if not digits or digits < 1 then return value end
    local fmt = string.format("%%.%dg", digits)
    return tonumber(string.format(fmt, value))
end

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

    -- Improved matrix inversion with partial pivoting
    local function improved_invert_matrix(m)
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

    -- Use improved matrix inversion
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
    
    -- Transform coefficients back to original scale
    local coeff = denormalize_coefficients(norm_coeff, x_min, x_range)
    return coeff
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
            coeff[idx] = coeff[idx] + norm_coeff[i] * bin_coeff * math.pow(-x_min/x_range, power-j) * math.pow(1/x_range, j)
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
    return string.format("{%s}", table.concat(terms, " + "))
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
                table.insert(terms, string.format("%.4fx", coeff)) -- Linear term
            else
                table.insert(terms, string.format("%.4fx^%d", coeff, power)) -- Higher-order terms
            end
        end
    end

    -- Concatenate terms into a single string formatted for pgfplots
    local equation = table.concat(terms, " + ")

    -- Replace "+ -" with "-" for negative terms
    equation = equation:gsub("%+ %-", "- ")

    return string.format("{%s}", equation)
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

-- Approximate significant figures
function sigFig(value, digits)
    if not digits or digits < 1 then return value end
    local fmt = string.format("%%.%dg", digits)
    return tonumber(string.format(fmt, value))
end

-- Example outlier filter using Z-scores
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

-- Log an error message to the LaTeX log file
function log_error(message)
    texio.write("log", "Lua Error: " .. message .. "\n")
end

-- Resamples data, does polynomial regression, and computes lower/upper bands
function generate_confidence_band(data, order, n_bootstrap, conf_level)
    -- Sort original data by x
    table.sort(data, function(a, b) return a[1] < b[1] end)
    local xs = {}
    for i, pt in ipairs(data) do
        xs[i] = pt[1]
    end

    local alpha = (1 - conf_level)/2
    n_bootstrap = n_bootstrap or 1000

    -- For storing predicted values from each bootstrap
    local predictions = {}
    for i = 1, #data do
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

        local coeff = polynomial_regression(sample, order)
        if coeff then
            for i, xval in ipairs(xs) do
                local pred = 0
                for cidx, cval in ipairs(coeff) do
                    local power = (#coeff - cidx)
                    pred = pred + cval * (xval ^ power)
                end
                table.insert(predictions[i], pred)
            end
        end
    end

    -- For each x, sort predictions and pick lower/upper percentiles
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

-- Function to process data and handle any level of polynomial regression
function process_data_with_options(file_path, options)
    local xcol = options["xcol"]
    local ycol = options["ycol"]
    local z_threshold = options["z_threshold"]
    local sig_figures = options["sig_figures"]
    local order = options["order"] or 1 -- Default to linear regression

    -- Read data from the CSV file
    local data = read_csv_to_table(file_path, xcol, ycol, options)
    if not data then
        print("Error: Unable to read data from file.")
        return
    end

    -- Apply Z-score filtering only if z_threshold is set
    local filtered_data = data
    if z_threshold then
        filtered_data = filter_outliers(data, z_threshold)
    end

    -- Perform polynomial regression
    local coefficients = polynomial_regression(filtered_data, order)
    if not coefficients then
        print("Error: Unable to calculate polynomial regression.")
        return
    end

    -- Generate confidence bands if the option is enabled
    local lower_band, upper_band = "", ""
    if options["ci"] then
        local n_bootstrap = options["bootstrap_samples"] or 1000
        local conf_level = 0.95
        lower_band, upper_band = generate_confidence_band(filtered_data, order, n_bootstrap, conf_level)
        tex.sprint("\\def\\qlwr{" .. lower_band .. "}")
        tex.sprint("\\def\\qupr{" .. upper_band .. "}")
    end

    -- Generate polynomial equation for plotting
    local poly_eq = generate_polynomial_equation(coefficients)
    local print_eq = format_equation_plot(coefficients)
    print("Generated Polynomial Equation: " .. poly_eq) -- Debug print

    -- Calculate R-squared for the polynomial regression
    local predict_function = function(x)
        local y_pred = 0
        for i, coeff in ipairs(coefficients) do
            local power = #coefficients - i
            y_pred = y_pred + coeff * x^power
        end
        return y_pred
    end
    
    local poly_r = calculate_r_squared(filtered_data, predict_function)
    poly_r = sigFig(poly_r, sig_figures)
    print("Calculated R-squared: " .. poly_r) -- Debug print

    -- Set LaTeX macros
    token.set_macro("polyR", poly_r)
    tex.sprint("\\def\\polyeq{" .. poly_eq .. "}")
    tex.sprint("\\def\\printeq{" .. print_eq .. "}")
end