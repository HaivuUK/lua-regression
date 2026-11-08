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

-- Log an error message to the LaTeX log file
function log_error(message)
    texio.write("log", "Pacakge lua-regression error: " .. message .. "\n")
end

-- Report error
function report_error(message)
    texio.write("term and log", "Package lua-regression error: " .. message .. "\n")
    tex.sprint("\\PackageError{lua-regression}{Lua regression error: " .. message .. "}{Please check your data and options.}")
end

-- Public rregression and latex macro output pipeline
function process_data_with_options(file_path, options)
    tex.sprint("\\global\\expandafter\\def\\csname luareg@polyR\\endcsname{\\csname luareg@missingresult\\endcsname\\polyR}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@polyeq\\endcsname{\\csname luareg@missingresult\\endcsname\\polyeq}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@printeq\\endcsname{\\csname luareg@missingresult\\endcsname\\printeq}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@plot\\endcsname{\\csname luareg@missingresult\\endcsname\\polyeq}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@qlwr\\endcsname{}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@qupr\\endcsname{}")
    if options and options["seed"] then
        math.randomseed(options["seed"])
    else
        math.randomseed(os.time())
    end
    local x = options["x"]
    local y = options["y"]
    local z_threshold = options["z_threshold"]
    local sig_figures = options["sig_figures"]
    local order = options["order"] or 1 -- Default to linear regression if order is not specified

    local data
    if type(file_path) == "table" then
        data = {}
        local xmin = (options and options["xmin"]) or -1e99
        local xmax = (options and options["xmax"]) or 1e99
        local lines_filter = (options and options["lines"] and options["lines"] ~= "") and parse_line_option(options["lines"]) or nil
        for row_idx, point in ipairs(file_path) do
            if not lines_filter or lines_filter[row_idx] then
                local x_val = point and tonumber(point[1])
                local y_val = point and tonumber(point[2])
                local x_finite = x_val and x_val == x_val and math.abs(x_val) < math.huge
                local y_finite = y_val and y_val == y_val and math.abs(y_val) < math.huge
                if x_finite and y_finite and x_val >= xmin and x_val <= xmax then
                    data[#data + 1] = {x_val, y_val}
                end
            end
        end
        if #data == 0 then
            report_error("No valid data points found in table input within the specified range.")
            return
        end
    else
        -- Read CSV data from file
        data = read_csv_to_table(file_path, x, y, options)
        if not data then
            report_error("Failed to read data from file: " .. file_path)
            return
        end
    end

    -- Z-score filter if the z threshold is set
    local filtered_data = data
    if z_threshold then
        filtered_data = filter_outliers(data, z_threshold)
    end

    if options["ignore_zero"] then
        local nonzero = {}
        for _, point in ipairs(filtered_data) do
            if point[1] ~= 0 and point[2] ~= 0 then nonzero[#nonzero + 1] = point end
        end
        filtered_data = nonzero
    end

    local raw_type = options["regression_type"] or options["regression type"] or options["type"] or "polynomial"
    local model_type = string.lower(tostring(raw_type):gsub("%s+", ""))
    if model_type == "power" or  model_type == "power-series" or model_type == "powerseries" or model_type == "logarithmic" or model_type == "log" then
        local domain_data = {}
        for _, point in ipairs(filtered_data) do 
            if point[1] > 0 and (model_type == "logarithmic" or model_type == "log" or point[2] > 0) then
                domain_data[#domain_data+1] = point
            end
        end
        filtered_data = domain_data
    elseif model_type == "exponential" or model_type == "exp" then
        local domain_data = {}
        for _, point in ipairs(filtered_data) do
            if point[2] > 0 then
                domain_data[#domain_data+1] = point
            end
        end
        filtered_data = domain_data
    elseif model_type == "poisson" then
        local count_data = {}
        for _, point in ipairs(filtered_data) do
            if point[2] >= 0 then count_data[#count_data+1] = point end
        end
        filtered_data = count_data
    end
    if #filtered_data < order + 1 then
        report_error("Not enough data points after filtering")
        return
    end
    local model = fit_model(filtered_data, options)
    if not model then
        report_error("Unable to calculate regression model")
        return
    end
    local coefficients = model.coeffs or model.coefficients

    -- Calculate p value for coefficients and overall model
    local pvals = model.pvalues
    local overall_p = model.overall_p
    if model_type == "polynomial" and not pvals then
        _, pvals = regression_pvalues(filtered_data, coefficients, order)
        overall_p = regression_overall_pvalue(filtered_data, coefficients, order)
    end
    local function format_pval(p)
        if not p then return "" end
        if p <= 0 then
            return "0"
        elseif p < 1e-4 then
            local e = math.floor(math.log(p) / math.log(10))
            local m = p / (10 ^ e)
            if m >= 9.9995 then m = 1; e = e + 1 end
            return string.format("%.2f \\times 10^{%d}", m, e)
        end
        return string.format("%.4f", p)
    end
    local pval_str = ""
    if pvals then
        local parts = {}
        for i, p in ipairs(pvals) do
            parts[#parts + 1] = format_pval(p)
        end
        pval_str = table.concat(parts, ", ")
    end
    local main_pval_str = format_pval(overall_p)
    tex.sprint("\\def\\polypvalues{" .. pval_str .. "}") -- all coefficient p-values
    tex.sprint("\\def\\polyoverallp{" .. main_pval_str .. "}") -- main overall regression p-value

    -- Generate confidence bands if the option is enabled
    local lower_band, upper_band = "", ""
    if options["ci"] then
        local n_bootstrap = options["bootstrap_samples"] or 1000
        local conf_level = options["ci_level"] or 0.95
        if conf_level >= 1 and conf_level < 100 then conf_level = conf_level / 100 end -- allow 95 for 95%
        if not (conf_level > 0 and conf_level < 1) then
            report_error("ci level must be between 0 and 1")
            return
        end
        local ci_truncate = options["truncate"] ~= false -- default true
        local ci_xmin, ci_xmax
        if not ci_truncate then
            -- Determine CI bounds when truncate is false either explicitly (ci max/min)
            -- or via PGFPlots axis limits (axis_xmin / axis_xmax) or explicit xmin/xmax
            -- or data range in that order.
            ci_xmin = options["ci_min"] or options["cimin"] or options["axis_xmin"]
            ci_xmax = options["ci_max"] or options["cimax"] or options["axis_xmax"]

            if (not ci_xmin or math.abs(ci_xmin) >= 1e90) and options["xmin"] and math.abs(options["xmin"]) < 1e90 then
                ci_xmin = options["xmin"]
            end
            if (not ci_xmax or math.abs(ci_xmax) >= 1e90) and options["xmax"] and math.abs(options["xmax"]) < 1e90 then
                ci_xmax = options["xmax"]
            end

            table.sort(filtered_data, function(a, b) return a[1] < b[1] end)
            if not ci_xmin or math.abs(ci_xmin) >= 1e90 then
                ci_xmin = filtered_data[1] and filtered_data[1][1] or 0
            end
            if not ci_xmax or math.abs(ci_xmax) >= 1e90 then
                ci_xmax = filtered_data[#filtered_data] and filtered_data[#filtered_data][1] or 1
            end
            if ci_xmin > ci_xmax then
                ci_xmin, ci_xmax = ci_xmax, ci_xmin
            end
            if (model_type == "logarithmic" or model_type == "log" or model_type == "power" or model_type == "powerseries" or model_type == "power-series") and ci_xmin <= 0 then
                ci_xmin = math.max(0.01, filtered_data[1] and filtered_data[1][1] or 0.1)
            end
        end
        lower_band, upper_band = generate_confidence_band(filtered_data, order, n_bootstrap, conf_level, ci_truncate, ci_xmin, ci_xmax, options)
        tex.sprint("\\global\\expandafter\\def\\csname luareg@qlwr\\endcsname{" .. lower_band .. "}")
        tex.sprint("\\global\\expandafter\\def\\csname luareg@qupr\\endcsname{" .. upper_band .. "}")
    end

    local poly_eq = model.display_equation and model.display_equation() or model.equation()
    local plot_eq = model.plot_equation()
    local print_eq = poly_eq
    local poly_r = model.r_squared
    if poly_r == nil then
        poly_r = calculate_r_squared(filtered_data, model.predict)
    end
    poly_r = poly_r and sigFig(poly_r, sig_figures) or "N/A"

    -- Set latex macros
    tex.sprint("\\global\\expandafter\\def\\csname luareg@polyR\\endcsname{" .. poly_r .. "}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@polyeq\\endcsname{" .. poly_eq .. "}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@printeq\\endcsname{" .. print_eq .. "}")
    tex.sprint("\\global\\expandafter\\def\\csname luareg@plot\\endcsname{" .. plot_eq .. "}")
end