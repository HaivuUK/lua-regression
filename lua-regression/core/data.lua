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

-- Parse lines options (e.g., "2,3,7-9")
function parse_line_option(lines_str)
    if not lines_str or lines_str == "" then
        return nil
    end
    local lines = {}
    for part in string.gmatch(lines_str, "[^,]+") do
        local range_start, range_end = string.match(part, "(%d+)%s*%-%s*(%d+)$")
        if range_start and range_end then
            for i = tonumber(range_start), tonumber(range_end) do
                lines[i] = true
            end
        else
            local idx = tonumber(part)
            if idx then lines[idx] = true end
        end
    end
    return lines
end

-- Function to read a CSV file and store the data in a table
function read_csv_to_table(file_path, x, y, options)
    local data = {}
    local file = io.open(file_path, "r")
    if not file then
        if options and options["debug"] then
            log_error("\\textbf{Error:} Unable to open file " .. file_path)
        end
        return nil
    end

    -- Parse lines option
    local lines_filter = nil
    if options and options["lines"] and options["lines"] ~= "" then
        lines_filter = parse_line_option(options["lines"])
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

    -- If x / y are names, convert them to column indices
    if type(x) == "string" then
        local idx = tonumber(x) or find_column_index(header, x)
        if not idx then
            if options and options["debug"] then
                log_error("\\textbf{Error:} Column '" .. x .. "' not found in header")
            end
            file:close()
            return nil
        end
        x = idx
    end

    if type(y) == "string" then
        local idx = tonumber(y) or find_column_index(header, y)
        if not idx then
            if options and options["debug"] then
                log_error("\\textbf{Error:} Column '" .. y .. "' not found in header.")
            end
            file:close()
            return nil
        end
        y = idx
    end

    -- Read the data lines
    local line_number = 1
    local skipped_lines = 0
    while true do
        local line = file:read("*l")
        if not line then break end

        line = line:gsub("^%s*(.-)%s*$", "%1") -- Trim
        if line ~= "" then

            -- Process if lines_filter is nil or the line is in the filter
            if not lines_filter or lines_filter[line_number] then
                local values = split(line, ',')

                -- Check if we have enough columns
                if #values < math.max(x, y) then
                    if options and options["debug"] then
                        log_error("\\textbf{Warning:} Line " .. line_number + 1 .. " has too few columns.\\\\")
                    end
                    skipped_lines = skipped_lines + 1
                elseif values[x] == "" or values[y] == "" then
                    if options and options["debug"] then
                        log_error("\\textbf{Warning:} Empty cell at line " .. line_number + 1 .. " (column " .. x .. " or " .. y .. ")\\\\")
                    end
                    skipped_lines = skipped_lines + 1
                else
                    local x_val = tonumber(values[x])
                    local y_val = tonumber(values[y])
                    local xmin = options and options["xmin"] or -1e99
                    local xmax = options and options["xmax"] or 1e99
                    local x_finite = x_val and x_val == x_val and math.abs(x_val) < math.huge -- Check for NaN and Inf
                    local y_finite = y_val and y_val == y_val and math.abs(y_val) < math.huge -- Check for NaN and Inf
                    if x_finite and y_finite and x_val >= xmin and x_val <= xmax then
                        table.insert(data, {x_val, y_val})
                    else
                        if options and options["debug"] then
                            log_error("\\textbf{Warning:} Invalid data at line " .. line_number + 1 .. " (" .. values[x] .. ", " .. values[y] .. ")\\\\")
                        end
                        skipped_lines = skipped_lines + 1
                    end
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