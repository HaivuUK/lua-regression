# LuaLaTeX Regression

A system for performing regression testing on CSV format data files in LaTeX, written purely in Lua for use with LuaLaTeX.
Includes the ability to fit polynomials of multiple orders to data, and plot the results using PGFPlots.
Offers some additional features such as the ability to generate confidence intervals and perform $R^2$ tests on the data.

## Features
- Fit polynomials of multiple orders to data
- Generate confidence intervals
- Perform $R^2$ tests on the data
- Plot the results using PGFPlots
- Support for CSV format data files
- Written purely in Lua for use with LuaLaTeX
- Control significant figures in the output
- Add and remove equation and $R^2$ from the legend.
- Outputs equations and $R^2$ values to LaTeX commands so they can be called in the document.

## Installation

This package can be installed with a local LaTeX distribution (e.g. TeX Live, MikTeX). (Once approved, it will be available on CTAN.)

Otherwise, you can clone the repository or download from 
[releases](https://github.com/HaivuUK/lua-regression/releases/latest)
and place the `tex` folder in your `texmf` local LaTeX directory.

## Documentation

Relevant documentation is included with each release.

## Authors

This package is authored and maintained by George Allison

## Bug Reporting

If you find a bug, please report it to the issue tracker on GitHub. 
Include a description of the bug, the version of LuaLaTeX you are using, and any relevant code or data files.

## License

This project is licensed under the LPPL (LaTeX Project Public License), version 1.3c or later.