#!/usr/bin/env bash
# ==============================================================================
# package-ctan.sh
# 
# Prepares distribution archives for 'lua-regression':
#   1. lua-regression.zip     - CTAN upload package (clean single root folder)
#   2. lua-regression.tds.zip - TeX Directory Structure (TDS) package
#   3. lua-regression-doc.pdf - Standalone documentation PDF for release asset
#
# Can be run locally or within GitHub Actions.
# ==============================================================================

set -euo pipefail

PACKAGE_NAME="lua-regression"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
BUILD_DIR="${REPO_ROOT}/build"

echo "======================================================================"
echo " Preparing CTAN and TDS archives for '${PACKAGE_NAME}'"
echo "======================================================================"
echo "Repository root: ${REPO_ROOT}"
echo "Output directory: ${DIST_DIR}"

# 1. Validation of required source files
echo ""
echo "--> [1/5] Verifying source files exist..."

REQUIRED_FILES=(
    "lua-regression/tex/lua-regression.sty"
    "lua-regression/tex/lua-regression.lua"
    "lua-regression/doc/lua-regression-doc.pdf"
    "lua-regression/doc/lua-regression-doc.tex"
    "lua-regression/doc/lua-regression-bib.bib"
    "lua-regression/doc/example/lua-regression-example.tex"
    "lua-regression/doc/example/mpg.csv"
    "lua-regression/README.md"
    "LICENSE.md"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${REPO_ROOT}/${file}" ]; then
        echo "  [ERROR] Missing required file: ${file}" >&2
        MISSING=$((MISSING + 1))
    else
        echo "  [OK] Found ${file}"
    fi
done

if [ ${MISSING} -gt 0 ]; then
    echo "Aborting: ${MISSING} required file(s) are missing." >&2
    exit 1
fi

# Reset clean output and build directories
rm -rf "${DIST_DIR}" "${BUILD_DIR}"
mkdir -p "${DIST_DIR}"
mkdir -p "${BUILD_DIR}/ctan/${PACKAGE_NAME}"
mkdir -p "${BUILD_DIR}/tds/tex/lualatex/${PACKAGE_NAME}"
mkdir -p "${BUILD_DIR}/tds/doc/lualatex/${PACKAGE_NAME}"

# 2. Build CTAN Package Structure
# CTAN requirement: Archive must extract to a single top-level folder matching
# the package name (lua-regression/), containing README, LICENSE, doc/, and tex/.
echo ""
echo "--> [2/5] Preparing CTAN archive structure (${PACKAGE_NAME}.zip)..."

cp -r "${REPO_ROOT}/lua-regression/"* "${BUILD_DIR}/ctan/${PACKAGE_NAME}/"
cp "${REPO_ROOT}/LICENSE.md" "${BUILD_DIR}/ctan/${PACKAGE_NAME}/"

# Clean out any temporary files, TeX auxiliary files, or OS-specific files
find "${BUILD_DIR}/ctan" -type f \( \
    -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" \
    -o -name "*.bbl" -o -name "*.bcf" -o -name "*.blg" -o -name "*.run.xml" \
    -o -name "*.synctex.gz" -o -name "*.fls" -o -name "*.fdb_latexmk" \
    -o -name ".DS_Store" -o -name "Thumbs.db" \
\) -delete

(
    cd "${BUILD_DIR}/ctan"
    zip -r -q -9 "${DIST_DIR}/${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
)
echo "  [OK] Created ${DIST_DIR}/${PACKAGE_NAME}.zip"

# 3. Build TDS Package Structure
# TDS allows users to unzip directly into their local texmf tree:
#   tex/lualatex/lua-regression/
#   doc/lualatex/lua-regression/
echo ""
echo "--> [3/5] Preparing TDS archive structure (${PACKAGE_NAME}.tds.zip)..."

cp "${REPO_ROOT}/lua-regression/tex/lua-regression.sty" "${BUILD_DIR}/tds/tex/lualatex/${PACKAGE_NAME}/"
cp "${REPO_ROOT}/lua-regression/tex/lua-regression.lua" "${BUILD_DIR}/tds/tex/lualatex/${PACKAGE_NAME}/"

cp "${REPO_ROOT}/LICENSE.md" "${BUILD_DIR}/tds/doc/lualatex/${PACKAGE_NAME}/"
cp "${REPO_ROOT}/lua-regression/README.md" "${BUILD_DIR}/tds/doc/lualatex/${PACKAGE_NAME}/"
cp -r "${REPO_ROOT}/lua-regression/doc/"* "${BUILD_DIR}/tds/doc/lualatex/${PACKAGE_NAME}/"

find "${BUILD_DIR}/tds" -type f \( \
    -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" \
    -o -name "*.bbl" -o -name "*.bcf" -o -name "*.blg" -o -name "*.run.xml" \
    -o -name "*.synctex.gz" -o -name "*.fls" -o -name "*.fdb_latexmk" \
    -o -name ".DS_Store" -o -name "Thumbs.db" \
\) -delete

(
    cd "${BUILD_DIR}/tds"
    zip -r -q -9 "${DIST_DIR}/${PACKAGE_NAME}.tds.zip" .
)
echo "  [OK] Created ${DIST_DIR}/${PACKAGE_NAME}.tds.zip"

# 4. Standalone Documentation PDF
echo ""
echo "--> [4/5] Copying standalone documentation PDF..."
cp "${REPO_ROOT}/lua-regression/doc/lua-regression-doc.pdf" "${DIST_DIR}/${PACKAGE_NAME}-doc.pdf"
echo "  [OK] Created ${DIST_DIR}/${PACKAGE_NAME}-doc.pdf"

# Clean up temporary staging directory
rm -rf "${BUILD_DIR}"

# 5. Archive Validation
echo ""
echo "--> [5/5] Validating archive contents against CTAN requirements..."

CTAN_ZIP_ENTRIES=$(unzip -Z1 "${DIST_DIR}/${PACKAGE_NAME}.zip")

# Validate top-level folder rule: every entry in lua-regression.zip must start with "lua-regression/"
BAD_ENTRIES=$(echo "${CTAN_ZIP_ENTRIES}" | grep -v "^${PACKAGE_NAME}/" || true)
if [ -n "${BAD_ENTRIES}" ]; then
    echo "  [ERROR] Found entries in CTAN zip not starting with '${PACKAGE_NAME}/':" >&2
    echo "${BAD_ENTRIES}" >&2
    exit 1
fi

# Validate key files inside CTAN archive
for expected in \
    "${PACKAGE_NAME}/README.md" \
    "${PACKAGE_NAME}/LICENSE.md" \
    "${PACKAGE_NAME}/tex/lua-regression.sty" \
    "${PACKAGE_NAME}/tex/lua-regression.lua" \
    "${PACKAGE_NAME}/doc/lua-regression-doc.pdf" \
    "${PACKAGE_NAME}/doc/lua-regression-doc.tex" \
    "${PACKAGE_NAME}/doc/lua-regression-bib.bib" \
    "${PACKAGE_NAME}/doc/example/lua-regression-example.tex" \
    "${PACKAGE_NAME}/doc/example/mpg.csv"
do
    if ! echo "${CTAN_ZIP_ENTRIES}" | grep -qx "${expected}"; then
        echo "  [ERROR] Missing expected file in CTAN zip: ${expected}" >&2
        exit 1
    fi
done

# Validate no forbidden files
FORBIDDEN=$(echo "${CTAN_ZIP_ENTRIES}" | grep -E '\.(aux|log|out|toc|bbl|bcf|blg|run\.xml|synctex\.gz|fls|fdb_latexmk)$|\.DS_Store|Thumbs\.db' || true)
if [ -n "${FORBIDDEN}" ]; then
    echo "  [ERROR] Forbidden auxiliary files found in CTAN zip:" >&2
    echo "${FORBIDDEN}" >&2
    exit 1
fi

echo "  [OK] CTAN archive verification passed."
echo ""
echo "======================================================================"
echo " Summary of Generated Artifacts:"
echo "======================================================================"
ls -lh "${DIST_DIR}"
echo ""
echo "CTAN zip structure:"
unzip -l "${DIST_DIR}/${PACKAGE_NAME}.zip"
echo ""
echo "All archives ready in: ${DIST_DIR}"
