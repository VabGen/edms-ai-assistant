# build_licenses.ps1
$env:PYTHONIOENCODING = "utf-8"
pip-licenses --format=markdown --with-license-file --output-file THIRD_PARTY_LICENSES.md
Write-Host "✅ Licenses exported successfully"
