# Store your Qualcomm AI Hub API token without echoing it to the screen or shell history.
# Token: https://app.aihub.qualcomm.com -> Settings -> API Token
# Usage (from the repo root):  & .\scripts\configure_aihub.ps1
$exe = Join-Path $PSScriptRoot "..\.venv\Scripts\qai-hub.exe"
$secure = Read-Host -AsSecureString "Paste your AI Hub API token (input hidden), then press Enter"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    & $exe configure --api_token $token --no-verbose 2>$null
    if ($LASTEXITCODE -ne 0) { & $exe configure --api_token $token | Out-Null }
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    Remove-Variable token -ErrorAction SilentlyContinue
}
if (Test-Path "$env:USERPROFILE\.qai_hub\client.ini") {
    Write-Host "AI Hub token saved to $env:USERPROFILE\.qai_hub\client.ini. You can close this tab and tell Claude 'done'." -ForegroundColor Green
} else {
    Write-Host "Something went wrong: no client.ini was written. Try again." -ForegroundColor Red
}
