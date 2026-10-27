$secureToken = Read-Host 'Label Studio API token (input hidden)' -AsSecureString
$env:LABEL_STUDIO_API_KEY = [Net.NetworkCredential]::new('', $secureToken).Password
Remove-Variable secureToken
. "$PSScriptRoot\Environment.ps1"
Write-Host 'Label Studio credential set for this session. Local control credential loaded securely.'
