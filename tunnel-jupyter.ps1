<#
.SYNOPSIS
  Opens an SSH tunnel from this Windows PC to JupyterLab running inside the
  AllianceAI Docker container on the remote Alpine server, so you can reach it
  locally at http://localhost:8888/lab - without exposing port 8888 to the
  internet.

.DESCRIPTION
  This is the "tunnel" from DEPLOY.md, recreated as a reusable script:
      ssh -N -L 8888:localhost:8888 user@SERVER_IP
  It forwards your local port 8888 to the server's port 8888 (where the
  container publishes JupyterLab). Keep the window open while you work.

.PARAMETER Server
  The SSH target, e.g. "arman@203.0.113.10" or an alias from ~/.ssh/config.
  Defaults to the ALLIANCEAI_SERVER environment variable if set.

.PARAMETER StartContainer
  Before tunneling, SSH in and run `docker compose up -d` so the container is
  guaranteed to be running.

.PARAMETER Open
  Open http://localhost:8888/lab in your default browser once the tunnel is up.

.EXAMPLE
  .\tunnel-jupyter.ps1 -Server arman@203.0.113.10

.EXAMPLE
  .\tunnel-jupyter.ps1 -Server myalpine -StartContainer -Open

.NOTES
  Set the server once so you never retype it (then open a NEW terminal):
      setx ALLIANCEAI_SERVER "user@SERVER_IP"
  After that just run:  .\tunnel-jupyter.ps1
#>
param(
    [string]$Server = $env:ALLIANCEAI_SERVER,
    [int]$LocalPort  = 8888,
    [int]$RemotePort = 8888,
    [string]$RemoteDir = "~/allianceai",
    [switch]$StartContainer,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Server)) {
    Write-Host "No server specified." -ForegroundColor Red
    Write-Host "  Usage:        .\tunnel-jupyter.ps1 -Server user@SERVER_IP" -ForegroundColor Yellow
    Write-Host "  Set default:  setx ALLIANCEAI_SERVER ""user@SERVER_IP""  (then reopen the terminal)" -ForegroundColor Yellow
    exit 1
}

# The OpenSSH client ships with Windows 10/11; make sure it's available.
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "OpenSSH client not found." -ForegroundColor Red
    Write-Host "  Install it: Settings > Apps > Optional Features > Add a feature > OpenSSH Client" -ForegroundColor Yellow
    exit 1
}

# A stale tunnel (or local Jupyter) may already hold the port.
$busy = $false
try {
    $busy = Test-NetConnection -ComputerName "localhost" -Port $LocalPort -InformationLevel Quiet -WarningAction SilentlyContinue
} catch { $busy = $false }
if ($busy) {
    Write-Host "Local port $LocalPort is already in use." -ForegroundColor Yellow
    Write-Host "An existing tunnel may already be open - try http://localhost:$LocalPort/lab," -ForegroundColor Yellow
    Write-Host "or pass a different -LocalPort (e.g. -LocalPort 8889)." -ForegroundColor Yellow
}

if ($StartContainer) {
    Write-Host "Ensuring the AllianceAI container is running on $Server ..." -ForegroundColor Cyan
    # Build the remote command via concatenation so the POSIX '&&' is a literal
    # passed to the server's shell, never parsed by Windows PowerShell.
    $remoteCmd = "cd " + $RemoteDir + " " + [char]0x26 + [char]0x26 + " docker compose up -d"
    ssh $Server $remoteCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Could not start the container (check the path '$RemoteDir' on the server)." -ForegroundColor Red
        exit 1
    }
}

if ($Open) {
    Start-Process "http://localhost:$LocalPort/lab"
}

Write-Host ""
Write-Host "Tunnel : localhost:$LocalPort  ->  ${Server} : $RemotePort  (JupyterLab in Docker)" -ForegroundColor Green
Write-Host "Open   : http://localhost:$LocalPort/lab" -ForegroundColor Green
Write-Host "Token  : the JUPYTER_TOKEN from the server's .env (default 'allianceai' if unset)" -ForegroundColor Green
Write-Host "Stop   : keep this window open; press Ctrl-C to close the tunnel." -ForegroundColor DarkGray
Write-Host ""

# -N          : do not run a remote command, just forward the port
# -L L:host:R : forward local $LocalPort to the server's $RemotePort
# ServerAlive*: send keepalives so the tunnel doesn't silently die when idle
$sshArgs = @(
    "-N",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=3",
    "-o", "ExitOnForwardFailure=yes",
    "-L", "${LocalPort}:localhost:${RemotePort}",
    $Server
)
ssh @sshArgs
