param(
  [string]$BaseUrl = 'http://127.0.0.1:8000',
  [int]$Timeout = 60
)

Write-Host "Esperando servicio en $BaseUrl/health (timeout ${Timeout}s)..."
$interval = 2
$elapsed = 0
$available = $false
while ($elapsed -lt $Timeout) {
  try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -ErrorAction Stop
    if ($r.status) { Write-Host 'Servicio disponible'; $available = $true; break }
  } catch {
    Start-Sleep -Seconds $interval
    $elapsed += $interval
  }
}
if (-not $available) {
  Write-Host "ERROR: el servicio no respondió en $BaseUrl/health dentro de $Timeout segundos"
  Write-Host "Salida de 'docker compose ps':"
  docker compose ps
  Write-Host "Últimas 200 líneas de logs:"
  docker compose logs --tail 200
  exit 2
}

$paths = @('/health','/reports/detail?rows=1','/reports/season','/model/version','/plots')
foreach ($p in $paths) {
  Write-Host "GET $BaseUrl$p"
  try {
    $out = Invoke-RestMethod -Uri "$BaseUrl$p" -Method Get -ErrorAction Stop
    $out | ConvertTo-Json -Depth 6
  } catch {
    Write-Host "Request failed: $_"
  }
  Write-Host ""
}
Write-Host "Pruebas completadas"
