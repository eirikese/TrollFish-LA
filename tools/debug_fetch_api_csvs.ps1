param(
  [Parameter(Mandatory=$true)]
  [string]$BaseUrl,

  [Parameter(Mandatory=$true)]
  [string]$ApiKey,

  [string]$FromDate = "2026-02-11",
  [string]$ToDate = "2026-02-13",
  [string]$OutDir = ".\debug_api_csvs",
  [switch]$IncludeRecordings
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Web

function Normalize-ApiBase([string]$url) {
  $u = $url.Trim().TrimEnd("/")
  if ($u.EndsWith("/api/data")) { return $u }
  return "$u/api/data"
}

function Invoke-Api([string]$path, [hashtable]$query = @{}) {
  $root = Normalize-ApiBase $BaseUrl
  $uriBuilder = [System.UriBuilder]::new("$root/$($path.TrimStart('/'))")
  $qs = [System.Web.HttpUtility]::ParseQueryString("")
  foreach ($key in $query.Keys) {
    if ($null -ne $query[$key] -and "$($query[$key])" -ne "") {
      $qs[$key] = "$($query[$key])"
    }
  }
  $uriBuilder.Query = $qs.ToString()
  $res = Invoke-RestMethod -Uri $uriBuilder.Uri.AbsoluteUri -Headers @{ Authorization = "Bearer $ApiKey" }
  if (-not $res.success) {
    throw "API failed: $($res.error) $($res.message)"
  }
  return $res
}

function Csv-Escape($value) {
  if ($null -eq $value) { return "" }
  $s = [string]$value
  if ($s -match '[,"\r\n]') { return '"' + $s.Replace('"', '""') + '"' }
  return $s
}

function Write-RowsCsv($rows, [string]$path) {
  $headers = @(
    "timestamp","unit_id","custom_name","lat","lon","alt","roll","pitch","yaw",
    "sog","cog","hdop","gnss_ms","gnss_iso","rudder_angle","boom_angle","torso_angle","seq"
  )
  $lines = [System.Collections.Generic.List[string]]::new()
  $lines.Add(($headers | ForEach-Object { Csv-Escape $_ }) -join ",")
  foreach ($row in $rows) {
    $vals = foreach ($h in $headers) { Csv-Escape $row.$h }
    $lines.Add($vals -join ",")
  }
  [System.IO.File]::WriteAllLines($path, $lines, [System.Text.UTF8Encoding]::new($false))
}

function Safe-Name([string]$value, [string]$fallback = "item") {
  $s = ""
  if ($null -ne $value) { $s = $value.Trim() }
  if (-not $s) { $s = $fallback }
  $s = $s -replace '[^a-zA-Z0-9._-]+', '_'
  $s = $s -replace '_+', '_'
  $s = $s.Trim("_")
  if (-not $s) { $s = $fallback }
  return $s.Substring(0, [Math]::Min(120, $s.Length))
}

function Is-ContinuousSession($session) {
  $id = [string]$session.id
  $type = ([string]$session.type).ToLowerInvariant()
  $source = ([string]$session.source).ToLowerInvariant()
  return ($id -match '^cont-\d{4}-\d{2}-\d{2}$' -or $type -eq 'legacy' -or $source -eq 'continuous')
}

$fromUtc = [DateTimeOffset]::ParseExact($FromDate, "yyyy-MM-dd", $null).ToUniversalTime()
$toUtc = [DateTimeOffset]::ParseExact($ToDate, "yyyy-MM-dd", $null).ToUniversalTime().AddDays(1).AddMilliseconds(-1)
$fromMs = [int64]$fromUtc.ToUnixTimeMilliseconds()
$toMs = [int64]$toUtc.ToUnixTimeMilliseconds()

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Querying sessions from $($fromUtc.UtcDateTime.ToString('u')) to $($toUtc.UtcDateTime.ToString('u'))"

$sessions = @()
$offset = 0
$limit = 200
do {
  $page = Invoke-Api "/sessions" @{
    from = $fromMs
    to = $toMs
    limit = $limit
    offset = $offset
  }
  $sessions += @($page.data)
  $offset += $limit
} while ($page.meta.has_more)

Write-Host "Found $($sessions.Count) session(s)."

$continuousSessions = @($sessions | Where-Object { Is-ContinuousSession $_ })
if (-not $IncludeRecordings -and $continuousSessions.Count -gt 0) {
  Write-Host "Using $($continuousSessions.Count) continuous session(s); skipping short recordings. Pass -IncludeRecordings to fetch both."
  $sessions = $continuousSessions
}

$fields = "lat,lon,alt,roll,pitch,yaw,sog,cog,hdop,gnss_ms,gnss_iso,rudder_angle,boom_angle,torso_angle,seq,custom_name"
$dayMs = [int64](24 * 60 * 60 * 1000)
$written = 0

foreach ($session in $sessions) {
  $sid = [string]$session.id
  if (-not $sid) { continue }

  $sessionStart = $fromMs
  if ($null -ne $session.start_time) { $sessionStart = [int64]$session.start_time }
  $sessionEnd = $toMs
  if ($null -ne $session.end_time) { $sessionEnd = [int64]$session.end_time }
  $fetchFrom = [Math]::Max($fromMs, $sessionStart)
  $fetchTo = [Math]::Min($toMs, $sessionEnd)
  if ($fetchTo -lt $fetchFrom) { continue }

  Write-Host "Fetching telemetry for $sid..."
  $allRows = @()
  for ($winFrom = $fetchFrom; $winFrom -le $fetchTo; $winFrom += $dayMs) {
    $winTo = [Math]::Min($fetchTo, $winFrom + $dayMs - 1)
    $rowOffset = 0
    $rowLimit = 5000
    do {
      $telemetry = Invoke-Api "/sessions/$sid/telemetry" @{
        from = $winFrom
        to = $winTo
        raw = "true"
        limit = $rowLimit
        offset = $rowOffset
        fields = $fields
      }
      $allRows += @($telemetry.data)
      $rowOffset += $rowLimit
    } while ($telemetry.meta.has_more)
  }

  $byUnit = $allRows | Where-Object { $_.unit_id } | Group-Object unit_id
  foreach ($group in $byUnit) {
    $unit = [string]$group.Name
    $rows = @($group.Group | Sort-Object timestamp)
    if ($rows.Count -eq 0) { continue }
    $athlete = ""
    if ($session.athlete_names -and $session.athlete_names.PSObject.Properties[$unit]) {
      $athlete = [string]$session.athlete_names.PSObject.Properties[$unit].Value
    }
    if (-not $athlete) {
      $named = $rows | Where-Object { $_.custom_name } | Select-Object -First 1
      if ($named) { $athlete = [string]$named.custom_name }
    }
    $startLabel = ([DateTimeOffset]::FromUnixTimeMilliseconds([int64]$rows[0].timestamp)).UtcDateTime.ToString("yyyyMMddTHHmmssZ")
    $name = "TrollSports_debug_$(Safe-Name $athlete 'athlete')_$(Safe-Name $sid 'session')_$(Safe-Name $unit 'unit')_$startLabel.csv"
    $path = Join-Path $OutDir $name
    Write-RowsCsv $rows $path
    $written += 1
    Write-Host "  wrote $path ($($rows.Count) rows)"
  }
}

Write-Host "Done. Wrote $written CSV file(s) to $OutDir"
