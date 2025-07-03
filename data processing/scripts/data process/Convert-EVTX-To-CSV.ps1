# Path to Eric Zimmerman's EvtxECmd.exe
$evtxeCmdPath = "C:\Users\AHMED\Desktop\new-approch\EvtxeCmd\EvtxECmd.exe"

# Root directory of your EVTX dataset
$rootPath = "C:\Users\AHMED\Desktop\new-approch\EVTX-ATTACK-SAMPLES"

# Output base directory
$outputBase = "C:\Users\AHMED\Desktop\new-approch\ConvertedCSVs"

# Log file path
$logPath = Join-Path $outputBase "summary.txt"

# Create output base directory if it doesn't exist
if (-not (Test-Path $outputBase)) {
    New-Item -Path $outputBase -ItemType Directory | Out-Null
}

# Start time
$startTime = Get-Date

# Tracking variables
$total = 0
$success = 0
$failed = 0
$errors = @()
$logLines = @()
$logLines += "Début d'exécution : $startTime"
$logLines += "--------------------------------------------"

# Get all .evtx files recursively
Get-ChildItem -Path $rootPath -Recurse -Filter *.evtx | ForEach-Object {
    $total++
    $evtxFile = $_.FullName
    $relativeDir = $_.Directory.FullName.Substring($rootPath.Length).TrimStart("\")
    $outputDir = Join-Path $outputBase $relativeDir

    # Create matching output directory
    if (-not (Test-Path $outputDir)) {
        New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
    }

    $csvName = "$($_.BaseName).csv"
    $csvPath = Join-Path $outputDir $csvName

    Write-Host "Processing: $evtxFile → $csvPath"
    $logLines += "Processing: $evtxFile → $csvPath"

    try {
        & $evtxeCmdPath -f "$evtxFile" --csv "$outputDir" --csvf "$csvName"
        if ($LASTEXITCODE -eq 0) {
            $success++
        } else {
            $failed++
            $errors += "Erreur (code $LASTEXITCODE) : $evtxFile"
        }
    } catch {
        $failed++
        $errors += "Exception lors du traitement : $evtxFile → $_"
    }
}

# End time and duration
$endTime = Get-Date
$duration = $endTime - $startTime

# Résumé
$logLines += ""
$logLines += "===== Résumé de l'exécution ====="
$logLines += "Total de fichiers traités : $total"
$logLines += "Succès                    : $success"
$logLines += "Échecs                    : $failed"
$logLines += "Durée totale              : $($duration.ToString())"

if ($failed -gt 0) {
    $logLines += ""
    $logLines += "--- Détails des erreurs ---"
    $errors | ForEach-Object { $logLines += $_ }
}

$logLines += ""
$logLines += "Fin d'exécution : $endTime"

# Afficher dans le terminal
$logLines | ForEach-Object { Write-Host $_ }

# Écrire le résumé dans le fichier log
$logLines | Out-File -FilePath $logPath -Encoding UTF8

Write-Host "`nRésumé écrit dans : $logPath" -ForegroundColor Cyan
