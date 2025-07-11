# --- Configurable paths ---
$csvPath = "C:\Users\AHMED\Desktop\new-approch\ConvertedCSVs\EVTX-ATT&CK-Content-Summary.csv"
$evtxRootDir = "C:\Users\AHMED\Desktop\new-approch\EVTX-ATTACK-SAMPLES"
$outputPath = "$env:USERPROFILE\Desktop\unmapped_evtx_files.txt"

# --- Step 1: Load mapped filenames from CSV ---
$mappedFilenames = @()
Import-Csv -Path $csvPath -Delimiter ';' | ForEach-Object {
    $names = $_.files -split ','  # in case it's multiple filenames in one cell
    foreach ($name in $names) {
        if ($name -and $name -like "*.evtx") {
            $mappedFilenames += $name.Trim()
        }
    }
}

$mappedSet = @{}
$mappedFilenames | ForEach-Object { $mappedSet[$_] = $true }

# --- Step 2: Get all EVTX files ---
$allEvtxFiles = Get-ChildItem -Path $evtxRootDir -Recurse -Filter *.evtx
$totalFiles = $allEvtxFiles.Count

# --- Step 3: Determine unmapped files ---
$unmappedFiles = $allEvtxFiles | Where-Object { -not $mappedSet.ContainsKey($_.Name) }
$unmappedNames = $unmappedFiles | Select-Object -ExpandProperty Name
$totalUnmapped = $unmappedNames.Count
$totalMapped = $totalFiles - $totalUnmapped

# --- Step 4: Count files per folder ---
$folderCounts = $allEvtxFiles | Group-Object { $_.Directory.FullName } | Sort-Object Name

# --- Step 5: Write unmapped files grouped by folder ---
$groupedUnmapped = $unmappedFiles | Group-Object { $_.Directory.FullName } | Sort-Object Name

# Clear existing content (if any)
Clear-Content -Path $outputPath -ErrorAction SilentlyContinue

# Write grouped unmapped files
foreach ($group in $groupedUnmapped) {
    Add-Content -Path $outputPath -Value "Folder: $($group.Name)"
    foreach ($file in $group.Group) {
        Add-Content -Path $outputPath -Value "  $($file.Name)"
    }
    Add-Content -Path $outputPath -Value ""
}

# Write summary
Add-Content -Path $outputPath -Value "`n--- SUMMARY ---"
Add-Content -Path $outputPath -Value "Total EVTX files: $totalFiles"
Add-Content -Path $outputPath -Value "Mapped files: $totalMapped"
Add-Content -Path $outputPath -Value "Unmapped files: $totalUnmapped"

Add-Content -Path $outputPath -Value "`nFiles per folder:"
foreach ($group in $folderCounts) {
    $folder = $group.Name
    $count = $group.Count
    Add-Content -Path $outputPath -Value "$folder : $count"
}

Write-Host "✅ Done! Unmapped files grouped by folder and summary saved to: $outputPath"
