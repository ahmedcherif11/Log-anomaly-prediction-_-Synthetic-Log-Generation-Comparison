# Excel file path
$excelPath = "C:\Users\AHMED\Desktop\new-approch\ConvertedCSVs\EVTX-ATT&CK-Content-Summary.xlsx"
$evtxRootDir = "C:\Users\AHMED\Desktop\new-approch\EVTX-ATTACK-SAMPLES"

# Launch Excel and open workbook
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$workbook = $excel.Workbooks.Open($excelPath)
$sheet = $workbook.Sheets.Item(1)  # Use the first sheet

# Output log
$log = @()

# Start reading from row 2 (assuming headers are in row 1)
$row = 2
while ($true) {
    $cell = $sheet.Cells.Item($row, 4)  # Column D (4th column)
    $hyperlink = $null
    try {
        $hyperlink = $cell.Hyperlinks.Item(1).Address
    } catch {}

    if (-not $cell.Text -and -not $hyperlink) {
        break  # Exit loop on empty row
    }

    if ($hyperlink) {
        # Extract clean filename from hyperlink
        $filename = [System.IO.Path]::GetFileName($hyperlink)
        $cleanName = $filename -replace "%40","@"

        # Try to find the old file name from cell text
        $originalName = $cell.Text.Trim()
        $evtxFile = Get-ChildItem -Path $evtxRootDir -Recurse -Filter $originalName -ErrorAction SilentlyContinue

        if ($evtxFile) {
            
            $newPath = Join-Path $evtxFile.Directory.FullName $cleanName
            Rename-Item -Path $evtxFile.FullName -NewName $cleanName -Force
            Write-Host "Renamed '$originalName' → '$cleanName'"
            $log += [PSCustomObject]@{ Old = $originalName; New = $cleanName }
        } else {
            Write-Warning "File not found: $originalName"
            $log += [PSCustomObject]@{ Old = $originalName; New = "NOT FOUND" }
        }
    }

    $row++
}

# Close Excel
$workbook.Close($false)
$excel.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null

# Export log to CSV
$log | Export-Csv -Path "$env:USERPROFILE\Desktop\evtx_renaming_log.csv" -NoTypeInformation
Write-Host "Log saved to Desktop as 'evtx_renaming_log.csv'"