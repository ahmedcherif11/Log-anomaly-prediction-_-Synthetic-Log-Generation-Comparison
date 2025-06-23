$excel = New-Object -ComObject Excel.Application
$workbook = $excel.Workbooks.Open("C:\Users\AHMED\Desktop\new-approch\ConvertedCSVs\EVTX-ATT&CK-Content-Summary.xlsx")
$sheet = $workbook.Sheets.Item(1)
$cell = $sheet.Cells.Item(4, 4)
$cellText = $cell.Text
$hyperlink = $cell.Hyperlinks.Item(1).Address

Write-Host "Visible text: $cellText"
Write-Host "Hyperlink: $hyperlink"

$excel.Quit()
