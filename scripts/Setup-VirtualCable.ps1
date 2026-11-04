# Helps to install VB-CABLE (the virtual microphone for Discord and games) without losing your audio devices.
# The VB-CABLE license does not allow bundling its installer into another program's setup, so this script does not
# download or run it: it remembers your current default microphone and output, opens the official page, waits while you
# install it, then checks that the cable appeared and switches your own devices back (the VB-CABLE installer makes the
# cable the default microphone and the default speakers).
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'AudioDefault.ps1')

function Test-Cable { [bool](Get-Mics | Where-Object { $_.Name -like '*CABLE Output*' }) }

$mic = Get-DefaultMic
$out = Get-DefaultSpeaker
if (Test-Cable) {
    Write-Host 'VB-CABLE уже встановлено («CABLE Output» є серед мікрофонів).' -ForegroundColor Green
    if ($mic.Name -like '*CABLE*' -or $out.Name -like '*CABLE*') {
        Write-Host "Але за замовчуванням зараз кабель: мікрофон «$($mic.Name)», звук «$($out.Name)»." -ForegroundColor Yellow
        Write-Host 'Виберіть свої пристрої в налаштуваннях звуку Windows (Параметри → Система → Звук).'
        Start-Process 'ms-settings:sound'
    }
    exit 0
}

Write-Host "Запам'ятав ваші пристрої: мікрофон «$($mic.Name)», звук «$($out.Name)»."
Write-Host ''
Write-Host 'Зараз відкриється офіційна сторінка VB-CABLE (vb-audio.com). Там:'
Write-Host '  1. Завантажте «VBCABLE_Driver_Pack…zip» і розпакуйте його.'
Write-Host '  2. Правий клік на VBCABLE_Setup_x64.exe → «Запуск від імені адміністратора» → Install Driver.'
Write-Host '  3. Дочекайтеся повідомлення про завершення установки й поверніться в це вікно.'
Write-Host 'VB-CABLE — donationware: для особистого використання безкоштовно.'
Start-Process 'https://vb-audio.com/Cable/'
Write-Host ''
Read-Host 'Коли установку завершено, натисніть Enter' | Out-Null

if (-not (Test-Cable)) {
    Write-Host 'Кабелю поки не видно. Якщо установка пройшла, перезавантажте ПК і запустіть цей файл ще раз.' -ForegroundColor Yellow
    exit 1
}
[VT.AudioDevices2]::SetDefault($mic.Id)
[VT.AudioDevices2]::SetDefault($out.Id)
Write-Host "Готово: «CABLE Output» з'явився, а за замовчуванням знову ваші пристрої:" -ForegroundColor Green
Write-Host "  мікрофон «$((Get-DefaultMic).Name)», звук «$((Get-DefaultSpeaker).Name)»."
Write-Host 'VB-Audio радить перезавантажити ПК, щоб завершити установку. Далі: Start-VoiceTranslate.bat — він сам знайде кабель.'
