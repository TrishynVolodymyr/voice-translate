# Default recording device of Windows, without third-party tools (Core Audio COM via a small inline C# class).
# Used by the launcher's virtual microphone: while the translator runs, the default microphone is "CABLE Output" (games that cannot
# choose a microphone hear the translated voice); on exit the previous default (your own microphone) is restored.
#   . .\AudioDefault.ps1; Get-Mics; Get-DefaultMic; Set-DefaultMic 'CABLE Output'; Get-Speakers; Set-DefaultSpeaker 'Headphones'
# Changes only the default device (all roles: console, multimedia, communications); nothing is installed.
if (-not ('VT.AudioDevices2' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace VT {
  [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int dataFlow, int stateMask, out IMMDeviceCollection devices);
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice device);
  }
  [ComImport, Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDeviceCollection { int GetCount(out int n); int Item(int i, out IMMDevice device); }
  [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDevice {
    int Activate(ref Guid iid, int ctx, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object o);
    int OpenPropertyStore(int access, out IPropertyStore store);
    int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
  }
  [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IPropertyStore {
    int GetCount(out int n); int GetAt(int i, out PropertyKey key); int GetValue(ref PropertyKey key, out PropVariant v);
  }
  [StructLayout(LayoutKind.Sequential)] struct PropertyKey { public Guid fmtid; public int pid; }
  [StructLayout(LayoutKind.Explicit)] struct PropVariant { [FieldOffset(0)] public short vt; [FieldOffset(8)] public IntPtr p; }
  [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator { }
  // Undocumented but long-stable interface that the Windows sound settings use to change the default device.
  [ComImport, Guid("F8679F50-850A-41CF-9C72-430F290290C8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IPolicyConfig {
    int GetMixFormat(); int GetDeviceFormat(); int ResetDeviceFormat(); int SetDeviceFormat(); int GetProcessingPeriod();
    int SetProcessingPeriod(); int GetShareMode(); int SetShareMode(); int GetPropertyValue(); int SetPropertyValue();
    int SetDefaultEndpoint([MarshalAs(UnmanagedType.LPWStr)] string id, int role);
    int SetEndpointVisibility();
  }
  [ComImport, Guid("870AF99C-171D-4F9E-AF0D-E63DF40C2BC9")] class CPolicyConfigClient { }
  public static class AudioDevices2 {
    static string Name(IMMDevice d) {
      IPropertyStore s; d.OpenPropertyStore(0, out s);
      var k = new PropertyKey { fmtid = new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"), pid = 14 };  // friendly name
      PropVariant v; s.GetValue(ref k, out v);
      return Marshal.PtrToStringUni(v.p);
    }
    // [id, name] of every device; flow 0 = playback, 1 = recording; states 1 = active only, 15 = also unplugged ones
    public static List<string[]> Devices(int flow, int states) {
      var e = (IMMDeviceEnumerator)new MMDeviceEnumerator(); IMMDeviceCollection c; e.EnumAudioEndpoints(flow, states, out c);
      int n; c.GetCount(out n); var r = new List<string[]>();
      for (int i = 0; i < n; i++) { IMMDevice d; c.Item(i, out d); string id; d.GetId(out id); r.Add(new[] { id, Name(d) }); }
      return r;
    }
    public static string[] Default(int flow) {
      var e = (IMMDeviceEnumerator)new MMDeviceEnumerator(); IMMDevice d; e.GetDefaultAudioEndpoint(flow, 0, out d);
      string id; d.GetId(out id); return new[] { id, Name(d) };
    }
    public static void SetDefault(string id) {
      var p = (IPolicyConfig)new CPolicyConfigClient();
      for (int role = 0; role < 3; role++) Marshal.ThrowExceptionForHR(p.SetDefaultEndpoint(id, role));
    }
  }
}
'@
}

function Get-AudioDevices([int]$Flow, [int]$States = 1) { [VT.AudioDevices2]::Devices($Flow, $States) | ForEach-Object { [pscustomobject]@{ Name = $_[1]; Id = $_[0] } } }
function Get-DefaultAudio([int]$Flow) { $d = [VT.AudioDevices2]::Default($Flow); [pscustomobject]@{ Name = $d[1]; Id = $d[0] } }
function Set-DefaultAudio([int]$Flow, [string]$NameOrId) {
    $m = @(Get-AudioDevices $Flow | Where-Object { $_.Id -eq $NameOrId -or $_.Name -like "*$NameOrId*" })
    if ($m.Count -ne 1) { throw "Пристрій '$NameOrId' не знайдено однозначно (збігів: $($m.Count))" }
    [VT.AudioDevices2]::SetDefault($m[0].Id)
    $m[0]
}
# Recording (1) and playback (0) shortcuts.
function Get-Mics { Get-AudioDevices 1 }
function Get-DefaultMic { Get-DefaultAudio 1 }
function Set-DefaultMic([string]$NameOrId) { Set-DefaultAudio 1 $NameOrId }
function Get-Speakers { Get-AudioDevices 0 }
function Get-DefaultSpeaker { Get-DefaultAudio 0 }
function Set-DefaultSpeaker([string]$NameOrId) { Set-DefaultAudio 0 $NameOrId }
