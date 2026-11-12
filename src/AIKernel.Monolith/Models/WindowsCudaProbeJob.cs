using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace AIKernel.Monolith.Models;

internal static partial class WindowsCudaProbeJob
{
    private const uint JobObjectLimitKillOnJobClose = 0x00002000;
    private const int JobObjectExtendedLimitInformation = 9;

    internal static unsafe SafeFileHandle Create()
    {
        var job = CreateJobObject(0, 0);
        if (job.IsInvalid)
        {
            var error = Marshal.GetLastPInvokeError();
            job.Dispose();
            throw new Win32Exception(error, "Could not create the CUDA probe job.");
        }

        // A null security attributes pointer makes the handle non-inheritable.
        ExtendedLimitInformation limits = default;
        limits.Basic.LimitFlags = JobObjectLimitKillOnJobClose;
        if (SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits,
            (uint)sizeof(ExtendedLimitInformation)) == 0)
        {
            var error = Marshal.GetLastPInvokeError();
            job.Dispose();
            throw new Win32Exception(error, "Could not configure the CUDA probe job.");
        }

        return job;
    }

    internal static void Assign(SafeFileHandle job, Process process)
    {
        if (AssignProcessToJobObject(job, process.SafeHandle) == 0)
            throw new Win32Exception(Marshal.GetLastPInvokeError(), "Could not assign the CUDA probe to its job.");
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimitInformation
    {
        internal long PerProcessUserTimeLimit;
        internal long PerJobUserTimeLimit;
        internal uint LimitFlags;
        internal nuint MinimumWorkingSetSize;
        internal nuint MaximumWorkingSetSize;
        internal uint ActiveProcessLimit;
        internal nuint Affinity;
        internal uint PriorityClass;
        internal uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        internal ulong ReadOperationCount;
        internal ulong WriteOperationCount;
        internal ulong OtherOperationCount;
        internal ulong ReadTransferCount;
        internal ulong WriteTransferCount;
        internal ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimitInformation
    {
        internal BasicLimitInformation Basic;
        internal IoCounters Io;
        internal nuint ProcessMemoryLimit;
        internal nuint JobMemoryLimit;
        internal nuint PeakProcessMemoryUsed;
        internal nuint PeakJobMemoryUsed;
    }

    [LibraryImport("kernel32.dll", EntryPoint = "CreateJobObjectW", SetLastError = true)]
    private static partial SafeFileHandle CreateJobObject(nint attributes, nint name);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static unsafe partial int SetInformationJobObject(SafeFileHandle job, int informationClass,
        ExtendedLimitInformation* information, uint length);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static partial int AssignProcessToJobObject(SafeFileHandle job, SafeProcessHandle process);
}
