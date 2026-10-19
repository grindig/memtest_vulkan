//! Optional GPU telemetry through NVIDIA's NVML library, loaded at runtime when the driver ships it.
use std::ffi::{CString, c_char, c_int, c_uint, c_ulonglong, c_void};

type NvmlReturn = c_int;
type NvmlDevice = *mut c_void;

const NVML_SUCCESS: NvmlReturn = 0;
const NVML_CLOCK_GRAPHICS: c_int = 0;
const NVML_CLOCK_MEM: c_int = 2;
const NVML_TEMPERATURE_GPU: c_int = 0;
const NVML_MEMORY_ERROR_TYPE_CORRECTED: c_int = 0;
const NVML_MEMORY_ERROR_TYPE_UNCORRECTED: c_int = 1;
const NVML_VOLATILE_ECC: c_int = 0;

#[cfg(windows)]
const LIB_NAME: &str = "nvml.dll";
#[cfg(not(windows))]
const LIB_NAME: &str = "libnvidia-ml.so.1";

pub const NVIDIA_VENDOR_ID: u32 = 0x10DE;

// Clock event reasons that mean clocks were pulled down by a power or thermal limit.
// GpuIdle, application clock, sync boost and display clock reasons are not limits.
const LIMIT_REASONS: [(u64, &str); 5] = [
    (0x4, "SwPowerCap"),
    (0x8, "HwSlowdown"),
    (0x20, "SwThermal"),
    (0x40, "HwThermal"),
    (0x80, "HwPowerBrake"),
];

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Sample {
    pub mem_clock_mhz: Option<u32>,
    pub gpu_clock_mhz: Option<u32>,
    pub temperature_c: Option<u32>,
    pub power_w: Option<f32>,
    pub limit_reasons: Option<u64>,
    /// corrected + uncorrected volatile ECC errors, only on GPUs with ECC enabled
    pub ecc_errors: Option<u64>,
}

type InitFn = unsafe extern "C" fn() -> NvmlReturn;
type DeviceByBusIdFn = unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn;
type DeviceCountFn = unsafe extern "C" fn(*mut c_uint) -> NvmlReturn;
type DeviceByIndexFn = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
type ClockFn = unsafe extern "C" fn(NvmlDevice, c_int, *mut c_uint) -> NvmlReturn;
type PowerFn = unsafe extern "C" fn(NvmlDevice, *mut c_uint) -> NvmlReturn;
type ReasonsFn = unsafe extern "C" fn(NvmlDevice, *mut c_ulonglong) -> NvmlReturn;
type EccFn = unsafe extern "C" fn(NvmlDevice, c_int, c_int, *mut c_ulonglong) -> NvmlReturn;

pub struct Nvml {
    device: NvmlDevice,
    shutdown: InitFn,
    get_clock: ClockFn,
    get_temperature: ClockFn,
    get_power: PowerFn,
    get_limit_reasons: Option<ReasonsFn>,
    get_ecc: Option<EccFn>,
    // keeps the function pointers above valid; dropped after nvmlShutdown in Drop
    _library: libloading::Library,
}

impl Nvml {
    /// Loads NVML and finds the device at the given PCI address. With no PCI address known
    /// (VK_EXT_pci_bus_info missing) the only NVML device is used, if there is exactly one.
    pub fn open(domain: u32, bus: u32, device: u32, function: u32) -> Result<Self, String> {
        let library = unsafe { libloading::Library::new(LIB_NAME) }
            .map_err(|e| format!("{LIB_NAME} not loaded: {e}"))?;
        unsafe {
            let symbol = |name: &[u8]| -> Result<*const c_void, String> {
                library
                    .get::<*const c_void>(name)
                    .map(|s| *s)
                    .map_err(|e| format!("{e}"))
            };
            let init = std::mem::transmute::<*const c_void, InitFn>(symbol(b"nvmlInit_v2\0")?);
            let shutdown = std::mem::transmute::<*const c_void, InitFn>(symbol(b"nvmlShutdown\0")?);
            let rc = init();
            if rc != NVML_SUCCESS {
                return Err(format!("nvmlInit_v2 failed with code {rc}"));
            }
            let found = Self::find_device(&symbol, domain, bus, device, function);
            let nvml_device = match found {
                Ok(d) => d,
                Err(e) => {
                    shutdown();
                    return Err(e);
                }
            };
            let optional = |name: &[u8]| symbol(name).ok().filter(|p| !p.is_null());
            Ok(Self {
                device: nvml_device,
                shutdown,
                get_clock: std::mem::transmute::<*const c_void, ClockFn>(symbol(
                    b"nvmlDeviceGetClockInfo\0",
                )?),
                get_temperature: std::mem::transmute::<*const c_void, ClockFn>(symbol(
                    b"nvmlDeviceGetTemperature\0",
                )?),
                get_power: std::mem::transmute::<*const c_void, PowerFn>(symbol(
                    b"nvmlDeviceGetPowerUsage\0",
                )?),
                // renamed from ...ClocksThrottleReasons in newer drivers
                get_limit_reasons: optional(b"nvmlDeviceGetCurrentClocksEventReasons\0")
                    .or_else(|| optional(b"nvmlDeviceGetCurrentClocksThrottleReasons\0"))
                    .map(|p| std::mem::transmute::<*const c_void, ReasonsFn>(p)),
                get_ecc: optional(b"nvmlDeviceGetTotalEccErrors\0")
                    .map(|p| std::mem::transmute::<*const c_void, EccFn>(p)),
                _library: library,
            })
        }
    }

    fn find_device(
        symbol: &dyn Fn(&[u8]) -> Result<*const c_void, String>,
        domain: u32,
        bus: u32,
        device: u32,
        function: u32,
    ) -> Result<NvmlDevice, String> {
        let mut handle: NvmlDevice = std::ptr::null_mut();
        if bus != 0 || device != 0 || domain != 0 {
            let by_bus_id: DeviceByBusIdFn = unsafe {
                std::mem::transmute::<*const c_void, DeviceByBusIdFn>(symbol(
                    b"nvmlDeviceGetHandleByPciBusId_v2\0",
                )?)
            };
            // NVML prints bus ids with an 8-digit domain; older formats use 4 digits
            for id in [
                format!("{domain:08X}:{bus:02X}:{device:02X}.{function:X}"),
                format!("{domain:04X}:{bus:02X}:{device:02X}.{function:X}"),
            ] {
                let id = CString::new(id).map_err(|e| e.to_string())?;
                if unsafe { by_bus_id(id.as_ptr(), &mut handle) } == NVML_SUCCESS {
                    return Ok(handle);
                }
            }
            return Err(format!(
                "no NVML device at PCI {domain:04X}:{bus:02X}:{device:02X}.{function:X}"
            ));
        }
        let count_fn: DeviceCountFn = unsafe {
            std::mem::transmute::<*const c_void, DeviceCountFn>(symbol(b"nvmlDeviceGetCount_v2\0")?)
        };
        let by_index: DeviceByIndexFn = unsafe {
            std::mem::transmute::<*const c_void, DeviceByIndexFn>(symbol(
                b"nvmlDeviceGetHandleByIndex_v2\0",
            )?)
        };
        let mut count: c_uint = 0;
        if unsafe { count_fn(&mut count) } != NVML_SUCCESS || count != 1 {
            return Err(format!(
                "PCI address unknown and NVML lists {count} devices, so the tested one can't be identified"
            ));
        }
        match unsafe { by_index(0, &mut handle) } {
            NVML_SUCCESS => Ok(handle),
            rc => Err(format!(
                "nvmlDeviceGetHandleByIndex_v2 failed with code {rc}"
            )),
        }
    }

    pub fn sample(&self) -> Sample {
        let read_u32 = |f: &dyn Fn(*mut c_uint) -> NvmlReturn| {
            let mut value: c_uint = 0;
            (f(&mut value) == NVML_SUCCESS).then_some(value)
        };
        let ecc = |kind: c_int| {
            let get_ecc = self.get_ecc?;
            let mut count: c_ulonglong = 0;
            (unsafe { get_ecc(self.device, kind, NVML_VOLATILE_ECC, &mut count) } == NVML_SUCCESS)
                .then_some(count)
        };
        Sample {
            mem_clock_mhz: read_u32(&|v| unsafe {
                (self.get_clock)(self.device, NVML_CLOCK_MEM, v)
            }),
            gpu_clock_mhz: read_u32(&|v| unsafe {
                (self.get_clock)(self.device, NVML_CLOCK_GRAPHICS, v)
            }),
            temperature_c: read_u32(&|v| unsafe {
                (self.get_temperature)(self.device, NVML_TEMPERATURE_GPU, v)
            }),
            power_w: read_u32(&|v| unsafe { (self.get_power)(self.device, v) })
                .map(|milliwatts| milliwatts as f32 / 1000.0),
            limit_reasons: self.get_limit_reasons.and_then(|get| {
                let mut reasons: c_ulonglong = 0;
                (unsafe { get(self.device, &mut reasons) } == NVML_SUCCESS).then_some(reasons)
            }),
            ecc_errors: ecc(NVML_MEMORY_ERROR_TYPE_CORRECTED)
                .zip(ecc(NVML_MEMORY_ERROR_TYPE_UNCORRECTED))
                .map(|(corrected, uncorrected)| corrected + uncorrected),
        }
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
        }
    }
}

/// Telemetry gathered over one report interval.
#[derive(Default, Debug)]
pub struct IntervalStats {
    last: Sample,
    mem_clock_min: Option<u32>,
    mem_clock_max: Option<u32>,
    temperature_max: Option<u32>,
    power_max: Option<f32>,
    limit_reasons: u64,
}

impl IntervalStats {
    pub fn add(&mut self, sample: Sample) {
        if let Some(clock) = sample.mem_clock_mhz {
            self.mem_clock_min = Some(self.mem_clock_min.map_or(clock, |c| c.min(clock)));
            self.mem_clock_max = Some(self.mem_clock_max.map_or(clock, |c| c.max(clock)));
        }
        self.temperature_max = sample.temperature_c.max(self.temperature_max);
        self.power_max = match (self.power_max, sample.power_w) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.limit_reasons |= sample.limit_reasons.unwrap_or(0);
        self.last = sample;
    }
    pub fn mem_clock_range(&self) -> Option<(u32, u32)> {
        self.mem_clock_min.zip(self.mem_clock_max)
    }
    pub fn throttle_names(&self) -> Vec<&'static str> {
        LIMIT_REASONS
            .iter()
            .filter(|(bit, _)| self.limit_reasons & bit != 0)
            .map(|(_, name)| *name)
            .collect()
    }
    /// Compact status-line text, e.g. " mem:13001MHz core:2775MHz 58C max 263W"
    pub fn status_text(&self) -> String {
        let mut text = String::new();
        match self.mem_clock_range() {
            Some((low, high)) if low != high => text += &format!(" mem:{low}-{high}MHz"),
            Some((clock, _)) => text += &format!(" mem:{clock}MHz"),
            None => {}
        }
        if let Some(clock) = self.last.gpu_clock_mhz {
            text += &format!(" core:{clock}MHz");
        }
        if let Some(temperature) = self.temperature_max {
            text += &format!(" max {temperature}C");
        }
        if let Some(power) = self.power_max {
            text += &format!(" {power:.0}W");
        }
        let throttle = self.throttle_names();
        if !throttle.is_empty() {
            text += &format!(" limit:{}", throttle.join(","));
        }
        text
    }
    pub fn last(&self) -> Sample {
        self.last
    }
}
