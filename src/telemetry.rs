//! Optional GPU telemetry through NVIDIA's NVML library, loaded at runtime when the driver ships it,
//! plus the GDDR6X memory junction temperature from NVAPI on Windows.
use std::ffi::{CString, c_char, c_int, c_uint, c_ulonglong, c_void};

type NvmlReturn = c_int;
type NvmlDevice = *mut c_void;

const NVML_SUCCESS: NvmlReturn = 0;
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

// Clock event reasons that mean clocks were pulled down by a power or thermal limit, with the
// short code shown in the status table. GpuIdle, application clock, sync boost and display clock
// reasons are not limits.
const LIMIT_REASONS: [(u64, &str, &str); 5] = [
    (0x4, "SwPowerCap", "power"),
    (0x8, "HwSlowdown", "hwslow"),
    (0x20, "SwThermal", "thermal"),
    (0x40, "HwThermal", "thermal"),
    (0x80, "HwPowerBrake", "brake"),
];

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Sample {
    pub mem_clock_mhz: Option<u32>,
    /// GDDR6X memory junction temperature, when the card has the sensor
    pub mem_temperature_c: Option<f32>,
    pub core_temperature_c: Option<u32>,
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
    memory_junction: Result<MemoryJunction, String>,
    // keeps the function pointers above valid; dropped after nvmlShutdown in Drop
    _library: libloading::Library,
}

impl Nvml {
    /// Loads NVML and finds the device at the given PCI address. With no PCI address known
    /// (VK_EXT_pci_bus_info missing) the only NVML device is used, if there is exactly one.
    pub fn open(
        domain: u32,
        bus: u32,
        device: u32,
        function: u32,
        device_name: &str,
    ) -> Result<Self, String> {
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
                memory_junction: MemoryJunction::open(bus, device_name),
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

    /// Why the memory junction temperature isn't available, or None when it is.
    pub fn memory_temperature_missing(&self) -> Option<&str> {
        self.memory_junction.as_ref().err().map(String::as_str)
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
            mem_temperature_c: self.memory_junction.as_ref().ok().and_then(|m| m.read()),
            core_temperature_c: read_u32(&|v| unsafe {
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

/// GDDR6X memory junction temperature from NVAPI's undocumented thermal-sensor query, the source
/// HWiNFO and LibreHardwareMonitor read. NVML has no memory temperature on GeForce cards.
#[cfg(windows)]
struct MemoryJunction {
    gpu: *mut c_void,
    sensor_index: usize,
    mask: u32,
    get_sensors: ThermalSensorsFn,
    _library: libloading::Library,
}

#[cfg(windows)]
#[repr(C)]
struct ThermalSensors {
    version: u32,
    mask: u32,
    reserved: [i32; 8],
    temperatures: [i32; 32], // 1/256 degrees C
}

#[cfg(windows)]
type ThermalSensorsFn = unsafe extern "C" fn(*mut c_void, *mut ThermalSensors) -> c_int;

#[cfg(windows)]
fn query_sensors(get: ThermalSensorsFn, gpu: *mut c_void, mask: u32) -> Option<ThermalSensors> {
    let mut sensors = ThermalSensors {
        version: size_of::<ThermalSensors>() as u32 | (2 << 16),
        mask,
        reserved: [0; 8],
        temperatures: [0; 32],
    };
    (unsafe { get(gpu, &mut sensors) } == 0).then_some(sensors)
}

#[cfg(windows)]
impl MemoryJunction {
    // nvapi_QueryInterface ids
    const INITIALIZE: u32 = 0x0150_E828;
    const ENUM_PHYSICAL_GPUS: u32 = 0xE5AC_921F;
    const GPU_GET_BUS_ID: u32 = 0x1BE0_B8E5;
    const GPU_THERMAL_GET_SENSORS: u32 = 0x65FE_3AAD; // undocumented

    fn open(pci_bus: u32, device_name: &str) -> Result<Self, String> {
        // the junction sensor's slot differs per generation, same mapping as LibreHardwareMonitor
        let sensor_index = if device_name.starts_with("NVIDIA GeForce RTX 50") {
            2
        } else if device_name.starts_with("NVIDIA GeForce RTX 40") {
            7
        } else {
            9
        };
        let library = unsafe { libloading::Library::new("nvapi64.dll") }
            .map_err(|e| format!("nvapi64.dll not loaded: {e}"))?;
        unsafe {
            let query = *library
                .get::<unsafe extern "C" fn(u32) -> *const c_void>(b"nvapi_QueryInterface\0")
                .map_err(|e| e.to_string())?;
            let lookup = |id: u32| {
                let function = query(id);
                if function.is_null() {
                    Err(format!("NVAPI function 0x{id:08X} not available"))
                } else {
                    Ok(function)
                }
            };
            let initialize = std::mem::transmute::<*const c_void, unsafe extern "C" fn() -> c_int>(
                lookup(Self::INITIALIZE)?,
            );
            let enum_gpus = std::mem::transmute::<
                *const c_void,
                unsafe extern "C" fn(*mut [*mut c_void; 64], *mut u32) -> c_int,
            >(lookup(Self::ENUM_PHYSICAL_GPUS)?);
            let get_bus_id = std::mem::transmute::<
                *const c_void,
                unsafe extern "C" fn(*mut c_void, *mut u32) -> c_int,
            >(lookup(Self::GPU_GET_BUS_ID)?);
            let get_sensors = std::mem::transmute::<*const c_void, ThermalSensorsFn>(lookup(
                Self::GPU_THERMAL_GET_SENSORS,
            )?);
            if initialize() != 0 {
                return Err("NvAPI_Initialize failed".to_owned());
            }
            let mut handles = [std::ptr::null_mut(); 64];
            let mut count = 0u32;
            if enum_gpus(&mut handles, &mut count) != 0 {
                return Err("NvAPI_EnumPhysicalGPUs failed".to_owned());
            }
            let gpus = &handles[..(count as usize).min(handles.len())];
            let gpu = gpus
                .iter()
                .copied()
                .find(|&gpu| {
                    let mut bus = 0u32;
                    get_bus_id(gpu, &mut bus) == 0 && bus == pci_bus
                })
                .or_else(|| (pci_bus == 0 && gpus.len() == 1).then(|| gpus[0]))
                .ok_or_else(|| format!("no NVAPI GPU on PCI bus 0x{pci_bus:02X}"))?;
            // the query fails for mask bits past the last sensor
            let sensor_count = (0..32)
                .take_while(|bit| query_sensors(get_sensors, gpu, 1 << bit).is_some())
                .count();
            if sensor_index >= sensor_count {
                return Err(format!(
                    "the GPU reports {sensor_count} thermal sensors, none in the memory junction slot"
                ));
            }
            let junction = Self {
                gpu,
                sensor_index,
                mask: ((1u64 << sensor_count) - 1) as u32,
                get_sensors,
                _library: library,
            };
            match junction.read() {
                Some(_) => Ok(junction),
                None => {
                    Err("the memory junction sensor reads 0 (no GDDR6X junction sensor)".to_owned())
                }
            }
        }
    }

    fn read(&self) -> Option<f32> {
        let sensors = query_sensors(self.get_sensors, self.gpu, self.mask)?;
        let celsius = sensors.temperatures[self.sensor_index] as f32 / 256.0;
        (celsius > 0.0 && celsius < 150.0).then_some(celsius)
    }
}

#[cfg(not(windows))]
struct MemoryJunction;

#[cfg(not(windows))]
impl MemoryJunction {
    fn open(_pci_bus: u32, _device_name: &str) -> Result<Self, String> {
        Err("memory junction temperature comes from NVAPI, which is Windows-only".to_owned())
    }

    fn read(&self) -> Option<f32> {
        None
    }
}

fn max_f32(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// Telemetry gathered over one report interval.
#[derive(Default, Debug)]
pub struct IntervalStats {
    last: Sample,
    mem_clock_min: Option<u32>,
    mem_clock_max: Option<u32>,
    mem_temperature_max: Option<f32>,
    core_temperature_max: Option<u32>,
    power_max: Option<f32>,
    limit_reasons: u64,
}

impl IntervalStats {
    pub fn add(&mut self, sample: Sample) {
        if let Some(clock) = sample.mem_clock_mhz {
            self.mem_clock_min = Some(self.mem_clock_min.map_or(clock, |c| c.min(clock)));
            self.mem_clock_max = Some(self.mem_clock_max.map_or(clock, |c| c.max(clock)));
        }
        self.mem_temperature_max = max_f32(self.mem_temperature_max, sample.mem_temperature_c);
        self.core_temperature_max = sample.core_temperature_c.max(self.core_temperature_max);
        self.power_max = max_f32(self.power_max, sample.power_w);
        self.limit_reasons |= sample.limit_reasons.unwrap_or(0);
        self.last = sample;
    }
    pub fn mem_clock_range(&self) -> Option<(u32, u32)> {
        self.mem_clock_min.zip(self.mem_clock_max)
    }
    pub fn mem_temperature_max(&self) -> Option<f32> {
        self.mem_temperature_max
    }
    pub fn core_temperature_max(&self) -> Option<u32> {
        self.core_temperature_max
    }
    pub fn power_max(&self) -> Option<f32> {
        self.power_max
    }
    pub fn throttle_names(&self) -> Vec<&'static str> {
        LIMIT_REASONS
            .iter()
            .filter(|(bit, _, _)| self.limit_reasons & bit != 0)
            .map(|(_, name, _)| *name)
            .collect()
    }
    /// Short limit codes for the status table, e.g. "power" or "power,thermal"
    pub fn limit_codes(&self) -> String {
        let mut codes: Vec<&str> = Vec::new();
        for (bit, _, code) in LIMIT_REASONS {
            if self.limit_reasons & bit != 0 && !codes.contains(&code) {
                codes.push(code);
            }
        }
        codes.join(",")
    }
    pub fn last(&self) -> Sample {
        self.last
    }
}
