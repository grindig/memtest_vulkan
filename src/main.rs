mod close;
mod erupt_vendored_utils_loading;
mod input;
mod output;
mod ram;
mod telemetry;

pub use erupt::{CustomEntryLoader, LoaderError}; // republish for vendored part

use byte_strings::c_str;
use core::cmp::{max, min};
use erupt::{
    DeviceLoader, ExtendableFrom, InstanceLoader, ObjectHandle,
    extensions::{ext_debug_utils, ext_memory_budget, ext_pci_bus_info},
    vk,
};
use erupt_vendored_utils_loading::{EntryLoader, new_loader};
use hhmmss::Hhmmss;
use std::{
    env,
    ffi::{CStr, OsString, c_void},
    fmt,
    io::Write,
    mem,
    sync::atomic::{AtomicBool, AtomicI64, AtomicIsize, Ordering::SeqCst},
    time,
};

pub trait CmpAssign {
    fn min_assign(&mut self, other: Self);
    fn max_assign(&mut self, other: Self);
}

impl<T> CmpAssign for T
where
    T: core::cmp::Ord,
{
    fn min_assign(&mut self, other: Self) {
        if &other < self {
            *self = other;
        }
    }
    fn max_assign(&mut self, other: Self) {
        if &other > self {
            *self = other;
        }
    }
}

struct CStrStaticPtr<const N: usize>([*const std::os::raw::c_char; N]);

impl<const N: usize> core::ops::Deref for CStrStaticPtr<N> {
    type Target = [*const std::os::raw::c_char; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl<const N: usize> Sync for CStrStaticPtr<N> {}

const VK_LOADER_DEBUG: &str = "VK_LOADER_DEBUG";
const LAYER_KHRONOS_VALIDATION: &CStr = c_str!("VK_LAYER_KHRONOS_validation");
static LAYER_KHRONOS_VALIDATION_ARRAY: CStrStaticPtr<1> =
    CStrStaticPtr([LAYER_KHRONOS_VALIDATION.as_ptr()]);

// erupt CreateInfoBuilder's has problem without passing enabled_extensions argument,
// https://github.com/GpuZelenograd/memtest_vulkan/issues/53
// Workaround it by explicitly passing empty array
static NO_EXTENSIONS_ARRAY: CStrStaticPtr<0> = CStrStaticPtr([]);

const GB: f32 = (1024 * 1024 * 1024) as f32; // a GiB: all sizes and speeds are printed as GiB and GiB/s
const READ_SHADER: &[u32] = memtest_vulkan_build::compiled_vk_compute_spirv!(
    r#"
struct IOBuf
{
    err_bit1_idx: array<u32, 32>,
    err_bitcount: array<u32, 32>,
    mem_bitcount: array<u32, 32>,
    actual_ff: u32,
    actual_max: u32,
    actual_min: u32,
    idx_max: u32,
    idx_min: u32,
    done_iter_or_err: u32,
    iter: u32,
    calc_param: u32,
    pattern: u32,
    march_flags: u32,
    march_chunk: u32,
    reserved1: u32,
    first_elem: vec4<u32>
}

@group(0) @binding(0) var<storage, read_write> io: IOBuf;
@group(0) @binding(1) var<storage, read_write> test: array<vec4<u32>>;

//pattern word layout: bits 0..7 family, bits 8..15 variant, bit 16 inverts the whole value
let PATTERN_RANDOM: u32 = 1u;
let PATTERN_SOLID: u32 = 2u;
let PATTERN_TOGGLE: u32 = 3u;
let PATTERN_WALKING: u32 = 4u;
let PATTERN_PRBS7: u32 = 5u;
let PATTERN_INVERT_BIT: u32 = 0x10000u;

//PRBS7 (x^7+x^6+1) sequence: 127 bits followed by its first 33 bits, LSB-first, so any 32-bit window is readable without wrap
fn prbs7_word(k: u32) -> u32
{
    return select(select(select(select(0x4F143040u, 0xFEA99DD2u, k == 3u), 0xC6F6B648u, k == 2u), 0xE17CAE68u, k == 1u), 0x9E286081u, k == 0u);
}

fn prbs7_window(n: u32) -> u32
{
    let w = n >> 5u;
    let b = n & 31u;
    let lo = prbs7_word(w) >> b;
    let hi = select(prbs7_word(w + 1u) << ((32u - b) & 31u), 0u, b == 0u);
    return lo | hi;
}

fn pattern_u32(e: u32, p: u32, pattern: u32) -> u32
{
    let family = pattern & 0xFFu;
    let variant = (pattern >> 8u) & 0xFFu;
    var value: u32 = 0u;
    if family == PATTERN_RANDOM {
        //lowbias32 hash of a per-pass bijection of the index: high entropy, still unique per address
        var x: u32 = (e * 0x9E3779B1u) ^ p;
        x = x ^ (x >> 16u);
        x = x * 0x7FEB352Du;
        x = x ^ (x >> 15u);
        x = x * 0x846CA68Bu;
        value = x ^ (x >> 16u);
    } else if family == PATTERN_SOLID {
        value = 0u;
    } else if family == PATTERN_TOGGLE {
        //bytes with exactly 4 zero bits are left alone by GDDR DBI-DC, so their complement swaps toggle every lane
        let base_idx = variant % 3u;
        let base = select(select(0x55555555u, 0x33333333u, base_idx == 1u), 0x0F0F0F0Fu, base_idx == 0u);
        let mode = (variant / 3u) % 5u;
        if mode == 0u {
            value = base ^ 0xFFFF0000u;
        } else {
            value = base ^ select(0u, 0xFFFFFFFFu, ((e >> (mode - 1u)) & 1u) == 1u);
        }
    } else if family == PATTERN_WALKING {
        let pos = e + p;
        if (variant & 1u) == 0u {
            value = 1u << (pos % 32u);
        } else {
            value = 0x01010101u << (pos % 8u);
        }
    } else if family == PATTERN_PRBS7 {
        value = prbs7_window((e + p) % 127u);
    } else {
        //original memtest_vulkan pattern: address rotated by itself
        let a = e + p + 1u;
        let shift = a % 31u;
        value = (a << shift) | (a >> (32u - shift));
    }
    if (pattern & PATTERN_INVERT_BIT) != 0u {
        value = value ^ 0xFFFFFFFFu;
    }
    return value;
}

fn test_value_by_index(i: u32, p: u32, pattern: u32)->vec4<u32>
{
    let e = i * 4u;
    return vec4<u32>(pattern_u32(e, p, pattern), pattern_u32(e + 1u, p, pattern), pattern_u32(e + 2u, p, pattern), pattern_u32(e + 3u, p, pattern));
}

//slow path, executed only on errors found
fn record_errors(effective_addr: u32, actual_value: vec4<u32>, expected_value: vec4<u32>)
{
    for(var i: i32 = 0; i < 4; i++) {
        let actual_u32 = actual_value[i];
        let error_mask = actual_u32 ^ expected_value[i];
        if error_mask == 0u {
            continue;
        }
        let one_bits = countOneBits(error_mask);
        if one_bits == 1u
        {
            let bit_idx = firstLeadingBit(error_mask);
            atomicAdd(&io.err_bit1_idx[bit_idx], 1u);
        }
        atomicAdd(&io.err_bitcount[one_bits % 32u], 1u);
        let vec_addr: u32 = effective_addr * 4u + u32(i);
        atomicMax(&io.idx_max, vec_addr);
        atomicMin(&io.idx_min, vec_addr);
        atomicMax(&io.done_iter_or_err, 0xFFFFFFFFu); //ERROR_STATUS
        let actual_bits = countOneBits(actual_u32);
        if actual_bits == 32u
        {
            atomicAdd(&io.actual_ff, 1u);
        }
        else
        {
            atomicAdd(&io.mem_bitcount[actual_bits], 1u);
            atomicMax(&io.actual_max, actual_u32);
            atomicMin(&io.actual_min, actual_u32);
        }
    }
}

let TEST_WINDOW_1D_MAX_GROUPS: u32 = 0x4000u;
let TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY: u32 = 0x2000u;//don't inner-multiply by window size

@compute @workgroup_size(64, 1, 1)
fn read(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    let addr_mod = effective_invocation_id % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let new_mod = (11 * effective_invocation_id + 999 * io.iter + io.calc_param +  7 * (effective_invocation_id / TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY)) % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let effective_addr = effective_invocation_id - addr_mod + new_mod; //make read order a bit rotated, not strictly sequential
    let actual_value : vec4<u32> = test[effective_addr];
    let expected_value : vec4<u32> = test_value_by_index(effective_addr, io.calc_param, io.pattern);
    if any(actual_value != expected_value) {
        record_errors(effective_addr, actual_value, expected_value);
    }
    //assign done_iter_or_err only on specific index (performance reasons)
    if effective_addr == 0 {
        atomicMax(&io.done_iter_or_err, io.iter);
    } else if effective_addr == 1 {
        io.first_elem = expected_value;
    }
}

@compute @workgroup_size(64, 1, 1)
fn write(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    //make global_invocation_id processing specific memory addr different on writing compared to reading
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 8u * TEST_WINDOW_1D_MAX_GROUPS;//don't inner-multiply by window size
    let proccessed_mod = effective_invocation_id % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = effective_invocation_id + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
    test[proccessed_idx] = test_value_by_index(proccessed_idx, io.calc_param, io.pattern);
}

@compute @workgroup_size(64, 1, 1)
fn emulate_write_bugs(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 8u * TEST_WINDOW_1D_MAX_GROUPS;//don't inner-multiply by window size
    let proccessed_mod = effective_invocation_id % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = effective_invocation_id + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
    test[proccessed_idx] = test_value_by_index(proccessed_idx, io.calc_param, io.pattern);
    if proccessed_idx == 0xADBAu {
        test[proccessed_idx][1] ^= 0x400000u;//error simulation for test
    } else if proccessed_idx == 0xADBBu {
        test[proccessed_idx] = vec4<u32>(0xFFFFFFFFu);//all-ones words, like a dead channel
    }
}

//One March C- element over the window: every cell is read, checked against the pattern (or its complement)
//and immediately overwritten with the opposite value, so reads and writes share the bus.
//Each workgroup walks its own contiguous chunk with coalesced 1KB steps; descending mode reverses chunks, steps and lanes.
@compute @workgroup_size(64, 1, 1)
fn march(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
    let p = io.calc_param;
    let pattern = io.pattern;
    let read_inverted = (io.march_flags & 1u) != 0u;
    let descending = (io.march_flags & 2u) != 0u;
    let chunk = io.march_chunk; //vec4 per thread, set by the host that also sizes the dispatch
    let group_count = num_workgroups[0] * num_workgroups[1];
    let linear_group = workgroup_id[0] + workgroup_id[1] * num_workgroups[0];
    let group = select(linear_group, group_count - 1u - linear_group, descending);
    let lane = select(local_index, 63u - local_index, descending);
    let read_mask = select(vec4<u32>(0u), vec4<u32>(0xFFFFFFFFu), vec4<bool>(read_inverted));
    let chunk_base = group * 64u * chunk;
    for(var j: u32 = 0u; j < chunk; j++) {
        let step = select(j, chunk - 1u - j, descending);
        let idx = chunk_base + step * 64u + lane;
        let expected_value = test_value_by_index(idx, p, pattern) ^ read_mask;
        let actual_value = test[idx];
        if any(actual_value != expected_value) {
            record_errors(idx, actual_value, expected_value);
        }
        test[idx] = expected_value ^ vec4<u32>(0xFFFFFFFFu);
        if idx == 0u {
            atomicMax(&io.done_iter_or_err, io.iter);
        }
    }
}
"#
);

const WG_SIZE: i64 = 64;
const VEC_SIZE: usize = 4; //vector processed by single workgroup item
const ELEMENT_SIZE: i64 = std::mem::size_of::<u32>() as i64;
const ELEMENT_BIT_SIZE: usize = (ELEMENT_SIZE * 8) as usize;
const TEST_WINDOW_1D_MAX_GROUPS: i64 = 0x4000;
const TEST_WINDOW_SIZE_GRANULARITY: i64 =
    VEC_SIZE as i64 * WG_SIZE * ELEMENT_SIZE * TEST_WINDOW_1D_MAX_GROUPS * 8_i64;
const TEST_WINDOW_MAX_SIZE: i64 = 4 * 1024 * 1024 * 1024 - TEST_WINDOW_SIZE_GRANULARITY;
const TEST_DATA_KEEP_FREE: i64 = 400 * 1024 * 1024;
const MIN_WANTED_ALLOCATION: i64 = TEST_DATA_KEEP_FREE;
const ALLOCATION_TRY_STEP: i64 = TEST_DATA_KEEP_FREE;

const MAP_FOR_GETTING_ADDRESS_SIZE: u64 = 2 * 1024 * 1024; // signle huge page

const MARCH_VEC4_PER_THREAD: u32 = 16; // passed to the shader in IOBuf::march_chunk
// March C- elements after the write pass: up(r0,w1) up(r1,w0) down(r0,w1) down(r1,w0); the read pass is the final r0
const MARCH_ELEMENTS: [(u32, &str); 4] = [
    (0b00, "MARCH_UP_R0W1"),
    (0b01, "MARCH_UP_R1W0"),
    (0b10, "MARCH_DOWN_R0W1"),
    (0b11, "MARCH_DOWN_R1W0"),
];
// March C- on every pass took ~64% of GPU time; every 2nd block keeps streaming passes around half
const DEFAULT_MARCH_EVERY: u32 = 2;
// how long one window keeps its data untouched for retention checking before another window takes over
const STATIC_WINDOW_ROTATE_PERIOD: time::Duration = time::Duration::from_secs(60);
// GPU-timed bandwidth this far below the best seen for the same kernel+pattern, two reports in a row, is reported
const BANDWIDTH_DROP_THRESHOLD: f64 = 0.05;
// a kernel+pattern needs this many timed dispatches in a report interval before its median is trusted
const BANDWIDTH_MIN_DISPATCHES: usize = 8;
const TELEMETRY_PERIOD: time::Duration = time::Duration::from_secs(1);

const PATTERN_ADDR_ROTATE: u32 = 0;
const PATTERN_RANDOM: u32 = 1;
const PATTERN_SOLID: u32 = 2;
const PATTERN_TOGGLE: u32 = 3;
const PATTERN_WALKING: u32 = 4;
const PATTERN_PRBS7: u32 = 5;
const PATTERN_FAMILY_COUNT: u32 = 6;
const PATTERN_INVERT_BIT: u32 = 0x10000;
const PATTERN_NAMES: [&str; PATTERN_FAMILY_COUNT as usize] = [
    "addr_rotate",
    "random",
    "solid",
    "toggle",
    "walking",
    "prbs7",
];
// same table as in the shader: PRBS7 sequence followed by its first 33 bits, LSB-first
const PRBS7_WORDS: [u32; 5] = [0x9E286081, 0xE17CAE68, 0xC6F6B648, 0xFEA99DD2, 0x4F143040];

fn prbs7_window(n: u32) -> u32 {
    let w = (n >> 5) as usize;
    let b = n & 31;
    let lo = PRBS7_WORDS[w] >> b;
    let hi = if b == 0 {
        0
    } else {
        PRBS7_WORDS[w + 1] << (32 - b)
    };
    lo | hi
}

/// CPU mirror of the shader `pattern_u32`, used to sanity-check shader execution.
#[allow(clippy::manual_rotate)] // kept as similar to WGSL shader above as possible
fn pattern_u32(e: u32, p: u32, pattern: u32) -> u32 {
    let variant = (pattern >> 8) & 0xFF;
    let value = match pattern & 0xFF {
        PATTERN_RANDOM => {
            let mut x = e.wrapping_mul(0x9E3779B1) ^ p;
            x ^= x >> 16;
            x = x.wrapping_mul(0x7FEB352D);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846CA68B);
            x ^ (x >> 16)
        }
        PATTERN_SOLID => 0,
        PATTERN_TOGGLE => {
            let base = [0x0F0F0F0F, 0x33333333, 0x55555555][(variant % 3) as usize];
            match (variant / 3) % 5 {
                0 => base ^ 0xFFFF0000,
                mode => {
                    base ^ if (e >> (mode - 1)) & 1 == 1 {
                        0xFFFFFFFF
                    } else {
                        0
                    }
                }
            }
        }
        PATTERN_WALKING => {
            let pos = e.wrapping_add(p);
            if variant & 1 == 0 {
                1 << (pos % 32)
            } else {
                0x01010101 << (pos % 8)
            }
        }
        PATTERN_PRBS7 => prbs7_window(e.wrapping_add(p) % 127),
        _ => {
            let a = e.wrapping_add(p).wrapping_add(1);
            let shift = a % 31;
            if shift > 0 {
                a << shift | a >> (32 - shift)
            } else {
                a
            }
        }
    };
    if pattern & PATTERN_INVERT_BIT != 0 {
        !value
    } else {
        value
    }
}

/// Pattern for a window in a given iteration. Neighbouring windows and consecutive iterations get
/// different families, so a dropped write never leaves a value that still matches.
fn pattern_for(iter: u32, window_idx: u32, extended: bool) -> u32 {
    if !extended {
        return PATTERN_ADDR_ROTATE;
    }
    let slot = iter.wrapping_add(window_idx);
    let family = slot % PATTERN_FAMILY_COUNT;
    let round = slot / PATTERN_FAMILY_COUNT;
    let invert = if round & 1 == 1 {
        PATTERN_INVERT_BIT
    } else {
        0
    };
    let variant = ((round >> 1) & 0xFF) << 8;
    family | variant | invert
}

/// Window that gets the March C- elements in this pass, or None for a pass without March.
/// A window's pattern family advances by one every pass, so March stays on one window for
/// PATTERN_FAMILY_COUNT consecutive passes to see every family, then skips
/// (march_every - 1) such blocks and moves on to the next non-static window.
fn march_window_for(
    iteration: u32,
    window_count: i64,
    static_window: i64,
    march_every: u32,
) -> Option<i64> {
    if march_every == 0 || window_count < 2 {
        return None;
    }
    let block = iteration.saturating_sub(1) / PATTERN_FAMILY_COUNT;
    if !block.is_multiple_of(march_every) {
        return None;
    }
    let march_block = (block / march_every) as i64;
    Some((static_window + 1 + march_block % (window_count - 1)) % window_count)
}

fn pattern_name(pattern: u32) -> String {
    format!(
        "{}{}/v{}",
        PATTERN_NAMES
            .get((pattern & 0xFF) as usize)
            .unwrap_or(&"unknown"),
        if pattern & PATTERN_INVERT_BIT != 0 {
            "~"
        } else {
            ""
        },
        (pattern >> 8) & 0xFF
    )
}

struct ComputePipelines {
    read: vk::Pipeline,
    #[allow(dead_code)]
    write: vk::Pipeline,
    #[allow(dead_code)]
    emulate_write_bugs: vk::Pipeline,
    march: vk::Pipeline,
}
#[derive(Default)]
struct U64HexDebug(i64);

impl fmt::Debug for U64HexDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

#[derive(Default)]
struct DriverVersionDebug(u32);

impl fmt::Debug for DriverVersionDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let upper10bits = self.0 >> 22;
        if upper10bits > 2 {
            // NVIDIA (0x75a04180 = 470.129.06) and intel-mesa-on-linux (0x5402006 = 21.2.6) driver versioning
            return write!(f, "v{}(0x{:X})", upper10bits, self.0);
        }
        let upper18bits = self.0 >> 14;
        if upper18bits > 2 && upper18bits < 500 {
            //intel-on-windows driver versioning (0x19453c = [30.0.]101.1340)
            return write!(f, "v{}(0x{:X})", upper18bits, self.0);
        }
        if self.0 < 64 {
            //basic small number versioning like llvm
            return write!(f, "ver{}", self.0);
        }
        // don't parse AMD versioning like "0x8000E6"
        write!(f, "0x{:X}", self.0)
    }
}

#[derive(Copy, Clone)]
struct MostlyZeroArr<const LEN: usize>([u32; LEN]);

impl<const LEN: usize> fmt::Display for MostlyZeroArr<LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if LEN < 8 {
            for i in 0..LEN {
                write!(f, "0x{:08X} ", self.0[i])?;
            }
            return Ok(());
        }
        for i in 0..LEN {
            let next_row = i / 16;
            if next_row > 0
                && self.0[next_row * 16..next_row * 16 + 16]
                    .iter()
                    .all(|v| v == &0u32)
            {
                continue; //skip entire line of zeroes
            }
            if i % 16 == 0 && i != 0 {
                write!(f, "   0x{:X}? ", i / 16)?;
            }
            let vali = self.0[i];
            if vali > 999999u32 {
                write!(f, "{:3}m", vali / 1000000u32)?;
            } else if vali > 9999u32 {
                write!(f, "{:3}k", vali / 1000u32)?;
            } else if vali > 0 {
                write!(f, "{:4}", vali)?;
            } else {
                write!(f, "    ")?; //zero
            }
            if i % 16 == 15 {
                writeln!(f)?;
            } else if i % 4 == 3 {
                write!(f, "|")?;
            } else if i % 4 == 1 {
                write!(f, " ")?;
            }
        }
        Ok(())
    }
}

impl<const LEN: usize> std::default::Default for MostlyZeroArr<LEN> {
    fn default() -> Self {
        Self([0; LEN])
    }
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
struct IOBuf {
    err_bit1_idx: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    err_bitcount: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    mem_bitcount: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    actual_ff: u32,
    actual_max: u32,
    actual_min: u32,
    idx_max: u32,
    idx_min: u32,
    done_iter_or_err: u32,
    iter: u32,
    calc_param: u32,
    pattern: u32,
    march_flags: u32,
    march_chunk: u32, // vec4 per March thread; the host sizes the dispatch with the same value
    reserved1: u32,   // padding: first_elem is a vec4 and must stay 16-byte aligned like in WGSL
    first_elem: MostlyZeroArr<VEC_SIZE>,
}

impl fmt::Display for IOBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // actual_min/max skip all-ones words, so with only those the range is empty
        let values_range = if self.actual_min <= self.actual_max {
            format!("0x{:08X}..=0x{:08X}", self.actual_min, self.actual_max)
        } else {
            "none".to_owned()
        };
        writeln!(
            f,
            "values range: {}   FFFFFFFF-like count:{}   pattern:{}    bit-level stats table:",
            values_range,
            self.actual_ff,
            pattern_name(self.pattern)
        )?;
        writeln!(
            f,
            "         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF"
        )?;
        write!(f, "SinglIdx{}", self.err_bit1_idx)?;
        write!(f, "TogglCnt{}", self.err_bitcount)?;
        write!(f, "1sInValu{}", self.mem_bitcount)?;
        Ok(())
    }
}

impl IOBuf {
    fn for_initial_iteration() -> Self {
        let mut result = Self::default();
        result.prepare_next_iter_write();
        result
    }
    fn prepare_next_iter_write(&mut self) {
        self.reset_errors();
        self.iter += 1;
        self.set_calc_param_for_starting_window();
    }
    fn set_calc_param_for_starting_window(&mut self) {
        self.calc_param = self.iter.wrapping_mul(0x100107);
    }
    /// Clean copy describing what `window_idx` holds after this iteration's write pass.
    fn for_window(&self, window_idx: i64, extended: bool) -> Self {
        let mut result = *self;
        result.reset_errors();
        result.set_calc_param_for_starting_window();
        result.calc_param = result.calc_param.wrapping_add(window_idx as u32 * 0x81_u32);
        result.pattern = pattern_for(self.iter, window_idx as u32, extended);
        result.march_flags = 0;
        result
    }
    fn reset_errors(&mut self) {
        *self = IOBuf {
            iter: self.iter,
            calc_param: self.calc_param,
            pattern: self.pattern,
            march_flags: self.march_flags,
            idx_max: u32::MIN,
            idx_min: u32::MAX,
            actual_max: u32::MIN,
            actual_min: u32::MAX,
            ..Self::default()
        };
    }
    fn get_error_addresses_and_count(
        &self,
        buf_offset: i64,
    ) -> Option<(std::ops::RangeInclusive<U64HexDebug>, i64)> {
        if self.done_iter_or_err == self.iter {
            None
        } else {
            // words read back as all-ones are counted in actual_ff instead of mem_bitcount
            let total_errors =
                self.mem_bitcount.0.iter().sum::<u32>() as i64 + self.actual_ff as i64;
            Some((
                std::ops::RangeInclusive::<U64HexDebug>::new(
                    U64HexDebug(buf_offset + self.idx_min as i64 * ELEMENT_SIZE),
                    U64HexDebug(buf_offset + (self.idx_max + 1) as i64 * ELEMENT_SIZE - 1),
                ),
                total_errors,
            ))
        }
    }
    fn check_vec_first(&self) -> Result<(), Box<dyn std::error::Error>> {
        const TEST_IDX: u32 = 1;
        for lane in 0..VEC_SIZE {
            let e = TEST_IDX * VEC_SIZE as u32 + lane as u32;
            let expected = pattern_u32(e, self.calc_param, self.pattern);
            if expected != self.first_elem.0[lane] {
                println!("{} 0x{:08X}", self, expected);
                return Err("unexpected calculated value, maybe shader execution is broken".into());
            }
        }
        Ok(())
    }
}

trait MapErrStr {
    type ValueType;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
    fn unwrap_or_display(self, env: &ProcessEnv) -> Self::ValueType
    where
        Self: Sized,
    {
        match self.err_as_str() {
            Err(e) => display_this_process_result(Some(e), env),
            Ok(v) => v,
        }
    }
}

trait MapErrRetryWithLowerMemory {
    type ValueType;
    fn err_retry_with_lower_memory(
        self,
        env: &ProcessEnv,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
}

impl<T> MapErrStr for std::result::Result<T, erupt::LoaderError> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|res| {
            let msg = match res {
                erupt::LoaderError::SymbolNotAvailable => {
                    "SymbolNotAvailable in Loader".to_string()
                }
                erupt::LoaderError::VulkanError(result) => format!("{}", result),
            } + " while getting "
                + std::any::type_name::<Self::ValueType>();
            msg.into()
        })
    }
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|res| {
            let msg = match res {
                erupt::LoaderError::SymbolNotAvailable => {
                    "SymbolNotAvailable in Loader".to_string()
                }
                erupt::LoaderError::VulkanError(result) => format!("{}", result),
            } + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            msg.into()
        })
    }
}

impl<T> MapErrStr for erupt::utils::VulkanResult<T> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg =
                res.to_string() + " while getting " + std::any::type_name::<Self::ValueType>();
            msg.into()
        })
    }
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg = res.to_string()
                + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            msg.into()
        })
    }
}
impl<T> MapErrRetryWithLowerMemory for erupt::utils::VulkanResult<T> {
    type ValueType = T;
    fn err_retry_with_lower_memory(
        self,
        env: &ProcessEnv,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg = res.to_string()
                + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            if !env.interactive
                && !close::check_any_bits_set(close::fetch_status(), close::app_status::INITED_OK)
            {
                if env.verbose() {
                    println!("Retrying with lower memory due to {}", msg);
                }
                //immediate exit in non-interactive during init to initiate try with lower memory
                close::immediate_exit(true);
            }
            msg.into()
        })
    }
}

unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let cstr_message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };

    eprintln!("{}", cstr_message.to_string_lossy());
    vk::FALSE
}
fn memory_requirements(
    device: &erupt::DeviceLoader,
    min_wanted_allocation: i64,
) -> Result<(vk::MemoryRequirements, vk::BufferCreateInfoBuilder<'_>), Box<dyn std::error::Error>> {
    let test_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(min_wanted_allocation as u64);
    let test_buffer =
        unsafe { device.create_buffer(&test_buffer_create_info, None) }.err_as_str()?;
    let test_mem_reqs = unsafe { device.get_buffer_memory_requirements(test_buffer) };
    unsafe { device.destroy_buffer(test_buffer, None) }; //buffer was used for getting memory reuirements. After allocation size may be smaller
    Ok((test_mem_reqs, test_buffer_create_info))
}

fn free_test_mem_and_buffers(
    device: &erupt::DeviceLoader,
    buffers: &mut Vec<vk::Buffer>,
    memories: &mut Vec<vk::DeviceMemory>,
) {
    for some_buffer in buffers.drain(..) {
        unsafe {
            device.destroy_buffer(some_buffer, None);
        }
    }
    for some_memory in memories.drain(..) {
        unsafe {
            device.free_memory(some_memory, None);
        }
    }
}
fn try_fill_default_mem_budget<Writer: std::io::Write>(
    selected_device: &NamedComputeDevice,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) {
    if env.verbose() {
        let _ = writeln!(log_dupler, "Loading memory info for selected device...",);
    }

    if env.mem_budget_estimated() {
        //don't estimate budget if it is already specified
        return;
    }

    let mut max_budget = 0;
    for i in 0..selected_device.memory_props.memory_heap_count as usize {
        if env.verbose() {
            let _ = writeln!(
                log_dupler,
                "heap size {:4.1}GiB budget {:4.1}GiB usage {:4.1}GiB flags={:#?}",
                selected_device.memory_props.memory_heaps[i].size as f32 / GB,
                selected_device.budget_props.heap_budget[i] as f32 / GB,
                selected_device.budget_props.heap_usage[i] as f32 / GB,
                selected_device.memory_props.memory_heaps[i].flags,
            );
        }
        if !selected_device.memory_props.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        {
            continue;
        }
        let mut heap_free = selected_device.memory_props.memory_heaps[i].size as i64;
        let usage = selected_device.budget_props.heap_usage[i] as i64;
        if usage > 0 && usage < heap_free {
            heap_free -= usage;
        }
        let budget = selected_device.budget_props.heap_budget[i] as i64;
        if budget > 0 {
            heap_free = min(heap_free, budget);
        }
        if selected_device.physical_props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
            // for non-descrete GPUs consult the system free memory
            if let Some(ram_budget) = ram::budget() {
                heap_free = min(heap_free, ram_budget as i64);
            }
        }
        max_budget.max_assign(heap_free - TEST_DATA_KEEP_FREE);
    }
    env.set_mem_budget_limit(max_budget);
}

fn prepare_and_test_device<Writer: std::io::Write>(
    instance: &erupt::InstanceLoader,
    selected: &NamedComputeDevice,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    let queue_create_info = vec![
        vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(selected.queue_family_index)
            .queue_priorities(&[1.0]),
    ];

    let device_create_info = vk::DeviceCreateInfoBuilder::new()
        .enabled_extension_names(&*NO_EXTENSIONS_ARRAY)
        .queue_create_infos(&queue_create_info);

    let device =
        match unsafe { DeviceLoader::new(instance, selected.physical_device, &device_create_info) }
        {
            Ok(device) => device,
            Err(e) => display_this_process_result(Some(e.into()), env),
        };
    let queue = unsafe { device.get_device_queue(selected.queue_family_index, 0) };

    let cmd_pool_info = vk::CommandPoolCreateInfoBuilder::new()
        .queue_family_index(selected.queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool =
        unsafe { device.create_command_pool(&cmd_pool_info, None) }.unwrap_or_display(env);

    let cmd_buf_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_info) }.unwrap_or_display(env);

    let desc_pool_sizes = &[vk::DescriptorPoolSizeBuilder::new()
        .descriptor_count(2)
        ._type(vk::DescriptorType::STORAGE_BUFFER)];
    let desc_pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(desc_pool_sizes)
        .max_sets(1);
    let desc_pool =
        unsafe { device.create_descriptor_pool(&desc_pool_info, None) }.unwrap_or_display(env);

    let desc_layout_bindings = &[
        vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let desc_layout_info =
        vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(desc_layout_bindings);
    let desc_layouts = [
        unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }
            .unwrap_or_display(env),
    ];

    let desc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(&desc_layouts);
    let desc_sets = unsafe { device.allocate_descriptor_sets(&desc_info) }.unwrap_or_display(env);

    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&desc_layouts);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .unwrap_or_display(env);

    let spv_code = Vec::from(READ_SHADER);
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv_code);
    let shader_mod =
        unsafe { device.create_shader_module(&create_info, None) }.unwrap_or_display(env);

    let pipeline_infos = [
        c_str!("read"),
        c_str!("write"),
        c_str!("emulate_write_bugs"),
        c_str!("march"),
    ]
    .map(|name| {
        let shader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(shader_mod)
            .name(name);
        vk::ComputePipelineCreateInfoBuilder::new()
            .layout(pipeline_layout)
            .stage(*shader_stage)
    });

    let pipelines =
        unsafe { device.create_compute_pipelines(Default::default(), &pipeline_infos, None) }
            .unwrap_or_display(env);
    let pipelines = ComputePipelines {
        read: pipelines[0],
        write: pipelines[1],
        emulate_write_bugs: pipelines[2],
        march: pipelines[3],
    };

    if let Err(e) = test_device(
        &device,
        queue,
        cmd_bufs,
        desc_sets,
        &pipeline_layout,
        &pipelines,
        log_dupler,
        &selected.label,
        selected.memory_props,
        selected.physical_props,
        selected.max_memory_allocation_size,
        selected.timestamp_valid_bits,
        selected.pci_bus_info,
        env,
    ) {
        display_this_process_result(Some(e), env)
    }
    display_this_process_result(None, env)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum KernelKind {
    Write,
    Read,
    March,
}

impl KernelKind {
    const COUNT: usize = 3;
    fn name(self) -> &'static str {
        match self {
            KernelKind::Write => "write",
            KernelKind::Read => "read",
            KernelKind::March => "march",
        }
    }
}

#[derive(Copy, Clone, Default)]
struct BandwidthSample {
    bytes: f64,
    secs: f64,
}

impl BandwidthSample {
    fn gbps(&self) -> Option<f64> {
        (self.secs > 0.0).then(|| self.bytes / GB as f64 / self.secs)
    }
}

#[derive(Default)]
struct BandwidthHistory {
    best_gbps: f64,
    low_reports: u32,
    dispatch_gbps: Vec<f64>, // one value per timed dispatch in the current report interval
}

#[derive(Debug, PartialEq)]
struct BandwidthDrop {
    kind: KernelKind,
    family: u32,
    gbps: f64,
    best_gbps: f64,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// GPU-timed bandwidth per kernel and pattern family. Link EDC retries on unstable GDDR
/// don't produce data errors, only lost bandwidth, so a sustained drop is worth reporting.
/// The median of individual dispatches is compared, so a few dispatches slowed by other
/// GPU work can't fake a drop.
#[derive(Default)]
struct BandwidthTracker {
    history: std::collections::BTreeMap<(KernelKind, u32), BandwidthHistory>,
    totals: [BandwidthSample; KernelKind::COUNT],
}

impl BandwidthTracker {
    fn add(&mut self, kind: KernelKind, pattern: u32, bytes: i64, gpu_secs: Option<f64>) {
        let Some(secs) = gpu_secs.filter(|s| *s > 0.0) else {
            return;
        };
        self.history
            .entry((kind, pattern & 0xFF))
            .or_default()
            .dispatch_gbps
            .push(bytes as f64 / GB as f64 / secs);
        self.totals[kind as usize].bytes += bytes as f64;
        self.totals[kind as usize].secs += secs;
    }
    fn interval_gbps(&self, kind: KernelKind) -> Option<f64> {
        self.totals[kind as usize].gbps()
    }
    /// Closes the report interval and returns the kernel+pattern pairs that were more than
    /// BANDWIDTH_DROP_THRESHOLD below their best for two comparable intervals in a row.
    /// An interval that isn't comparable (startup, partial, too few dispatches) breaks the streak.
    fn finish_interval(&mut self, comparable: bool) -> Vec<BandwidthDrop> {
        let mut dropped = Vec::new();
        for (&(kind, family), entry) in self.history.iter_mut() {
            let mut samples = mem::take(&mut entry.dispatch_gbps);
            if !comparable || samples.len() < BANDWIDTH_MIN_DISPATCHES {
                entry.low_reports = 0;
                continue;
            }
            let gbps = median(&mut samples);
            if entry.best_gbps > 0.0 && gbps < entry.best_gbps * (1.0 - BANDWIDTH_DROP_THRESHOLD) {
                entry.low_reports += 1;
                if entry.low_reports >= 2 {
                    dropped.push(BandwidthDrop {
                        kind,
                        family,
                        gbps,
                        best_gbps: entry.best_gbps,
                    });
                }
            } else {
                entry.low_reports = 0;
            }
            entry.best_gbps = entry.best_gbps.max(gbps);
        }
        self.totals = Default::default();
        dropped
    }
}

/// Warning text for sustained drops, using the interval's telemetry to name the likely cause.
fn bandwidth_warning(
    drops: &[BandwidthDrop],
    telemetry: Option<&telemetry::IntervalStats>,
) -> Option<String> {
    if drops.is_empty() {
        return None;
    }
    let list = drops
        .iter()
        .map(|d| {
            format!(
                "{}/{} {:.1}GiB/s ({:+.1}%)",
                d.kind.name(),
                PATTERN_NAMES.get(d.family as usize).unwrap_or(&"unknown"),
                d.gbps,
                (d.gbps / d.best_gbps - 1.0) * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    // power/thermal limits mostly lower the core clock, so a steady memory clock still matters with them
    let cause = match telemetry.map(|t| (t.mem_clock_range(), t.throttle_names())) {
        Some((Some((low, high)), _)) if low != high => format!(
            "The memory clock varied {low}-{high}MHz in this interval, so a clock change is the likely cause."
        ),
        Some((Some((clock, _)), limits)) if !limits.is_empty() => format!(
            "The memory clock stayed at {clock}MHz, but the driver also reports clock limiting ({}), which lowers the core clock. If the drop persists without limiting, suspect link EDC retries (unstable memory).",
            limits.join(",")
        ),
        Some((Some((clock, _)), _)) => format!(
            "The memory clock stayed at {clock}MHz with no clock limiting reported, which points to link EDC retries (unstable memory)."
        ),
        _ => "If the memory clock is steady this points to link EDC retries (unstable memory), otherwise to thermal or power throttling.".to_owned(),
    };
    Some(format!(
        "BANDWIDTH DROP: median dispatch bandwidth more than {:.0}% below best for 2+ reports: {list}. {cause}",
        BANDWIDTH_DROP_THRESHOLD * 100.0
    ))
}

fn report_window_errors<Writer: std::io::Write>(
    log_dupler: &mut output::LogDupler<Writer>,
    buffer_out: &IOBuf,
    test_offset: i64,
    test_window_size: i64,
    mode: &str,
    first_iter_start: time::Instant,
    last_error_state: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some((error_range, total_errors)) = buffer_out.get_error_addresses_and_count(test_offset)
    {
        *last_error_state = format!("LAST ERROR at {}", first_iter_start.elapsed().hhmmssxxx());
        close::raise_status_bit(close::app_status::RUNTIME_ERRORS);
        let test_elems = test_window_size / ELEMENT_SIZE;
        write!(
            log_dupler,
            "Error found. Mode {}, total errors 0x{:X} out of 0x{:X} ({:2.8}%)\nErrors address range: {:?}",
            mode,
            total_errors,
            test_elems,
            total_errors as f64 / test_elems as f64 * 100.0f64,
            error_range,
        )?;
        writeln!(
            log_dupler,
            "  iteration:{}\n{}",
            buffer_out.iter, buffer_out
        )?;
    }
    Ok(())
}

fn allocate_test_windows(
    device: &erupt::DeviceLoader,
    buffer_create_info: vk::BufferCreateInfoBuilder<'_>,
    memory_type_index: u32,
    window_count: i64,
    window_size: i64,
    env: &ProcessEnv,
    buffers: &mut Vec<vk::Buffer>,
    memories: &mut Vec<vk::DeviceMemory>,
) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..window_count {
        let buffer =
            unsafe { device.create_buffer(&buffer_create_info.size(window_size as u64), None) }
                .err_retry_with_lower_memory(env, "create_buffer")?;
        buffers.push(buffer);
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocate_info = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(reqs.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { device.allocate_memory(&allocate_info, None) }
            .err_retry_with_lower_memory(env, "allocate_memory")?;
        memories.push(memory);
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .err_retry_with_lower_memory(env, "bind_buffer_memory")?;
    }
    Ok(())
}

fn test_device<Writer: std::io::Write>(
    device: &erupt::DeviceLoader,
    queue: vk::Queue,
    cmd_bufs: erupt::SmallVec<vk::CommandBuffer>,
    desc_sets: erupt::SmallVec<vk::DescriptorSet>,
    pipeline_layout: &vk::PipelineLayout,
    pipelines: &ComputePipelines,
    log_dupler: &mut output::LogDupler<Writer>,
    selected_label: &String,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    physical_props: vk::PhysicalDeviceProperties,
    max_memory_allocation_size: u64,
    timestamp_valid_bits: u32,
    pci_bus_info: ext_pci_bus_info::PhysicalDevicePCIBusInfoPropertiesEXT,
    env: &ProcessEnv,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut allocation_size = env.max_test_bytes.load(SeqCst);
    if allocation_size < MIN_WANTED_ALLOCATION {
        return Err("requested test size is smaller than minimum wanted".into());
    }
    // MEMTEST_VULKAN_MODE=classic keeps the original single-pattern test without March passes
    let extended =
        !env::var("MEMTEST_VULKAN_MODE").is_ok_and(|mode| mode.eq_ignore_ascii_case("classic"));
    // MEMTEST_VULKAN_MARCH_EVERY=N runs March C- in 1 of every N blocks of passes, 0 disables it
    let march_every = if extended {
        env::var("MEMTEST_VULKAN_MARCH_EVERY")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(DEFAULT_MARCH_EVERY)
    } else {
        0
    };

    let io_data_size = mem::size_of::<IOBuf>() as vk::DeviceSize;

    let io_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .size(io_data_size);
    let io_buffer = unsafe { device.create_buffer(&io_buffer_create_info, None) }.err_as_str()?;
    let io_mem_reqs = unsafe { device.get_buffer_memory_requirements(io_buffer) };
    let mut io_mem_indices = Vec::new();
    for i in 0..memory_props.memory_type_count {
        //test buffer comptibility flags expressed as bitmask
        let suitable = (io_mem_reqs.memory_type_bits & (1 << i)) != 0;
        let memory_type = memory_props.memory_types[i as usize];
        if env.verbose() && !memory_type.property_flags.is_empty() {
            let _ = writeln!(log_dupler, "{:2} {:?} ", i, memory_type);
        }
        if suitable
            && memory_type.property_flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        {
            io_mem_indices.push(i);
        }
    }

    // sorting by a flag value allows selection of index with the minimum count of new unknown flags
    let io_mem_index = io_mem_indices
        .into_iter()
        .min_by_key(|i| memory_props.memory_types[*i as usize].property_flags)
        .ok_or("This device lacks support for DEVICE_LOCAL+HOST_COHERENT memory type.")?;
    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "CoherentIO memory          type {} inside heap {:?}",
            io_mem_index,
            memory_props.memory_heaps
                [memory_props.memory_types[io_mem_index as usize].heap_index as usize]
        );
    }

    let io_memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(io_mem_reqs.size)
        .memory_type_index(io_mem_index);
    let io_memory =
        unsafe { device.allocate_memory(&io_memory_allocate_info, None) }.err_as_str()?;

    let mapped: *mut IOBuf =
        unsafe { device.map_memory(io_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::default()) }
            .err_as_str()?
            .cast::<IOBuf>();
    unsafe { device.bind_buffer_memory(io_buffer, io_memory, 0) }
        .err_as_str_context("bind_buffer_memory")?;

    let (test_mem_reqs, test_buffer_create_info) =
        memory_requirements(device, MIN_WANTED_ALLOCATION)?;

    let test_mem_index = (0..memory_props.memory_type_count)
        .filter(|mem_index| {
            //test buffer comptibility flags expressed as bitmask
            let memory_type = memory_props.memory_types[*mem_index as usize];
            let suitable = (test_mem_reqs.memory_type_bits & (1 << mem_index)) != 0;
            suitable
                && memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .max_by_key(|mem_index| {
            let memory_type = memory_props.memory_types[*mem_index as usize];
            let heap_size = memory_props.memory_heaps[memory_type.heap_index as usize].size;
            // Among greatest heap_size select index with HOST_VISIBLE flag if it is available,
            // but with the minimum count of other flags and minimal index
            (
                heap_size,
                memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                std::cmp::Reverse(memory_type.property_flags),
                std::cmp::Reverse(*mem_index),
            )
        })
        .ok_or("DEVICE_LOCAL test memory type not available")?;

    unsafe {
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSetBuilder::new()
                .dst_set(desc_sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                    .buffer(io_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)])],
            &[],
        );
    }

    let fence =
        unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.err_as_str()?;

    // GPU-side timing of every dispatch; without it speeds fall back to CPU wall-clock
    let timestamp_period_ns = physical_props.limits.timestamp_period as f64;
    let query_pool = if timestamp_valid_bits > 0 && timestamp_period_ns > 0.0 {
        unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfoBuilder::new()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .err_as_str()
        .ok()
    } else {
        None
    };
    let timestamp_mask = if timestamp_valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << timestamp_valid_bits) - 1
    };

    let submit_info = &[vk::SubmitInfoBuilder::new().command_buffers(&cmd_bufs)];

    // Every window is its own allocation and buffer, so a driver cap on a single allocation
    // no longer limits the whole test to ~3.5GB.
    let mut max_window_size = TEST_WINDOW_MAX_SIZE;
    if max_memory_allocation_size > 0 {
        max_window_size = min(
            max_window_size,
            min(max_memory_allocation_size, i64::MAX as u64) as i64,
        );
    }
    max_window_size = min(
        max_window_size,
        physical_props.limits.max_storage_buffer_range as i64,
    );
    max_window_size -= max_window_size % TEST_WINDOW_SIZE_GRANULARITY;
    max_window_size = max(max_window_size, TEST_WINDOW_SIZE_GRANULARITY);
    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "maxMemoryAllocationSize {} maxStorageBufferRange {:5.1}GiB -> window size limit {:5.3}GiB, timestamps {}",
            match max_memory_allocation_size {
                0 => "unknown".to_owned(),
                size if size >= 1 << 40 => "unlimited".to_owned(), // NVIDIA reports u64::MAX
                size => format!("{:5.1}GiB", size as f32 / GB),
            },
            physical_props.limits.max_storage_buffer_range as f32 / GB,
            max_window_size as f32 / GB,
            if query_pool.is_some() { "on" } else { "off" },
        );
    }

    //all preparations except huge buffer allocation done. Now allocate huge buffers as a last step to minize chance of allocation failure for small structures
    let mut test_memories: Vec<vk::DeviceMemory> = Vec::new();
    let mut test_buffers: Vec<vk::Buffer> = Vec::new();
    let mut test_window_count;
    let mut test_window_size;
    let mut warn_on_budget_alloc_fail = true;
    let mut execute_wait_queue;

    //The error state before all allocation tries
    let mut last_err: Box<dyn std::error::Error> =
        "No heap reports memory enough for testing".into();
    'memsize: loop {
        free_test_mem_and_buffers(device, &mut test_buffers, &mut test_memories);

        if allocation_size < MIN_WANTED_ALLOCATION {
            return Err(last_err);
        }

        test_window_count =
            allocation_size / max_window_size + i64::from(allocation_size % max_window_size != 0);
        test_window_count = max(test_window_count, 2); //at least 2 windows: for testing rereads and rws
        test_window_size = allocation_size / test_window_count;
        test_window_size = test_window_size - test_window_size % TEST_WINDOW_SIZE_GRANULARITY;
        if env.verbose() {
            let _ = writeln!(
                log_dupler,
                "Trying {:7.3}GiB as {} windows of {:5.3}GiB...",
                allocation_size as f32 / GB,
                test_window_count,
                test_window_size as f32 / GB
            );
        }
        match allocate_test_windows(
            device,
            test_buffer_create_info,
            test_mem_index,
            test_window_count,
            test_window_size,
            env,
            &mut test_buffers,
            &mut test_memories,
        ) {
            Err(err) => last_err = err,
            Ok(()) => {
                execute_wait_queue =
                    |window_idx: i64,
                     pipeline: vk::Pipeline,
                     vec4_per_thread: u32|
                     -> Result<Option<f64>, Box<dyn std::error::Error>> {
                        let test_vec4_count =
                            (test_window_size / ELEMENT_SIZE) as u32 / VEC_SIZE as u32;
                        unsafe {
                            device.update_descriptor_sets(
                                &[vk::WriteDescriptorSetBuilder::new()
                                    .dst_set(desc_sets[0])
                                    .dst_binding(1)
                                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                    .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                                        .buffer(test_buffers[window_idx as usize])
                                        .offset(0)
                                        .range(test_window_size as u64)])],
                                &[],
                            );
                            let cmd_buf = cmd_bufs[0];
                            device
                                .begin_command_buffer(
                                    cmd_buf,
                                    &vk::CommandBufferBeginInfo::default(),
                                )
                                .err_retry_with_lower_memory(env, "begin_command_buffer")?;
                            if let Some(pool) = query_pool {
                                device.cmd_reset_query_pool(cmd_buf, pool, 0, 2);
                                device.cmd_write_timestamp(
                                    cmd_buf,
                                    vk::PipelineStageFlagBits::TOP_OF_PIPE,
                                    pool,
                                    0,
                                );
                            }
                            device.cmd_bind_pipeline(
                                cmd_buf,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline,
                            );
                            device.cmd_bind_descriptor_sets(
                                cmd_buf,
                                vk::PipelineBindPoint::COMPUTE,
                                *pipeline_layout,
                                0,
                                &desc_sets,
                                &[],
                            );
                            device.cmd_dispatch(
                                cmd_buf,
                                (TEST_WINDOW_1D_MAX_GROUPS / WG_SIZE) as u32,
                                test_vec4_count
                                    / vec4_per_thread
                                    / TEST_WINDOW_1D_MAX_GROUPS as u32,
                                1,
                            );
                            if let Some(pool) = query_pool {
                                device.cmd_write_timestamp(
                                    cmd_buf,
                                    vk::PipelineStageFlagBits::BOTTOM_OF_PIPE,
                                    pool,
                                    1,
                                );
                            }
                            device
                                .end_command_buffer(cmd_buf)
                                .err_retry_with_lower_memory(env, "end_command_buffer")?;
                            device
                                .queue_submit(queue, submit_info, fence)
                                .err_retry_with_lower_memory(env, "queue_submit")?;
                            device
                                .wait_for_fences(&[fence], true, u64::MAX)
                                .err_retry_with_lower_memory(env, "wait_for_fences")?;
                            device
                                .reset_fences(&[fence])
                                .err_retry_with_lower_memory(env, "reset_fences")?;
                            if let Some(pool) = query_pool {
                                let mut stamps = [0u64; 2];
                                device
                                    .get_query_pool_results(
                                        pool,
                                        0,
                                        2,
                                        mem::size_of_val(&stamps),
                                        stamps.as_mut_ptr().cast(),
                                        mem::size_of::<u64>() as vk::DeviceSize,
                                        vk::QueryResultFlags::_64 | vk::QueryResultFlags::WAIT,
                                    )
                                    .err_as_str_context("get_query_pool_results")?;
                                let ticks = (stamps[1] & timestamp_mask)
                                    .wrapping_sub(stamps[0] & timestamp_mask)
                                    & timestamp_mask;
                                return Ok(Some(ticks as f64 * timestamp_period_ns * 1e-9));
                            }
                            Ok(None)
                        }
                    };
                unsafe { std::ptr::write(mapped, IOBuf::for_initial_iteration()) }
                //try to do initial memory fill to verify that allocation is really usable
                let mut overall_exec_result = Ok(());
                'window: for window_idx in 0..test_window_count {
                    for pipeline in [pipelines.write, pipelines.read] {
                        if let Err(e) = execute_wait_queue(window_idx, pipeline, 1) {
                            overall_exec_result = Err(e);
                            break 'window;
                        }
                    }
                }
                match overall_exec_result {
                    Err(e) => last_err = e,
                    Ok(()) => break 'memsize,
                }
            }
        }
        if env.verbose() {
            let _ = writeln!(
                log_dupler,
                "Don't testing {:5.1}GiB due to error: {}",
                allocation_size as f32 / GB,
                last_err
            );
        }
        if !env.interactive {
            close::immediate_exit(true);
        }
        if warn_on_budget_alloc_fail {
            warn_on_budget_alloc_fail = false;
            let _ = writeln!(
                log_dupler,
                "Failed allocating {:5.1}GiB, trying to use smaller size. More system memory can help.",
                allocation_size as f32 / GB
            );
        }
        allocation_size -= ALLOCATION_TRY_STEP;
    }

    let test_memory_type = memory_props.memory_types[test_mem_index as usize];
    if test_memory_type
        .property_flags
        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    {
        match unsafe {
            device.map_memory(
                test_memories[0],
                0,
                MAP_FOR_GETTING_ADDRESS_SIZE,
                vk::MemoryMapFlags::default(),
            )
        }
        .err_as_str()
        {
            Err(e) => {
                if env.verbose() {
                    let _ = writeln!(log_dupler, "Failed getting virtual address due to {e}");
                }
            }
            Ok(virt_address) => {
                let addr_details = ram::virt_addr_details(virt_address, physical_props.device_type);
                unsafe { device.unmap_memory(test_memories[0]) }; // After getting addresses the actual mapping is not needed anymore
                if let Some(details) = addr_details {
                    writeln!(
                        log_dupler,
                        "CPU address of tested memory: virtual=x{:0pw$X} {}",
                        virt_address.addr(),
                        details,
                        pw = ram::POINTER_HEX_PRINT_WIDTH
                    )?
                }
            }
        }
    }

    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "Test memory size {:5.1}GiB in {} windows  type {:2}: {:?} {:?} suitable 0b{:032b}",
            (test_window_size * test_window_count) as f32 / GB,
            test_window_count,
            test_mem_index,
            test_memory_type,
            memory_props.memory_heaps[test_memory_type.heap_index as usize],
            test_mem_reqs.memory_type_bits
        );
    }
    if extended {
        writeln!(
            log_dupler,
            "Extended mode: {} data-pattern families, {}, retention window rotates every {}s (MEMTEST_VULKAN_MODE=classic for the original test)",
            PATTERN_FAMILY_COUNT,
            match march_every {
                0 => "March C- off".to_owned(),
                1 => "March C- on every pass".to_owned(),
                n => format!("March C- on 1 of every {n} blocks of {PATTERN_FAMILY_COUNT} passes"),
            },
            STATIC_WINDOW_ROTATE_PERIOD.as_secs()
        )?;
    }
    if query_pool.is_none() {
        writeln!(
            log_dupler,
            "GPU timestamps unavailable: speeds use CPU time and bandwidth-drop detection is off"
        )?;
    }
    // MEMTEST_VULKAN_TELEMETRY=0 skips loading NVML
    let nvml = if physical_props.vendor_id != telemetry::NVIDIA_VENDOR_ID
        || env::var("MEMTEST_VULKAN_TELEMETRY").is_ok_and(|v| v == "0")
    {
        None
    } else {
        match telemetry::Nvml::open(
            pci_bus_info.pci_domain,
            pci_bus_info.pci_bus,
            pci_bus_info.pci_device,
            pci_bus_info.pci_function,
        ) {
            Ok(nvml) => {
                writeln!(
                    log_dupler,
                    "Telemetry: NVML (memory/core clock, temperature, power, clock limit reasons, ECC if enabled)"
                )?;
                Some(nvml)
            }
            Err(e) => {
                if env.verbose() {
                    let _ = writeln!(log_dupler, "Telemetry unavailable: {e}");
                }
                None
            }
        }
    };
    let mut telemetry_interval = telemetry::IntervalStats::default();
    // NVML lags behind the GPU waking from idle, so the first sample waits one TELEMETRY_PERIOD
    let mut last_telemetry_sample = time::Instant::now();
    // ECC counters only exist on GPUs with ECC enabled; any increase is a memory error the test didn't see
    let mut ecc_errors_reported = nvml.as_ref().and_then(|nvml| nvml.sample().ecc_errors);

    // allow write bugs emulation for testing purposes
    let emulate_write_bugs_iteration = env::var("MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or_default();
    let iter_count = 100000000; //by default exit after several days of testing
    let mut written_bytes = 0i64;
    let mut read_bytes = 0i64;
    let mut next_report_duration = time::Duration::from_secs(0);
    let extended_test_report_duration = time::Duration::from_secs(30);
    let mut reports_before_standard_done = 12i32;
    let mut write_duration = time::Duration::ZERO;
    let mut read_duration = time::Duration::ZERO;
    let mut buffer_in = IOBuf::for_initial_iteration();
    let first_iter_start = time::Instant::now();
    let mut last_status_output = first_iter_start;
    let mut last_error_state = "          NO ERRORS       ".to_owned();
    // The static window is written once and only re-read, which catches retention errors. It moves to
    // the next window periodically so every window gets both rewrites and long dwell times.
    let mut static_window = 0i64;
    let mut static_buf = IOBuf::for_initial_iteration(); // what the initial fill left in window 0
    let mut static_since = first_iter_start;
    let mut bandwidth = BandwidthTracker::default();
    for iteration in 1..=iter_count {
        let mut static_rewrite = false;
        if static_since.elapsed() >= STATIC_WINDOW_ROTATE_PERIOD {
            static_window = (static_window + 1) % test_window_count;
            static_rewrite = true;
            static_since = time::Instant::now();
        }
        let write_start = time::Instant::now();
        for window_idx in 0..test_window_count {
            if window_idx == static_window && !static_rewrite {
                continue;
            }
            let window_buf = buffer_in.for_window(window_idx, extended);
            if window_idx == static_window {
                static_buf = window_buf;
            }
            unsafe { std::ptr::write(mapped, window_buf) }
            let gpu_secs = execute_wait_queue(
                window_idx,
                if iteration != emulate_write_bugs_iteration {
                    pipelines.write
                } else {
                    pipelines.emulate_write_bugs
                },
                1,
            )?;
            written_bytes += test_window_size;
            bandwidth.add(
                KernelKind::Write,
                window_buf.pattern,
                test_window_size,
                gpu_secs,
            );
        }
        write_duration += write_start.elapsed();
        // March C- elements on one non-static window; the read pass below is the final r0
        if let Some(march_window) = march_window_for(
            iteration as u32,
            test_window_count,
            static_window,
            march_every,
        ) {
            let window_buf = buffer_in.for_window(march_window, extended);
            for (march_flags, mode) in MARCH_ELEMENTS {
                let mut march_buf = window_buf;
                march_buf.march_flags = march_flags;
                march_buf.march_chunk = MARCH_VEC4_PER_THREAD;
                unsafe { std::ptr::write(mapped, march_buf) }
                let gpu_secs =
                    execute_wait_queue(march_window, pipelines.march, MARCH_VEC4_PER_THREAD)?;
                // each element reads and writes the whole window
                bandwidth.add(
                    KernelKind::March,
                    march_buf.pattern,
                    2 * test_window_size,
                    gpu_secs,
                );
                let march_out = unsafe { std::ptr::read(mapped) };
                report_window_errors(
                    log_dupler,
                    &march_out,
                    test_window_size * march_window,
                    test_window_size,
                    mode,
                    first_iter_start,
                    &mut last_error_state,
                )?;
            }
        }
        let read_start = time::Instant::now();
        for window_idx in 0..test_window_count {
            let reread_mode_for_this_win = window_idx == static_window && !static_rewrite;
            let window_buf = if window_idx == static_window {
                // a fresh iter only changes the read order; the expected data comes from calc_param+pattern
                IOBuf {
                    iter: buffer_in.iter,
                    ..static_buf
                }
            } else {
                buffer_in.for_window(window_idx, extended)
            };
            unsafe { std::ptr::write(mapped, window_buf) }
            let test_offset = test_window_size * window_idx;
            let gpu_secs = execute_wait_queue(window_idx, pipelines.read, 1)?;
            bandwidth.add(
                KernelKind::Read,
                window_buf.pattern,
                test_window_size,
                gpu_secs,
            );
            let last_buffer_out = unsafe { std::ptr::read(mapped) };
            report_window_errors(
                log_dupler,
                &last_buffer_out,
                test_offset,
                test_window_size,
                if reread_mode_for_this_win {
                    "NEXT_RE_READ"
                } else {
                    "INITIAL_READ"
                },
                first_iter_start,
                &mut last_error_state,
            )?;
            last_buffer_out.check_vec_first()?;
        }
        read_bytes += test_window_size * test_window_count;
        read_duration += read_start.elapsed();
        if let Some(nvml) = &nvml
            && last_telemetry_sample.elapsed() >= TELEMETRY_PERIOD
        {
            telemetry_interval.add(nvml.sample());
            last_telemetry_sample = time::Instant::now();
        }
        let moment_iter_ends = time::Instant::now();
        let elapsed = moment_iter_ends - last_status_output;
        let stop_testing = close::close_requested();
        if elapsed > next_report_duration || stop_testing {
            // CPU-timed fallback when GPU timestamps are unavailable; March time counts toward neither
            let write_secs = write_duration.as_secs_f32();
            let passed_secs = read_duration.as_secs_f32();
            let (write_speed_gbps, check_speed_gbps) = match (
                bandwidth.interval_gbps(KernelKind::Write),
                bandwidth.interval_gbps(KernelKind::Read),
            ) {
                (Some(write_gbps), Some(read_gbps)) => (write_gbps as f32, read_gbps as f32),
                _ => (
                    if write_secs > 0.0001 {
                        written_bytes as f32 / GB / write_secs
                    } else {
                        0f32
                    },
                    if passed_secs > 0.0001 {
                        read_bytes as f32 / GB / passed_secs
                    } else {
                        0f32
                    },
                ),
            };
            let march_speed = bandwidth
                .interval_gbps(KernelKind::March)
                .map(|gbps| format!("   march:{gbps:7.1}GiB/s"))
                .unwrap_or_default();
            let telemetry_text = nvml
                .as_ref()
                .map(|_| telemetry_interval.status_text())
                .unwrap_or_default();
            let full_length_interval =
                next_report_duration == extended_test_report_duration && !stop_testing;
            let bandwidth_drops = bandwidth.finish_interval(full_length_interval);
            let second1 = time::Duration::from_secs(1);
            if next_report_duration.is_zero() {
                writeln!(log_dupler, "Standard 5-minute test of {}", selected_label)?;
                next_report_duration = second1; //2nd report after 1 second
            } else if next_report_duration == second1 {
                close::raise_status_bit(close::app_status::INITED_OK);
                next_report_duration = second1 * 5; //3rd report after 5 seconds
            } else {
                next_report_duration = extended_test_report_duration; //all later reports
            }
            if reports_before_standard_done == 0 {
                let has_errors = close::check_any_bits_set(
                    close::fetch_status(),
                    close::app_status::RUNTIME_ERRORS,
                );
                match has_errors {
                    true => writeln!(log_dupler, "Standard 5-minute test fail - ERRORS FOUND"),
                    false => writeln!(
                        log_dupler,
                        "Standard 5-minute test PASSed! Just press Ctrl+C unless you plan long test run."
                    ),
                }?;
                writeln!(
                    log_dupler,
                    "Extended endless test started; testing more than 2 hours is usually unneeded"
                )?;
                writeln!(
                    log_dupler,
                    "use Ctrl+C to stop it when you decide it's enough"
                )?;
            } else {
                let formatted_time_hhmmss = (moment_iter_ends - first_iter_start).hhmmssxxx();
                writeln!(
                    log_dupler,
                    "{}    written:{:7.1}GiB{:7.1}GiB/s      checked:{:7.1}GiB{:7.1}GiB/s{}     {}{}",
                    last_error_state,
                    written_bytes as f32 / GB,
                    write_speed_gbps,
                    read_bytes as f32 / GB,
                    check_speed_gbps,
                    march_speed,
                    formatted_time_hhmmss,
                    telemetry_text
                )?;
            }
            if let Some(warning) =
                bandwidth_warning(&bandwidth_drops, nvml.as_ref().map(|_| &telemetry_interval))
            {
                writeln!(log_dupler, "{warning}")?;
            }
            if let (Some(reported), Some(now)) =
                (ecc_errors_reported, telemetry_interval.last().ecc_errors)
                && now > reported
            {
                last_error_state =
                    format!("LAST ERROR at {}", first_iter_start.elapsed().hhmmssxxx());
                close::raise_status_bit(close::app_status::RUNTIME_ERRORS);
                writeln!(
                    log_dupler,
                    "ECC errors found: the driver counted {} new corrected/uncorrected memory errors",
                    now - reported
                )?;
                ecc_errors_reported = Some(now);
            }
            telemetry_interval = Default::default();
            reports_before_standard_done -= 1;
            if reports_before_standard_done == 0 {
                // The last iteration before report has a sleep before it to test hot gpu behaviour
                // in a situation of load pause and low-performance memory frequency
                std::thread::sleep(next_report_duration / 2);
            }
            written_bytes = 0i64;
            read_bytes = 0i64;
            write_duration = time::Duration::ZERO;
            read_duration = time::Duration::ZERO;
            last_status_output = time::Instant::now();
        }
        if stop_testing {
            let _ = writeln!(log_dupler, "received user interruption, testing stopped");
            break;
        }
        buffer_in.prepare_next_iter_write();
    }
    // Cleanup & Destruction
    unsafe {
        device.device_wait_idle().err_as_str()?;

        free_test_mem_and_buffers(device, &mut test_buffers, &mut test_memories);

        if let Some(pool) = query_pool {
            device.destroy_query_pool(pool, None);
        }
        device.destroy_buffer(io_buffer, None);
        device.unmap_memory(io_memory);
        device.free_memory(io_memory, None);

        device.destroy_fence(fence, None);
    }
    close::declare_exit_due_timeout();
    Ok(())
}

struct NamedComputeDevice {
    label: String,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    physical_props: vk::PhysicalDeviceProperties,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    budget_props: ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXT,
    index_in_device_iteration: usize,
    max_memory_allocation_size: u64, // 0 when unknown (Vulkan 1.0)
    timestamp_valid_bits: u32,
    pci_bus_info: ext_pci_bus_info::PhysicalDevicePCIBusInfoPropertiesEXT, // zeroes when unknown
}

fn load_instance<Writer: std::io::Write>(
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<
    (
        erupt::InstanceLoader,
        EntryLoader,
        vk::DebugUtilsMessengerEXT,
    ),
    Box<dyn std::error::Error>,
> {
    let override_vk_loader_debug = env::var_os(VK_LOADER_DEBUG).is_none();
    if env.verbose() && override_vk_loader_debug {
        // memtest_vulkan is single-threaded during init.
        unsafe { env::set_var(VK_LOADER_DEBUG, "error,warn") };
    }

    let mut entry = new_loader()?;
    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "Verbose feature enabled (or 'verbose' found in name). Vulkan instance {}.{}.{}",
            vk::api_version_major(entry.instance_version()),
            vk::api_version_minor(entry.instance_version()),
            vk::api_version_patch(entry.instance_version())
        );
        for (idx, prop) in unsafe { entry.enumerate_instance_layer_properties(None) }
            .err_as_str()
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            if idx == 0 {
                let _ = writeln!(log_dupler, "Available: ");
            } else {
                let _ = write!(log_dupler, ", ");
            }

            let _ = write!(log_dupler, "{}", unsafe {
                CStr::from_ptr(prop.layer_name.as_ptr())
                    .to_str()
                    .unwrap_or("Invalid property_name")
            });
        }
        let _ = writeln!(log_dupler);
        for (idx, ext) in unsafe { entry.enumerate_instance_extension_properties(None, None) }
            .err_as_str()
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            if idx == 0 {
                let _ = write!(log_dupler, "Extensions: ");
            } else {
                let _ = write!(log_dupler, ", ");
            }

            let _ = write!(log_dupler, "{}", unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr())
                    .to_str()
                    .unwrap_or("Invalid extension_name")
            });
        }
        let _ = writeln!(log_dupler);
        let _ = writeln!(log_dupler);
    }

    let instance_extensions = vec![ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME];

    let app_info = vk::ApplicationInfoBuilder::new().api_version(vk::API_VERSION_1_1);
    let instance_create_info = vk::InstanceCreateInfoBuilder::new()
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&*LAYER_KHRONOS_VALIDATION_ARRAY)
        .application_info(&app_info);

    let mut messenger = vk::DebugUtilsMessengerEXT::null();
    match unsafe {
        InstanceLoader::new(&entry, &instance_create_info)
            .err_as_str_context("instance with validation")
    } {
        Ok(instance_with_validation) => {
            let mut severity = ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                | ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT;
            if env.verbose() {
                severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::INFO_EXT;
                //lists all extensions, very verbose
                //severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT;
                let _ = writeln!(log_dupler, "DebugUtilsMessenger with seveity {severity:?}");
            }
            let create_info = ext_debug_utils::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                .message_severity(severity)
                .message_type(ext_debug_utils::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(debug_callback));
            messenger = unsafe {
                instance_with_validation.create_debug_utils_messenger_ext(&create_info, None)
            }
            .result()
            .unwrap_or_default();
            return Ok((instance_with_validation, entry, messenger));
        }
        Err(e) => {
            if env.verbose() {
                let _ = writeln!(log_dupler, "Not using validation layers due to {e}");
            }
        }
    }

    //fallback creation without validation and extensions
    let simple_instance_try = unsafe {
        InstanceLoader::new(
            &entry,
            &vk::InstanceCreateInfoBuilder::new()
                .enabled_extension_names(&*NO_EXTENSIONS_ARRAY)
                .application_info(&app_info),
        )
    }
    .err_as_str_context("instance. Try specifying icd.json via VK_DRIVER_FILES env var");
    match simple_instance_try {
        Ok(instance) => Ok((instance, entry, messenger)),
        Err(e) => {
            if !override_vk_loader_debug {
                return Err(e);
            }
            drop(entry);
            //retry instance creation with loader debug enabled
            // memtest_vulkan is single-threaded during init.
            unsafe { env::set_var(VK_LOADER_DEBUG, "all") };
            entry = new_loader()?;
            let debug_instance_try = unsafe {
                InstanceLoader::new(
                    &entry,
                    &vk::InstanceCreateInfoBuilder::new().application_info(&app_info),
                )
            }
            .map_err(|_second_error_ignired| e)?;
            Ok((debug_instance_try, entry, messenger))
        }
    }
}
struct LoadedDevices {
    // use Option to enforce InstanceLoader must be dropped before EntryLoader inside Drop implementation;
    // outside Drop theay are always Some
    instance: Option<erupt::InstanceLoader>,
    entry: Option<EntryLoader>,
    messenger: vk::DebugUtilsMessengerEXT,
    devices: Vec<NamedComputeDevice>,
}

impl LoadedDevices {
    pub fn instance(&self) -> &erupt::InstanceLoader {
        self.instance.as_ref().unwrap()
    }
    pub fn new(
        instance: erupt::InstanceLoader,
        entry: EntryLoader,
        messenger: vk::DebugUtilsMessengerEXT,
        devices: Vec<NamedComputeDevice>,
    ) -> LoadedDevices {
        LoadedDevices {
            instance: Some(instance),
            entry: Some(entry),
            messenger,
            devices,
        }
    }
}
impl Drop for LoadedDevices {
    fn drop(&mut self) {
        unsafe {
            println!("Destroying vk instance...");
            if !self.messenger.is_null() {
                self.instance()
                    .destroy_debug_utils_messenger_ext(self.messenger, None);
            }
            self.instance().destroy_instance(None);
        }
        self.instance = None;
        self.entry = None;
    }
}
fn list_devices_ordered_labaled_from_1<Writer: std::io::Write>(
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<LoadedDevices, Box<dyn std::error::Error>> {
    let (instance, entry, messenger) = load_instance(env, log_dupler)?;
    //TODO: handle negative index passed as an internel index for using only single device during enumeration
    let mut compute_capable_devices: Vec<_> = unsafe { instance.enumerate_physical_devices(None) }
        .err_as_str()?
        .into_iter()
        .enumerate()
        .filter_map(|(index_in_device_iteration, physical_device)| unsafe {
            if ![None, Some(index_in_device_iteration)]
                .contains(&env.requested_index_in_device_iteration())
            {
                return None;
            }
            let (queue_family, timestamp_valid_bits) = match instance
                .get_physical_device_queue_family_properties(physical_device, None)
                .into_iter()
                .enumerate()
                .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                Some((queue_family, properties)) => {
                    (queue_family as u32, properties.timestamp_valid_bits)
                }
                None => return None,
            };

            let mut properties = instance.get_physical_device_properties(physical_device);
            let effective_version = (
                vk::api_version_major(properties.api_version),
                vk::api_version_minor(properties.api_version),
            );

            let mut pci_structure_request =
                ext_pci_bus_info::PhysicalDevicePCIBusInfoPropertiesEXTBuilder::new();

            let mut budget_request =
                ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXTBuilder::new();

            let mut maintenance3_request = vk::PhysicalDeviceMaintenance3PropertiesBuilder::new();

            //older vulkan implementations like broadcom on RaspberryPi lacks vk_1_1 support even if application requested it

            let memory_props = if effective_version >= (1, 1) {
                let mut device_props2 = vk::PhysicalDeviceProperties2Builder::new()
                    .extend_from(&mut pci_structure_request)
                    .extend_from(&mut maintenance3_request);

                instance.get_physical_device_properties2(physical_device, &mut device_props2);
                properties = device_props2.properties;
                let mut memory_props2 = vk::PhysicalDeviceMemoryProperties2Builder::new()
                    .extend_from(&mut budget_request);
                instance
                    .get_physical_device_memory_properties2(physical_device, &mut memory_props2);
                memory_props2.memory_properties
            } else {
                instance.get_physical_device_memory_properties(physical_device)
            };
            let pci_props_structure = *pci_structure_request;
            let budget_props = *budget_request;
            let max_memory_allocation_size = maintenance3_request.max_memory_allocation_size;

            Some((
                physical_device,
                queue_family,
                properties,
                pci_props_structure,
                memory_props,
                budget_props,
                index_in_device_iteration,
                max_memory_allocation_size,
                timestamp_valid_bits,
            ))
        })
        .collect();
    compute_capable_devices.sort_by_key(|(_, _, props, pci_props, _, _, _, _, _)| {
        // assuming that for multi-GPU scenarious the minimal PCIe bus number is used for picture output
        // and the maximal is the test target select by default "the descrete GPU with maximal PCIe bus number"
        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => (0, std::cmp::Reverse(pci_props.pci_bus)),
            vk::PhysicalDeviceType::INTEGRATED_GPU => (1, std::cmp::Reverse(pci_props.pci_bus)),
            _ => (2, std::cmp::Reverse(pci_props.pci_bus)),
        }
    });
    let mut numbered_devices: Vec<NamedComputeDevice> = Vec::new();
    for (i, d) in compute_capable_devices.iter().enumerate() {
        let props = d.2;
        let pci_props = d.3;
        let api_info = if env.verbose() {
            std::format!(
                "-{}- API {}.{}.{}  {:?}",
                d.6 + 1,
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
                DriverVersionDebug(props.driver_version),
            )
        } else {
            String::new()
        };

        let memory_props = d.4;
        let mut total_mem_estimation = 0i64;
        for i in 0..memory_props.memory_heap_count as usize {
            if !memory_props.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            {
                continue;
            }
            total_mem_estimation = max(
                total_mem_estimation,
                memory_props.memory_heaps[i].size as i64,
            );
        }
        numbered_devices.push(NamedComputeDevice {
            label: std::format!(
                "{}: Bus=0x{:02X}:{:02X} DevId=0x{:04X} {api_info}  {}GiB {}",
                i + 1,
                pci_props.pci_bus,
                pci_props.pci_device,
                props.device_id,
                (total_mem_estimation as f32 / GB).ceil(),
                unsafe {
                    CStr::from_ptr(props.device_name.as_ptr())
                        .to_str()
                        .unwrap_or("Invalid device_name")
                }
            ),
            physical_device: d.0,
            queue_family_index: d.1,
            physical_props: props,
            memory_props,
            budget_props: d.5,
            index_in_device_iteration: d.6,
            max_memory_allocation_size: d.7,
            timestamp_valid_bits: d.8,
            pci_bus_info: d.3,
        });
    }
    Ok(LoadedDevices::new(
        instance,
        entry,
        messenger,
        numbered_devices,
    ))
}

fn prompt_for_label(verbose: bool) -> Option<isize> {
    let mut device_test_index = Some(0isize);
    let prompt_start = time::Instant::now();
    let mut prompt_duration: Option<time::Duration> = Some(time::Duration::from_secs(10));

    let mut input_reader = input::Reader::default();
    let no_timer_prompt =
        String::from("                                                   Override index to test:");
    loop {
        let mut prompt = &no_timer_prompt;
        let formatted_prompt: String;
        if let Some(effective_duration) = prompt_duration {
            if effective_duration < prompt_start.elapsed() {
                println!();
                println!("    ...first device autoselected");
                break;
            } else {
                let duration_left = effective_duration - prompt_start.elapsed();
                formatted_prompt = std::format!(
                    "(first device will be autoselected in {} seconds)   Override index to test:",
                    duration_left.as_secs()
                );
                prompt = &formatted_prompt;
            }
        }
        match input_reader.input_digit_step(prompt, &time::Duration::from_millis(250)) {
            Ok(input::ReaderEvent::Edited) => prompt_duration = None,
            Ok(input::ReaderEvent::Canceled) => {
                println!();
                device_test_index = None;
                break;
            }
            Ok(input::ReaderEvent::Completed) => {
                match input_reader.current_input.len()
                    == input_reader.current_input.matches('0').count()
                {
                    true => {
                        //empty or all zeroes
                        println!();
                        println!("    ...testing default device confirmed");
                    }
                    false => match input_reader.current_input.parse::<isize>() {
                        Ok(parsed_idx) => {
                            device_test_index = Some(parsed_idx);
                            println!();
                        }
                        Err(_) => {
                            input_reader.current_input.clear();
                            continue;
                        }
                    },
                }
                break;
            }
            Ok(input::ReaderEvent::Timeout) => {} //just redraw prompt
            Err(e) => {
                //if input machinery doesnt work treat it as nothing was enetred
                if verbose {
                    println!("Input machinery failure: {e}");
                }
                break;
            }
        }
    }
    drop(input_reader);
    device_test_index
}
struct TestStatus {
    test_status: u8,
}
fn test_in_this_process<Writer: std::io::Write>(
    instance: &erupt::InstanceLoader,
    selected: &NamedComputeDevice,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    if !env.mem_budget_estimated() {
        display_this_process_result(Some("Failed determining memory budget".into()), env)
    }

    prepare_and_test_device(instance, selected, env, log_dupler)
}

enum SubprocessMode {
    NotExeced,
    FailedRetryLowerMemory,
    DoneOrFailedNoretry,
}
fn test_selected_label<Writer: std::io::Write>(
    loaded_devices: LoadedDevices,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<(Option<LoadedDevices>, TestStatus), Box<dyn std::error::Error>> {
    // prevent running drop on loaded_devices

    match loaded_devices.devices.as_slice() {
        [single_dev] => {
            if env.interactive {
                let mut mode;
                let mut main_code: u8 = 0;
                loop {
                    mode = SubprocessMode::NotExeced;
                    if let Some(argv0) = &env.argv0
                        && let Ok(mut child) = std::process::Command::new(argv0)
                            .arg((-1 - (single_dev.index_in_device_iteration as isize)).to_string())
                            .arg(env.max_test_bytes.load(SeqCst).to_string())
                            .spawn()
                    {
                        if env.verbose() {
                            let _ = writeln!(
                                log_dupler,
                                "Spawned child {child:?} with PID {}",
                                child.id()
                            );
                        }
                        let wait_result = child.wait();
                        let parent_close_requested = close::close_requested();
                        match wait_result {
                            Err(e) => {
                                let _ = writeln!(
                                    log_dupler,
                                    "wait error: {e}  parent_close_requested: {parent_close_requested}"
                                );
                                return Err("Problem waiting for subprocess".into());
                            }
                            Ok(exit_status) => {
                                if env.verbose() {
                                    let _ = writeln!(
                                        log_dupler,
                                        "Subprocess status {exit_status} parent_close_requested {parent_close_requested}"
                                    );
                                }
                                match exit_status.code() {
                                    None => {
                                        return Err(
                                            "Exit code of test process not available".into()
                                        );
                                    }
                                    Some(subprocess_code) => {
                                        main_code = subprocess_code as u8;
                                        let strange_code = (main_code
                                            & close::app_status::SIGNATURE_MASK)
                                            != close::app_status::SIGNATURE;
                                        if strange_code {
                                            let _ = writeln!(
                                                log_dupler,
                                                "Unexpected code {subprocess_code}"
                                            );
                                            return Err(
                                                "Exit code of test process can't be interpreted"
                                                    .into(),
                                            );
                                        }
                                        if main_code
                                            == (close::app_status::SIGNATURE
                                                | close::app_status::RUNTIME_ABORT)
                                        {
                                            mode = SubprocessMode::FailedRetryLowerMemory;
                                        } else {
                                            mode = SubprocessMode::DoneOrFailedNoretry;
                                            if !parent_close_requested
                                                && !close::check_any_bits_set(
                                                    main_code,
                                                    close::app_status::QUIT_JOB_REQUESTED,
                                                )
                                            {
                                                let _ = writeln!(
                                                    log_dupler,
                                                    "Seems child exited for no reason, code {subprocess_code}"
                                                );
                                                main_code |= close::app_status::RUNTIME_ABORT;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    match mode {
                        SubprocessMode::NotExeced => {
                            let _ =
                                writeln!(log_dupler, "Subprocess testing method not available!");
                            break;
                        }
                        SubprocessMode::FailedRetryLowerMemory => {
                            let tried_bytes = env.max_test_bytes.load(SeqCst);
                            let smaller_memory = tried_bytes - ALLOCATION_TRY_STEP;
                            if smaller_memory < MIN_WANTED_ALLOCATION {
                                let _ = writeln!(
                                    log_dupler,
                                    "Subprocess testing method failed even with small memory limit {tried_bytes}"
                                );
                                break;
                            }
                            env.set_mem_budget_limit(smaller_memory);
                            if env.verbose() {
                                let _ = writeln!(
                                    log_dupler,
                                    "retrying subprocess with smaller memory limit {smaller_memory}"
                                );
                            }
                            continue;
                        }
                        SubprocessMode::DoneOrFailedNoretry => {
                            return Ok((
                                Some(loaded_devices),
                                TestStatus {
                                    test_status: main_code
                                        | (close::fetch_status()
                                            & close::app_status::QUIT_JOB_REQUESTED),
                                },
                            ));
                        }
                    }
                }
                env.make_verbose();
                let _ = writeln!(log_dupler, "Falled back to verbose in-process test method");
            }

            test_in_this_process(loaded_devices.instance(), single_dev, env, log_dupler);
        }
        _ => {
            env.make_verbose();
            Err("While asking selected device exactly 1 device need to present and already selected".into())
        }
    }
}
fn init_vk_and_check_errors<Writer: std::io::Write>(
    mut loaded_devices: LoadedDevices,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<(Option<LoadedDevices>, TestStatus), Box<dyn std::error::Error>> {
    if env.interactive {
        let _ = writeln!(log_dupler,);
        for desc in loaded_devices.devices.iter() {
            let _ = writeln!(log_dupler, "{}", desc.label);
        }
        if loaded_devices.devices.len() > 1 {
            if let Some(selected_label) = prompt_for_label(env.verbose()) {
                env.device_label.store(selected_label, SeqCst);
            } else {
                return Err("Test cancelled, no device selected".into());
            }
        }
        close::setup_handler(true); //for interactive environments setup handler only after input prompt was run
    }

    let index_from_1 = env.effective_index_in_loaded_devices() + 1;
    let available_len = loaded_devices.devices.len();
    let selected = match std::mem::take(&mut loaded_devices.devices)
        .into_iter()
        .nth(index_from_1 - 1)
    {
        None => {
            env.make_verbose();
            return Err(format!("Specified device index ({index_from_1}) greater then available device count: {available_len}").into());
        }
        Some(selected) => selected,
    };

    try_fill_default_mem_budget(&selected, env, log_dupler);
    loaded_devices.devices = vec![selected];

    test_selected_label(loaded_devices, env, log_dupler)
}

#[derive(Default)]
struct ProcessEnv {
    argv0: Option<OsString>,
    device_label: AtomicIsize, //0 = autoselect, >0 index as displayed&sorted by memtest_vulkan, <0 -index as reported by device enumeration
    max_test_bytes: AtomicI64,
    verbose_requested: AtomicBool,
    interactive: bool,
}

impl ProcessEnv {
    fn effective_index_in_loaded_devices(&self) -> usize {
        // converts internal represntation to always-nonnegative index
        // 0 or nagative value means that selection is done at the moment of filling loaded_devices
        // so report selected index if positive and 0 otherwise
        (self.device_label.load(SeqCst) - 1).try_into().unwrap_or(0)
    }

    fn requested_index_in_device_iteration(&self) -> Option<usize> {
        // convert negative value to an index, zero or positive values convert to None
        (-1 - self.device_label.load(SeqCst)).try_into().ok()
    }

    fn verbose(&self) -> bool {
        self.verbose_requested.load(SeqCst)
    }

    fn make_verbose(&self) {
        self.verbose_requested.store(true, SeqCst)
    }

    fn mem_budget_estimated(&self) -> bool {
        self.max_test_bytes.load(SeqCst) > 0
    }

    fn set_mem_budget_limit(&self, new_limit: i64) {
        if self.mem_budget_estimated() {
            let old_limit = self.max_test_bytes.fetch_min(new_limit, SeqCst);
            assert!(old_limit >= new_limit);
        } else {
            let old_limit = self.max_test_bytes.swap(new_limit, SeqCst);
            assert!(old_limit == 0);
        }
    }
}

fn init_running_env() -> ProcessEnv {
    let mut process_env = ProcessEnv {
        verbose_requested: AtomicBool::new(cfg!(feature = "verbose")),
        ..Default::default()
    };
    let mut args_os_iter = std::env::args_os();
    if let Some(argv0) = args_os_iter.next() {
        if let Some(file_stem) = std::path::PathBuf::from(&argv0)
            .file_stem()
            .and_then(|os_str| os_str.to_str())
            && file_stem.to_ascii_lowercase().contains("verbose")
        {
            process_env.make_verbose();
        }
        process_env.argv0 = Some(argv0);
        process_env.interactive = true;
        if let Some(argv1_label_str) = args_os_iter
            .next()
            .as_ref()
            .and_then(|os_str| os_str.to_str())
        {
            if let Ok(label_parsed) = argv1_label_str.parse::<isize>() {
                process_env.interactive = false;
                process_env.device_label.store(label_parsed, SeqCst);
            }
            if let Some(argv2_mem_max_str) = args_os_iter
                .next()
                .as_ref()
                .and_then(|os_str| os_str.to_str())
                && let Ok(mem_max_parsed) = argv2_mem_max_str.parse::<i64>()
            {
                process_env.set_mem_budget_limit(mem_max_parsed);
            }
        }
    }
    if process_env.interactive {
        print!("https://github.com/GpuZelenograd/");
        let _ = std::io::stdout().flush();
        let mut color_setter = input::Reader::default();
        color_setter.set_pass_fail_accent_color(false);
        println!(
            "memtest_vulkan v{} by GpuZelenograd",
            env!("CARGO_PKG_VERSION")
        );
        drop(color_setter);
        println!("To finish testing use Ctrl+C");
    }
    process_env
}

fn display_this_process_result(
    maybe_err: Option<Box<dyn std::error::Error>>,
    env: &ProcessEnv,
) -> ! {
    if let Some(e) = maybe_err {
        println!("Runtime error: {e}");
        close::raise_status_bit(close::app_status::RUNTIME_ABORT);
    }
    display_testing_outcome(
        TestStatus {
            test_status: close::fetch_status(),
        },
        env,
    )
}

fn display_testing_outcome(test_status: TestStatus, env: &ProcessEnv) -> ! {
    if !env.interactive {
        close::immediate_exit(false);
    }
    println!();
    let mut key_reader = input::Reader::default();
    let status = test_status.test_status;
    if close::check_any_bits_set(status, close::app_status::QUIT_JOB_REQUESTED) {
        //propagate closing flag to this process, so no risky-during close functions would be used
        close::raise_status_bit(close::app_status::QUIT_JOB_REQUESTED);
    }
    if !close::check_any_bits_set(status, close::app_status::INITED_OK) {
        println!("memtest_vulkan: INIT OR FIRST testing failed due to runtime error");
    } else if close::check_any_bits_set(status, close::app_status::RUNTIME_ABORT)
        && !close::check_any_bits_set(status, close::app_status::RUNTIME_ERRORS)
    {
        println!("memtest_vulkan: First test passed, but THEN runtime error occured");
    } else {
        let has_errors = close::check_any_bits_set(status, close::app_status::RUNTIME_ERRORS);
        if env.interactive {
            key_reader.set_pass_fail_accent_color(has_errors);
        }
        match has_errors {
            true => println!("memtest_vulkan: memory/gpu ERRORS FOUND, testing finished."),
            false => println!("memtest_vulkan: no any errors, testing PASSed."),
        }
    }
    if env.interactive {
        key_reader.wait_any_key(true);
    }
    drop(key_reader); //restore terminal state before exiting
    close::immediate_exit(false)
}

fn display_result<Writer: std::io::Write>(
    result: Result<(Option<LoadedDevices>, TestStatus), Box<dyn std::error::Error>>,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    match result {
        Ok((_, test_status)) => {
            let _ = log_dupler.flush();
            display_testing_outcome(test_status, env)
        }
        Err(e) => {
            if env.interactive || env.verbose() {
                println!();
                //make more readable message about libvulkan
                let mut error_fmt = format!("{e}");
                let _ = writeln!(
                    log_dupler,
                    "memtest_vulkan early exit during init:\n{error_fmt}"
                );
                let mut source = e.source();
                while let Some(inner) = source {
                    let inner_fmt = format!("{inner}");
                    //exclude duplicate/contained errors
                    if !error_fmt.contains(&inner_fmt) {
                        let _ = writeln!(log_dupler, "    due to: {inner_fmt}");
                        error_fmt = inner_fmt;
                    }
                    source = inner.source();
                }
                let _ = log_dupler.flush();
                if env.interactive {
                    let mut key_reader = input::Reader::default();
                    key_reader.wait_any_key(true);
                    drop(key_reader); //restore terminal state before exiting
                }
            }
            close::immediate_exit(false)
        }
    }
}

fn main() {
    let env = init_running_env();
    if !env.interactive {
        close::setup_handler(false);
    }
    const MAX_LOG_SIZE: u64 = 50 * 1024 * 1024;
    //log is put in current directory. This is intentional - run from other dir to use another log
    let mut log_dupler = output::LogDupler::new(
        std::io::stdout(),
        Some("memtest_vulkan.log".into()),
        MAX_LOG_SIZE,
        if env.interactive {
            "Tester console"
        } else {
            "Tester worker"
        },
    );
    let result = list_devices_ordered_labaled_from_1(&env, &mut log_dupler)
        .and_then(|loaded_devices| init_vk_and_check_errors(loaded_devices, &env, &mut log_dupler));
    display_result(result, &env, &mut log_dupler);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prbs7_bits() -> Vec<u32> {
        // x^7 + x^6 + 1, all-ones seed
        let mut state = 0x7Fu32;
        (0..127)
            .map(|_| {
                let out = state & 1;
                let feedback = ((state >> 6) ^ (state >> 5)) & 1;
                state = ((state << 1) | feedback) & 0x7F;
                out
            })
            .collect()
    }

    #[test]
    fn prbs7_windows_match_lfsr() {
        let bits = prbs7_bits();
        for n in 0..127u32 {
            let expected = (0..32).fold(0u32, |acc, j| acc | bits[((n + j) % 127) as usize] << j);
            assert_eq!(prbs7_window(n), expected, "window at {n}");
        }
    }

    #[test]
    fn addr_rotate_matches_original_formula() {
        for (e, p) in [
            (0u32, 0u32),
            (4, 0x100107),
            (5, 0xFFFF_FFF0),
            (123_456, 0x8708AB91),
        ] {
            let a = e.wrapping_add(p).wrapping_add(1);
            assert_eq!(
                pattern_u32(e, p, PATTERN_ADDR_ROTATE),
                a.rotate_left(a % 31)
            );
        }
    }

    #[test]
    fn toggle_bytes_survive_dbi_dc() {
        // DBI-DC inverts bytes with more than 4 zeros; toggle patterns must keep exactly 4
        for variant in 0..15u32 {
            for e in 0..64 {
                let value = pattern_u32(e, 0, PATTERN_TOGGLE | (variant << 8));
                for byte in value.to_le_bytes() {
                    assert_eq!(
                        byte.count_zeros(),
                        4,
                        "variant {variant} word {e}: {value:08X}"
                    );
                }
            }
        }
    }

    #[test]
    fn consecutive_iterations_change_every_window_pattern() {
        for iter in 1..200u32 {
            for window in 0..8u32 {
                let now = pattern_for(iter, window, true);
                let next = pattern_for(iter + 1, window, true);
                assert_ne!(now & 0xFF, next & 0xFF);
                assert_ne!(now & 0xFF, pattern_for(iter, window + 1, true) & 0xFF);
            }
        }
        assert_eq!(pattern_for(7, 3, false), PATTERN_ADDR_ROTATE);
    }

    #[test]
    fn all_ones_words_count_as_errors() {
        let mut buf = IOBuf::for_initial_iteration();
        buf.done_iter_or_err = 0xFFFF_FFFF;
        buf.idx_min = 0;
        buf.idx_max = 3;
        buf.actual_ff = 4;
        buf.mem_bitcount.0[17] = 1;
        let (_, total) = buf.get_error_addresses_and_count(0).unwrap();
        assert_eq!(total, 5);
    }

    #[test]
    fn march_schedule_covers_every_family_on_every_window() {
        // the old "next window every pass" schedule missed half the families with 3, 4, 5 or 7 windows
        let passes = 6 * 3 * 7 * 2;
        for window_count in 2..=8i64 {
            for static_window in 0..window_count {
                for march_every in 1..=3u32 {
                    let mut seen =
                        vec![[false; PATTERN_FAMILY_COUNT as usize]; window_count as usize];
                    let mut march_passes = 0;
                    for iteration in 1..=passes {
                        if let Some(window) =
                            march_window_for(iteration, window_count, static_window, march_every)
                        {
                            assert_ne!(window, static_window);
                            let family = pattern_for(iteration, window as u32, true) & 0xFF;
                            seen[window as usize][family as usize] = true;
                            march_passes += 1;
                        }
                    }
                    for window in (0..window_count).filter(|w| *w != static_window) {
                        assert!(
                            seen[window as usize].iter().all(|s| *s),
                            "{window_count} windows, static {static_window}, every {march_every}: window {window} saw {:?}",
                            seen[window as usize]
                        );
                    }
                    assert_eq!(march_passes * march_every, passes);
                }
            }
        }
        assert_eq!(march_window_for(5, 4, 0, 0), None);
    }

    #[test]
    fn march_dispatch_covers_whole_windows() {
        // windows are multiples of TEST_WINDOW_SIZE_GRANULARITY; each March dispatch row covers
        // TEST_WINDOW_1D_MAX_GROUPS threads of MARCH_VEC4_PER_THREAD vec4 each
        let vec4_per_granule = TEST_WINDOW_SIZE_GRANULARITY / (ELEMENT_SIZE * VEC_SIZE as i64);
        let vec4_per_row = MARCH_VEC4_PER_THREAD as i64 * TEST_WINDOW_1D_MAX_GROUPS;
        assert_eq!(vec4_per_granule % vec4_per_row, 0);
    }

    fn feed(tracker: &mut BandwidthTracker, kind: KernelKind, dispatch_gbps: &[f64]) {
        for gbps in dispatch_gbps {
            tracker.add(kind, PATTERN_RANDOM, (gbps * GB as f64) as i64, Some(1.0));
        }
    }

    #[test]
    fn bandwidth_drop_needs_two_comparable_low_intervals_in_a_row() {
        let mut tracker = BandwidthTracker::default();
        let (steady, low) = ([700.0; 10], [640.0; 10]);
        let mut warned = Vec::new();
        for (samples, comparable) in [
            (&steady, false),
            (&steady, true),
            (&low, true),
            (&low, true),
            (&steady, true),
            (&low, true),
            (&low, false), // a skipped interval breaks the streak
            (&low, true),
            (&low, true),
        ] {
            feed(&mut tracker, KernelKind::Read, samples);
            warned.push(!tracker.finish_interval(comparable).is_empty());
        }
        assert_eq!(
            warned,
            [false, false, false, true, false, false, false, false, true]
        );
    }

    #[test]
    fn bandwidth_median_ignores_a_few_slow_dispatches() {
        let mut tracker = BandwidthTracker::default();
        feed(&mut tracker, KernelKind::Write, &[700.0; 10]);
        assert!(tracker.finish_interval(true).is_empty());
        let mut disturbed = vec![700.0; 8];
        disturbed.extend([200.0; 3]); // other GPU work slowed 3 of 11 dispatches
        for _ in 0..3 {
            feed(&mut tracker, KernelKind::Write, &disturbed);
            assert!(tracker.finish_interval(true).is_empty());
        }
    }

    #[test]
    fn bandwidth_ignores_intervals_with_too_few_dispatches() {
        let mut tracker = BandwidthTracker::default();
        feed(&mut tracker, KernelKind::March, &[700.0; 10]);
        assert!(tracker.finish_interval(true).is_empty());
        for _ in 0..3 {
            feed(
                &mut tracker,
                KernelKind::March,
                &[300.0; BANDWIDTH_MIN_DISPATCHES - 1],
            );
            assert!(tracker.finish_interval(true).is_empty());
        }
    }

    fn clock_stats(samples: &[(u32, u64)]) -> telemetry::IntervalStats {
        let mut stats = telemetry::IntervalStats::default();
        for &(mem_clock, reasons) in samples {
            stats.add(telemetry::Sample {
                mem_clock_mhz: Some(mem_clock),
                limit_reasons: Some(reasons),
                ..Default::default()
            });
        }
        stats
    }

    #[test]
    fn bandwidth_warning_names_the_likely_cause() {
        let drop = [BandwidthDrop {
            kind: KernelKind::Read,
            family: PATTERN_PRBS7,
            gbps: 600.0,
            best_gbps: 700.0,
        }];
        assert!(bandwidth_warning(&[], None).is_none());
        let warning =
            |stats: Option<&telemetry::IntervalStats>| bandwidth_warning(&drop, stats).unwrap();
        assert!(warning(None).contains("read/prbs7 600.0GiB/s (-14.3%)"));
        // GpuIdle (0x1) is not a clock limit
        let steady = clock_stats(&[(13001, 0x1), (13001, 0)]);
        assert!(warning(Some(&steady)).contains("stayed at 13001MHz with no clock limiting"));
        let limited = clock_stats(&[(13001, 0x4), (13001, 0x40)]);
        let limited_warning = warning(Some(&limited));
        assert!(limited_warning.contains("stayed at 13001MHz, but"));
        assert!(limited_warning.contains("(SwPowerCap,HwThermal)"));
        // a varying memory clock explains a drop even when limits are also active
        let varied = clock_stats(&[(13001, 0x4), (10501, 0)]);
        assert!(warning(Some(&varied)).contains("varied 10501-13001MHz"));
    }

    #[test]
    fn telemetry_status_text_shows_interval_peaks() {
        let mut stats = telemetry::IntervalStats::default();
        for (core, temperature, power) in [(2790, 55, 240.4), (2775, 58, 263.2), (2775, 57, 250.0)]
        {
            stats.add(telemetry::Sample {
                mem_clock_mhz: Some(13001),
                gpu_clock_mhz: Some(core),
                temperature_c: Some(temperature),
                power_w: Some(power),
                limit_reasons: Some(0),
                ecc_errors: None,
            });
        }
        assert_eq!(
            stats.status_text(),
            " mem:13001MHz core:2775MHz max 58C 263W"
        );
        assert_eq!(telemetry::IntervalStats::default().status_text(), "");
    }

    #[test]
    fn iobuf_layout_matches_wgsl() {
        // WGSL aligns the trailing vec4<u32> to 16 bytes
        assert_eq!(mem::offset_of!(IOBuf, first_elem), 432);
        assert_eq!(mem::size_of::<IOBuf>(), 448);
    }
}
