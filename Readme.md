# memtest_vulkan (fork)

memtest_vulkan tests the video memory (VRAM) of a graphics card. The test operates on the GPU with Vulkan compute shaders. The CPU does not do the test work.

Use this tool to find memory errors after you overclock a graphics card or after a repair.

This repository is a fork of [GpuZelenograd/memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan). The fork adds more test algorithms and sensor data from NVIDIA graphics cards.

## What the tool tests

The tool fills the free video memory with data. Then it reads the data and compares it with the correct values. It repeats these steps until you stop it.

The fork adds these tests to the original test:

- **Data patterns:** The tool uses six pattern families in a sequence: address, random, solid, toggle, walking and PRBS7. Each family puts a different load on the memory chips and on the data bus.
- **March C- test:** This algorithm writes and reads each memory cell in a specified sequence. It finds address errors and errors between adjacent cells.
- **Retention test:** One memory window keeps its data for 60 seconds. The tool reads this window again on each pass. This test finds cells that do not keep their data.
- **Speed monitor:** The tool measures the speed of each pass with GPU timestamps. If the speed decreases by more than 5%, the tool shows a warning and the possible cause.
- **NVIDIA sensor data:** The tool shows the memory temperature, the memory clock, the power and the clock limits. If the ECC counter of the driver increases, the tool records a memory error.

To use the original test of memtest_vulkan, refer to [Settings](#settings).

## Requirements

- A 64-bit Windows or Linux computer.
- A GPU with a Vulkan 1.1 driver. The graphics driver usually installs Vulkan.
- For the sensor data: an NVIDIA GPU with the NVIDIA driver.
- For the memory temperature: Windows and an NVIDIA card with a memory temperature sensor, for example a card with GDDR6X or GDDR7 memory.

## Get the program

### Download

Download the file for your computer from the [v0.6.0 release](https://github.com/grindig/memtest_vulkan/releases/tag/v0.6.0).

| File | Use it on |
| --- | --- |
| `memtest_vulkan-v0.6.0.exe` | 64-bit Windows 7 or later. |
| `memtest_vulkan-v0.6.0-alt-experimental.exe` | 64-bit Windows 10 or later. Use this file if the other `.exe` file does not start. |
| `memtest_vulkan-v0.6.0_DesktopLinux_X86_64.tar.xz` | 64-bit Linux on x86_64 computers. |
| `memtest_vulkan-v0.6.0_EmbeddedLinux_AARCH64.tar.xz` | 64-bit Linux on ARM computers, for example NVIDIA Jetson or Raspberry Pi 4. |

The Windows files are ready to use. On Linux, extract the file before you start the program:

```bash
tar -xf memtest_vulkan-v0.6.0_DesktopLinux_X86_64.tar.xz
```

The release page also shows the SHA-256 checksums of the files.

You can also build the program from the source code. Refer to the steps that follow.

### Build on Windows

1. Install Rust:
   ```bash
   winget install Rustlang.Rustup
   ```
2. Install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload.
3. Open a new terminal.
4. Get the source code:
   ```bash
   git clone https://github.com/grindig/memtest_vulkan
   ```
5. Go to the source folder:
   ```bash
   cd memtest_vulkan
   ```
6. Build the program:
   ```bash
   cargo build --release
   ```

The build makes the file `target\release\memtest_vulkan.exe`. The first build is slower, because Cargo downloads and compiles all dependencies.

> **NOTE:** The Vulkan SDK is not necessary. The build compiles the shaders, and the program uses the Vulkan library of the graphics driver.

### Build on Linux

1. Install Rust with [rustup](https://rustup.rs).
2. Install a C linker, for example `gcc`.
3. Do steps 3 to 6 of [Build on Windows](#build-on-windows).

The build makes the file `target/release/memtest_vulkan`.

## Start a test

On Windows, double-click the `.exe` file. Administrator rights are not necessary.

On Linux, open a terminal in the folder of the program. Then start the program:

```bash
./memtest_vulkan
```

> **CAUTION:** Do not start the program with a double-click in a Linux file manager. If you do, the test operates in the background and you cannot stop it.

If the computer has more than one GPU, the tool shows a list. Type the number of the GPU to test. If you do not type a number in 10 seconds, the tool selects the first GPU.

### Test phases

1. The standard test operates for approximately 5 minutes.
2. Near the end of the standard test, the load stops for approximately 15 seconds. The memory clock decreases and then increases again. Some errors occur only at a low clock or during a clock change.
3. At the end of the standard test, the tool shows the result: `PASSed` or `ERRORS FOUND`.
4. Then an endless test starts. A test of more than 2 hours is usually not necessary.

To stop the test, push `Ctrl+C`. The tool then shows the final result.

## Read the results

The tool adds a line to the status table every 30 seconds. Example:

```
time          status             write    read   march  memT  memMHz  power  limit          coreT    tested
                                 GiB/s   GiB/s   GiB/s
00:05:36.512  no errors          678.9   697.7   652.4   74C   10501   262W  power            58C  226.4TiB
```

| Column | Shows |
| --- | --- |
| `time` | The time since the start of the test (hours:minutes:seconds.milliseconds). |
| `status` | `no errors`, or `ERR` and the time of the last error. |
| `write` | The write speed. |
| `read` | The read speed. The tool also compares the data in this step. |
| `march` | The March C- speed. A dash (`-`) shows that the tool did not do a March C- test in this interval. |
| `memT` | The highest memory temperature in the interval. |
| `memMHz` | The current memory clock. |
| `power` | The highest power in the interval. |
| `limit` | The clock limits in the interval: `power`, `thermal`, `hwslow` or `brake`. |
| `coreT` | The highest GPU core temperature in the interval. |
| `tested` | The quantity of data that the tool wrote and read since the start. |

The tool shows the columns from `memT` to `coreT` only for NVIDIA cards. If the card has no memory temperature sensor, `coreT` replaces `memT`. If the window is too narrow, the tool removes columns from the right side.

### Errors

If the tool finds an error, it shows an error report immediately. Each error shows a hardware problem, usually in the GPU or in the video memory.

To get help with an error report, refer to the [discussions of the original project](https://github.com/GpuZelenograd/memtest_vulkan/discussions).

## Settings

To change the test, set an environment variable before you start the program.

| Variable | Effect |
| --- | --- |
| `MEMTEST_VULKAN_MODE=classic` | Uses the original test: one pattern and no March C- test. |
| `MEMTEST_VULKAN_MARCH_EVERY=N` | Does the March C- test in 1 of each N blocks of passes. The default is 2. The value 0 stops the March C- test. |
| `MEMTEST_VULKAN_TELEMETRY=0` | Does not read the NVIDIA sensor data. |

Example for the Windows command prompt:

```bat
set MEMTEST_VULKAN_MODE=classic
memtest_vulkan.exe
```

Example for Linux:

```bash
MEMTEST_VULKAN_MODE=classic ./memtest_vulkan
```

To show diagnostic data, rename the program file. The new name must contain `verbose`, for example `memtest_vulkan_verbose.exe`.

## If the test does not start

- **`The library failed to load`:** The Vulkan library is not installed. Install the newest driver for your GPU. On Linux, you can also install the `libvulkan1` package.
- **`ERROR_INCOMPATIBLE_DRIVER` or `ERROR_INITIALIZATION_FAILED`:** There is no Vulkan driver for your GPU. Install the newest driver for your GPU.

For other problems, start the program with a `verbose` file name. Then open an [issue](https://github.com/grindig/memtest_vulkan/issues) and add the output.

## Development

- `cargo test` starts the unit tests.
- `bash run-lint.sh` examines the format and the code, and then starts the unit tests. Before you use it, install the tools: `rustup component add rustfmt clippy`.

## License and credits

The [GpuZelenograd repair center](https://gpuzelenograd.github.io/README?memtest_vulkan) made the original memtest_vulkan. The idea came from [memtestCL](https://github.com/ihaque/memtestCL). The Vulkan bindings come from the [erupt library](https://gitlab.com/Friz64/erupt).

memtest_vulkan uses the [zlib License](LICENSE).
