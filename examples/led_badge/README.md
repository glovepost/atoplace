## LED Badge (ESP32‑C3 + 10×10 Addressable Matrix)

A hackable, battery‑powered LED badge built with Atopile for hardware and PlatformIO/Arduino for firmware. It features a 10×10 SK6805 LED matrix, an ESP32‑C3 module, USB‑C power/data, on‑board charging and regulation, a 6‑axis IMU, an I²S microphone, and a user button. Press the button to cycle through multiple visual/audio‑reactive modes.

![LED Badge demo](./badge.gif)

### Highlights

- **MCU**: `ESP32-C3-MINI-1`
- **LEDs**: `SK6805-EC20` 10×10 addressable matrix (100 pixels)
- **Power**: USB‑C, `BQ25185` Li‑ion charger + power path, `TPS63020` 3V3 buck‑boost
- **Sensors**: `LSM6DS3` IMU (accel + gyro), `ICS‑43434` I²S microphone
- **I/O**: 1x user button, USB‑CDC serial console at 115200 baud

### Repository layout

- `led_badge.ato` — Hardware design (Ato/Atopile)
- `ato.yaml` — Build targets and package dependencies
- `layouts/` — Generated KiCad projects for different form factors
  - `badge/` (full badge), `grid100x100/`, `strip10/`, `esp32_minimal/`
- `parts/` — Atomic parts (symbols/footprints/STEP)
- `firmware/` — PlatformIO project (Arduino framework)

### Hardware architecture (Ato)

The badge is defined in `LED_BADGE` and uses Atopile packages from the registry:

- ESP32‑C3 module, USB‑C connector
- `BQ25185` charger → battery → system power → `TPS63020` buck‑boost → 3V3 rail
- 10×10 SK6805 matrix, LSM6DS3 IMU over I²C, ICS‑43434 microphone over I²S

Build targets (from `ato.yaml`):

- **badge** → `led_badge.ato:LED_BADGE`
- **grid100x100** → `led_badge.ato:SK6805EC20_grid100x100`
- **strip10** → `led_badge.ato:SK6805EC20_strip10`

### Building the PCB

- Open the generated KiCad project at `layouts/badge/badge.kicad_pro` to view/modify the PCB.
- To regenerate from Ato sources, use the Atopile extension (Cursor/VS Code) and run the build for the `badge` target. This will refresh the KiCad files under `layouts/`.

### Firmware (PlatformIO/Arduino)

The firmware lives in `firmware/`. It targets `esp32-c3-devkitm-1` and uses:

- `Adafruit NeoPixel` (LEDs)
- `Adafruit LSM6DS` + `Adafruit AHRS` (IMU fusion)
- `arduinoFFT` (audio spectral analysis)

Quick start:

1. Install the PlatformIO extension for VS Code (or `pipx install platformio`).
2. Open the `firmware/` folder as a PlatformIO project.
3. Connect the badge over USB‑C.
4. Build and upload to the `esp32-c3-devkitm-1` environment.

CLI equivalents:

```bash
cd firmware
pio run                      # build
pio run -t upload            # flash
pio device monitor -b 115200 # serial console
```

### Pin map (logical)

- **LED data**: `GPIO8`
- **Button**: `GPIO9` (INPUT_PULLUP; press = LOW)
- **I²C**: `SDA GPIO5`, `SCL GPIO6`
- **I²S mic (ICS‑43434)**: `SCK/BCK GPIO0`, `WS/LRCLK GPIO3`, `SD GPIO1`

### Run modes (press button to cycle)

- **Rainbow**: ambient dissolve across the matrix
- **Ball**: physics‑like ball steered by tilt (IMU)
- **Life**: Conway’s Game of Life with age‑based coloring
- **Sweep**: moving highlight across pixels
- **Center**: static center block
- **Multi‑ball**: 3 bouncing balls
- **Level**: electronic bubble level using IMU gravity vector
- **Nyan**: placeholder for simple 4‑frame animation
- **Spectrogram**: scrolling FFT spectrogram from the I²S mic
- **Vertical line**: left‑edge test pattern
- **Audio levels**: 10‑band logarithmic histogram
- **Beat flash**: bass‑beat detection flashes the matrix

Tip: open the serial monitor at 115200 to see I²C scan results, mic/FFT debug, and mode logs.

### Flashing notes (ESP32‑C3)

- If auto‑upload fails, hold the module’s BOOT button, tap RESET, then release BOOT to enter download mode and retry.
- USB‑CDC is enabled by default; the serial port appears without external UART hardware.

### Customizing

- LED matrix size and chaining are defined in `led_badge.ato` (`SK6805EC20_grid100x100` and `SK6805EC20_strip10`).
- Power rail limits (USB 5V, charger currents, 3V3 regulation) are constrained in `LED_BADGE` and can be adjusted as needed.

### License

This project is provided under the terms in `LICENSE.txt`.

### Credits

- Hardware is defined in Ato using packages from the Atopile registry.
- Firmware builds on libraries from Adafruit and the Arduino ecosystem.
