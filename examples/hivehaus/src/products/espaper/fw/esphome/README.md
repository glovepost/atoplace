ESPHome ePaper Workout Display

Overview
- ESPHome project for an ESP32‑C3 driving a Waveshare 7.5" e‑paper panel.
- LVGL renders an analog clock and a rotating workout schedule.
- Scheduling rules live in C++ helpers; editable workout data lives in a single YAML file.

Project Layout
- `workout.yaml`: Main ESPHome config
  - Includes helpers via `esphome.includes: schedules/workout_logic.h`.
  - Imports workout data with `substitutions: !include schedules/circuit_unified.yaml`.
  - Defines the LVGL UI inline (clock meter, date label, workout/participant labels).
  - Minimal lambdas call helper functions to update LVGL.
- `schedules/circuit_unified.yaml`: Single source of truth for data
  - Timing: `c1_start_hour`, `c1_start_minute`, `c2_start_hour`, `c2_start_minute`, `slot_minutes`.
  - Workouts: `circuit1_w1..w3`, `circuit2_w1..w3`.
  - Participants: `participants_1..3` (supports `\n` for multi‑line names).
- `schedules/workout_logic.h`: C++ helpers (compiled via `esphome.includes`)
  - `advance_fake_datetime(h, m, s, dow, mon, dom, add_seconds)` — advances time/date.
  - `get_circuit_idx(hour, minute, cfg)` — selects morning/afternoon circuit.
  - `get_workout_idx(hour, minute, circuit_idx, cfg)` — selects which workout (0..2).
  - `get_workout_name(circuit_idx, idx, c1w1..c1w3, c2w1..c2w3)` — resolves workout name.
  - `minute_hand_value(minute)`, `hour_hand_value(hour, minute)` — analog clock values.
  - `build_title(circuit_idx, workout_idx)`, `build_date_with_day(dow, mon, dom)` — UI text.

Behavior
- Fake time: A 1s interval advances time by +8 minutes to demonstrate rotation.
  - Updates `fake_hour`, `fake_minute`, `fake_second`, and also `fake_day_of_week`, `fake_month`, `fake_day_of_month`.
- Circuits and rotation:
  - Circuit 1 runs from `c1_start_*`; Circuit 2 from `c2_start_*`.
  - Each workout slot is `slot_minutes` long; three slots per circuit (indexes 0..2).
  - Day-based rotation offset: the starting workout shifts by day-of-week:
    - Mon: 1,2,3; Tue: 2,3,1; Wed: 3,1,2; Thu: 1,2,3; Fri: 2,3,1; Sat: 3,1,2; Sun: 1,2,3.
    - Implemented in `day_offset_for_rotation(dow)` and applied by `get_workout_idx(...)`.
- UI details:
  - Date label shows “DOW MON DD” above the clock (e.g., `MON JAN  1`).
  - Three participant columns with corresponding workout names that rotate hourly.

Editing Data
- Change only `schedules/circuit_unified.yaml` to update timing, workouts, or participants.
- No code changes are needed for data edits.

Build and Flash
- Board: `esp32-c3-devkitm-1`; Framework: ESP‑IDF.
- Typical commands:
  - `esphome run workout.yaml` (local build/flash) or use your IDE’s ESPHome integration.
- OTA and Wi‑Fi credentials are set in `workout.yaml`.

Extending
- Add more circuits or slots: extend the data schema and update helpers in `workout_logic.h`.
- Replace fake time with real time: add an ESPHome `time:` component and trigger `time_update` each minute; remove `fake_time_update`.
 - Rotation rules: adjust `day_offset_for_rotation(dow)` in `schedules/workout_logic.h` if you want a different weekly pattern. Default starts week on Monday (fake clock uses `fake_day_of_week=2`).

Notes
- YAML strings with newlines are passed to LVGL using C++ raw string literals to avoid escaping issues.
- The layout was kept inline in `workout.yaml` to avoid duplication; only workout data is externalized.

Use Real Time Instead of Fake Time
- Add an SNTP time source and update once per minute. Comment out or remove the `interval:` block that calls `fake_time_update`, then add:

```
time:
  - platform: sntp
    id: sntp_time
    timezone: America/Los_Angeles  # set yours
    on_time:
      - seconds: 0
        minutes: "*"
        then:
          - lambda: |-
              auto now = id(sntp_time).now();
              if (!now.is_valid()) return;
              // Push real time into the existing globals used by the UI
              id(fake_hour) = now.hour;
              id(fake_minute) = now.minute;
              id(fake_second) = now.second;
              id(fake_day_of_week) = now.day_of_week;   // 1..7
              id(fake_month) = now.month;               // 1..12
              id(fake_day_of_month) = now.day_of_month; // 1..31
          - script.execute: time_update
          - script.execute: update_workout_display
```

- This keeps all downstream logic unchanged and switches the source of truth to real time.
