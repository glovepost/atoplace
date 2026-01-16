// Centralized helpers for workout schedule logic
#pragma once

struct ScheduleConfig {
  int c1_h;
  int c1_m;
  int c2_h;
  int c2_m;
  int slot_min;
};

inline void advance_fake_datetime(int &hour, int &minute, int &second,
                                 int &day_of_week /*1-7*/, int &month /*1-12*/, int &day_of_month /*1-31*/,
                                 int add_seconds) {
  // Compute seconds since midnight
  long long day_seconds = static_cast<long long>(hour) * 3600LL + static_cast<long long>(minute) * 60LL + second;
  long long total = day_seconds + static_cast<long long>(add_seconds);
  long long days_added = total / 86400LL;
  long long rem = total % 86400LL;
  if (rem < 0) { rem += 86400LL; --days_added; }

  hour = static_cast<int>(rem / 3600LL);
  rem %= 3600LL;
  minute = static_cast<int>(rem / 60LL);
  second = static_cast<int>(rem % 60LL);

  if (days_added != 0) {
    // Advance day-of-week 1..7
    int dow0 = (day_of_week - 1 + static_cast<int>(days_added)) % 7;
    if (dow0 < 0) dow0 += 7;
    day_of_week = dow0 + 1;

    // Advance calendar day with simple month lengths (no leap year)
    static const int mdays[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    int d = day_of_month;
    int m = month;
    long long add = days_added;
    while (add > 0) {
      int dim = mdays[(m - 1) % 12];
      ++d;
      if (d > dim) { d = 1; ++m; if (m > 12) m = 1; }
      --add;
    }
    while (add < 0) {
      --d;
      if (d < 1) {
        --m; if (m < 1) m = 12;
        int dim = mdays[(m - 1) % 12];
        d = dim;
      }
      ++add;
    }
    day_of_month = d;
    month = m;
  }
}

inline int get_circuit_idx(int hour, int minute, const ScheduleConfig &cfg) {
  const int total = hour * 60 + minute;
  const int c2_start = cfg.c2_h * 60 + cfg.c2_m;
  return total >= c2_start ? 1 : 0;
}

inline int day_offset_for_rotation(int dow /*1..7*/) {
  // Mon=2 -> 0, Tue=3 -> 1, Wed=4 -> 2, Thu=5 -> 0, Fri=6 -> 1, Sat=7 -> 2, Sun=1 -> 0
  if (dow == 1) return 0; // Sunday
  return (dow - 2) % 3;
}

inline int get_workout_idx(int hour, int minute, int circuit_idx, int day_of_week, const ScheduleConfig &cfg) {
  const int total = hour * 60 + minute;
  const int start = (circuit_idx == 0) ? (cfg.c1_h * 60 + cfg.c1_m) : (cfg.c2_h * 60 + cfg.c2_m);
  const int delta = total - start;
  int idx_by_time;
  if (delta < cfg.slot_min) idx_by_time = 0;
  else if (delta < 2 * cfg.slot_min) idx_by_time = 1;
  else idx_by_time = 2;
  int off = day_offset_for_rotation(day_of_week);
  return (idx_by_time + off) % 3;
}

inline int minute_hand_value(int minute) { return minute; }
inline int hour_hand_value(int hour, int minute) { return (hour % 12) * 60 + minute; }

inline std::string build_title(int circuit_idx, int workout_idx) {
  char buf[32];
  snprintf(buf, sizeof(buf), "Circuit %d Workout %d", circuit_idx + 1, workout_idx + 1);
  return std::string(buf);
}

inline std::string build_date_with_day(int dow, int month, int day_of_month) {
  static const char *const mon_names[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  static const char *const day_names[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
  int d = dow; if (d < 1) d = 1; if (d > 7) d = 7;
  int m = month; if (m < 1) m = 1; if (m > 12) m = 12;
  char buf[24];
  snprintf(buf, sizeof(buf), "%s %s %2d", day_names[d - 1], mon_names[m - 1], day_of_month);
  return std::string(buf);
}

inline const char *get_workout_name(
    int circuit_idx, int idx,
    const char *c1w1, const char *c1w2, const char *c1w3,
    const char *c2w1, const char *c2w2, const char *c2w3) {
  if (circuit_idx == 0) {
    switch (idx % 3) {
      case 0: return c1w1;
      case 1: return c1w2;
      default: return c1w3;
    }
  } else {
    switch (idx % 3) {
      case 0: return c2w1;
      case 1: return c2w2;
      default: return c2w3;
    }
  }
}
