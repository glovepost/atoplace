#include <stdio.h>
#include <string.h>
#include <time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/usb_serial_jtag.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "main.h"
#include "font5x7.h"

// Assumes these macros exist in main.h:
// SPI_MOSI, SPI_SCLK, DC, RST, BUSY, EPD_WIDTH(800), EPD_HEIGHT(480)

#define LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
static inline uint32_t ms(void){ return (xTaskGetTickCount() * 1000 / configTICK_RATE_HZ); }


static spi_device_handle_t spi;
static bool g_invert_pixels = true; // use native EPD polarity (0xFF = white, 0x00 = black)

static const uint8_t Voltage_Frame_7IN5_V2[] = {
    0x06, 0x3F, 0x3F, 0x11, 0x24, 0x07, 0x17,
};

static const uint8_t LUT_VCOM_7IN5_V2[42] = {
    0x00,0x0F,0x0F,0x00,0x00,0x01, 0x00,0x0F,0x01,0x0F,0x01,0x02,
    0x00,0x0F,0x0F,0x00,0x00,0x01, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,
};

static const uint8_t LUT_WW_7IN5_V2[42] = {
    0x10,0x0F,0x0F,0x00,0x00,0x01, 0x84,0x0F,0x01,0x0F,0x01,0x02,
    0x20,0x0F,0x0F,0x00,0x00,0x01, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,
};

static const uint8_t LUT_BW_7IN5_V2[42] = {
    0x10,0x0F,0x0F,0x00,0x00,0x01, 0x84,0x0F,0x01,0x0F,0x01,0x02,
    0x20,0x0F,0x0F,0x00,0x00,0x01, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,
};

static const uint8_t LUT_WB_7IN5_V2[42] = {
    0x80,0x0F,0x0F,0x00,0x00,0x01, 0x84,0x0F,0x01,0x0F,0x01,0x02,
    0x40,0x0F,0x0F,0x00,0x00,0x01, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,
};

static const uint8_t LUT_BB_7IN5_V2[42] = {
    0x80,0x0F,0x0F,0x00,0x00,0x01, 0x84,0x0F,0x01,0x0F,0x01,0x02,
    0x40,0x0F,0x0F,0x00,0x00,0x01, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,
};

// --- Simple 1bpp framebuffer and 5x7 font ----------------------------------
#define FB_WIDTH   (EPD_WIDTH)
#define FB_HEIGHT  (EPD_HEIGHT)
#define FB_STRIDE  (FB_WIDTH/8)
#define LINE_HEIGHT 8  // 7px font + 1px spacing
#define STATUS_BAR_LINES 6
#define STATUS_BAR_HEIGHT (STATUS_BAR_LINES * LINE_HEIGHT)
#define TERMINAL_LINE_MAX 1024
#define USB_RX_CHUNK 64

static uint8_t g_fb[FB_STRIDE * FB_HEIGHT]; // 48,000 bytes
static int g_line_count = 0;
static int g_last_status_minute = -1;
static int g_last_status_line_count = -1;

static inline void fb_clear_white(void){ memset(g_fb, 0xFF, sizeof(g_fb)); }

static inline void fb_set_pixel(int x,int y,bool black){
    if((unsigned)x>=FB_WIDTH || (unsigned)y>=FB_HEIGHT) return;
    uint8_t *p = &g_fb[y*FB_STRIDE + (x>>3)];
    uint8_t mask = 0x80 >> (x & 7);
    if (black) *p &= ~mask; else *p |= mask; // 1=white, 0=black in framebuffer
}

static void fb_draw_char(int x,int y,char c){
    if(c < 32 || c > 127) c = '?';
    const uint8_t *col = FONT5x7[c-32];
    for(int cx=0; cx<5; ++cx){
        uint8_t bits = col[cx];
        for(int cy=0; cy<7; ++cy){
            bool on = (bits >> cy) & 1; // LSB at top
            fb_set_pixel(x+cx, y+cy, on); // on = black pixel
        }
    }
}

static void fb_draw_char_scaled(int x,int y,char c,int scale){
    if(scale <= 1){ fb_draw_char(x,y,c); return; }
    if(c < 32 || c > 127) c = '?';
    const uint8_t *col = FONT5x7[c-32];
    for(int cx=0; cx<5; ++cx){
        uint8_t bits = col[cx];
        for(int cy=0; cy<7; ++cy){
            if(!((bits >> cy) & 1)) continue;
            const int base_x = x + cx * scale;
            const int base_y = y + cy * scale;
            for(int dx=0; dx<scale; ++dx){
                for(int dy=0; dy<scale; ++dy){
                    fb_set_pixel(base_x + dx, base_y + dy, true);
                }
            }
        }
    }
}

static int fb_draw_text_line_scaled(int x,int y,const char* s,int scale){
    int cursor_x = x;
    const int advance = (5 * scale) + scale;
    while(*s){
        if(*s=='\n') break;
        fb_draw_char_scaled(cursor_x, y, *s++, scale);
        cursor_x += advance;
    }
    return cursor_x;
}

static int fb_draw_text_line(int x,int y,const char* s){
    int cursor_x = x;
    int cursor_y = y;
    while(*s){
        if(*s=='\n'){
            cursor_x = x;
            cursor_y += LINE_HEIGHT;
            ++s;
            if(cursor_y + 7 > FB_HEIGHT) break;
            continue;
        }
        if(cursor_x + 5 > FB_WIDTH){
            cursor_x = x;
            cursor_y += LINE_HEIGHT;
            if(cursor_y + 7 > FB_HEIGHT) break;
        }
        if(cursor_y + 7 > FB_HEIGHT) break;
        fb_draw_char(cursor_x, cursor_y, *s++);
        cursor_x += 6; // 5px glyph + 1px space
    }
    return cursor_y;
}

static void fb_scroll_up(int rows){
    if (rows <= 0) return;
    const int start_row = STATUS_BAR_HEIGHT;
    const int usable_rows = FB_HEIGHT - start_row;
    if (usable_rows <= 0) return;
    if (rows >= usable_rows){
        for (int y = start_row; y < FB_HEIGHT; ++y){
            memset(&g_fb[y * FB_STRIDE], 0xFF, FB_STRIDE);
        }
        return;
    }
    uint8_t *base = &g_fb[start_row * FB_STRIDE];
    const size_t row_bytes = FB_STRIDE;
    const size_t move_bytes = (usable_rows - rows) * row_bytes;
    memmove(base, base + rows * row_bytes, move_bytes);
    memset(base + move_bytes, 0xFF, rows * row_bytes);
}

static void fb_draw_status_bar_content(int seconds_since_boot){
    for (int row = 0; row < STATUS_BAR_HEIGHT; ++row){
        memset(&g_fb[row * FB_STRIDE], 0xFF, FB_STRIDE);
    }

    char big_time[16];
    struct tm tm_now;
    time_t now = 0;
    bool have_time = (time(&now) != (time_t)-1) && localtime_r(&now, &tm_now);
    int hours, minutes, seconds;
    if (have_time){
        hours = tm_now.tm_hour;
        minutes = tm_now.tm_min;
        seconds = 0;
    } else {
        hours = (seconds_since_boot / 3600) % 24;
        minutes = (seconds_since_boot / 60) % 60;
        seconds = 0;
    }
    snprintf(big_time, sizeof(big_time), "%02d:%02d", hours, minutes);

    const int scale = 4;
    const int advance = (5 * scale) + scale;
    const int len = (int)strlen(big_time);
    int total_width = len * advance - (len > 0 ? scale : 0);
    int start_x = (FB_WIDTH - total_width) / 2;
    if (start_x < 0) start_x = 0;
    fb_draw_text_line_scaled(start_x, 0, big_time, scale);

    char info_line[96];
    if (have_time){
        strftime(info_line, sizeof(info_line), "%Y-%m-%d %a", &tm_now);
    } else {
        snprintf(info_line, sizeof(info_line), "No RTC sync");
    }
    fb_draw_text_line(0, scale * 7 + 2, info_line);

    char line3[96];
    int hours_uptime = seconds_since_boot / 3600;
    int minutes_uptime = (seconds_since_boot / 60) % 60;
    snprintf(line3, sizeof(line3), "Lines %d  Uptime %02d:%02d", g_line_count, hours_uptime % 100, minutes_uptime);
    fb_draw_text_line(0, scale * 7 + LINE_HEIGHT + 2, line3);
}

static void epd_display_framebuffer(void){
    epd_send_command(0x10);
    for (uint32_t i=0;i<sizeof(g_fb);++i){ uint8_t b = g_invert_pixels ? (uint8_t)~g_fb[i] : g_fb[i]; epd_send_data(b); }
    epd_send_command(0x13);
    for (uint32_t i=0;i<sizeof(g_fb);++i){ uint8_t b = g_invert_pixels ? (uint8_t)~g_fb[i] : g_fb[i]; epd_send_data(b); }
    epd_refresh();
}

static void epd_refresh_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
    if (x0 > x1 || y0 > y1 || x0 >= FB_WIDTH || y0 >= FB_HEIGHT) return;
    if (x1 >= FB_WIDTH) x1 = FB_WIDTH - 1;
    if (y1 >= FB_HEIGHT) y1 = FB_HEIGHT - 1;

    uint16_t aligned_x0 = (uint16_t)(x0 & ~7);
    uint16_t aligned_x1 = (uint16_t)(x1 | 7);
    if (aligned_x1 >= FB_WIDTH) aligned_x1 = FB_WIDTH - 1;
    uint16_t start_byte = aligned_x0 / 8;
    uint16_t width_bytes = (uint16_t)(((aligned_x1 - aligned_x0 + 1) + 7) / 8);

    epd_send_command(0x91);
    epd_send_command(0x90);
    epd_send_data((uint8_t)(aligned_x0 >> 8));
    epd_send_data((uint8_t)(aligned_x0 & 0xF8));
    epd_send_data((uint8_t)(aligned_x1 >> 8));
    epd_send_data((uint8_t)((aligned_x1 & 0xF8) | 0x07));
    epd_send_data((uint8_t)(y0 >> 8));
    epd_send_data((uint8_t)(y0 & 0xFF));
    epd_send_data((uint8_t)(y1 >> 8));
    epd_send_data((uint8_t)(y1 & 0xFF));

    epd_send_command(0x10);
    for (uint16_t y = y0; y <= y1; ++y){
        const uint8_t *row = &g_fb[y * FB_STRIDE + start_byte];
        for (uint16_t b = 0; b < width_bytes; ++b){
            uint8_t v = g_invert_pixels ? (uint8_t)~row[b] : row[b];
            epd_send_data(v);
        }
    }

    epd_send_command(0x13);
    for (uint16_t y = y0; y <= y1; ++y){
        const uint8_t *row = &g_fb[y * FB_STRIDE + start_byte];
        for (uint16_t b = 0; b < width_bytes; ++b){
            uint8_t v = g_invert_pixels ? (uint8_t)~row[b] : row[b];
            epd_send_data(v);
        }
    }

    epd_refresh();
    epd_send_command(0x92);
}

static void epd_refresh_status_bar(void)
{
    epd_refresh_window(0, 0, FB_WIDTH - 1, STATUS_BAR_HEIGHT - 1);
}

static void fb_update_status_bar(bool force_full_refresh)
{
    int minutes_since_boot = (int)(esp_timer_get_time() / 60000000);
    if (!force_full_refresh && minutes_since_boot == g_last_status_minute && g_line_count == g_last_status_line_count) return;
    fb_draw_status_bar_content(minutes_since_boot * 60);
    g_last_status_minute = minutes_since_boot;
    g_last_status_line_count = g_line_count;
    if (force_full_refresh) {
        epd_display_framebuffer();
    } else {
        epd_refresh_status_bar();
    }
}

static int fb_render_terminal_line(int cursor_y, const char* line){
    int last_line_y = fb_draw_text_line(0, cursor_y, line);
    cursor_y = last_line_y + LINE_HEIGHT;
    if (cursor_y + 7 >= FB_HEIGHT) {
        fb_scroll_up(LINE_HEIGHT);
        cursor_y = FB_HEIGHT - LINE_HEIGHT;
    }
    if (line && line[0] != '\0') {
        ++g_line_count;
    }
    fb_update_status_bar(true);
    return cursor_y;
}

// --- Low-level helpers ------------------------------------------------------
esp_err_t epd_send_command(uint8_t cmd)
{
    gpio_set_level(DC, 0);
    spi_transaction_t t = {
        .length = 8,
        .tx_buffer = &cmd,
    };
    return spi_device_transmit(spi, &t);
}

esp_err_t epd_send_data(uint8_t data)
{
    gpio_set_level(DC, 1);
    spi_transaction_t t = {
        .length = 8,
        .tx_buffer = &data,
    };
    return spi_device_transmit(spi, &t);
}

bool epd_is_busy(void)
{
    return gpio_get_level(BUSY) == 0; // LOW = busy
}

static inline void epd_wait_ready_high(void)
{
    // Wait until BUSY = HIGH (ready)
    while (epd_is_busy()) vTaskDelay(pdMS_TO_TICKS(10));
}

esp_err_t epd_wait_for_idle(void)
{
    const uint32_t t0 = ms();
    vTaskDelay(pdMS_TO_TICKS(10));
    while (epd_is_busy()) {
        if ((ms() - t0) > 10000) { // 10s safety
            LOGI("EPD", "BUSY stuck LOW for >10s");
            return ESP_ERR_TIMEOUT;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    LOGI("EPD", "BUSY HIGH after %u ms", (unsigned)(ms() - t0));
    return ESP_OK;
}

esp_err_t epd_reset(void)
{
    // Robust reset timing
    gpio_set_level(RST, 1); vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(RST, 0); vTaskDelay(pdMS_TO_TICKS(20));
    gpio_set_level(RST, 1); vTaskDelay(pdMS_TO_TICKS(50));
    return ESP_OK;
}

// --- Public API -------------------------------------------------------------
static esp_err_t epd_init_bus(int spi_mode)
{
    // SPI bus (no MISO; CS is held low externally per hardware)
    spi_bus_config_t bus_cfg = {
        .miso_io_num = -1,
        .mosi_io_num = SPI_MOSI,
        .sclk_io_num = SPI_SCLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4096,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO));

    spi_device_interface_config_t dev_cfg = {
        .clock_speed_hz = 10000000,
        .mode = spi_mode,
        .spics_io_num = -1,          // CS not used (held low on board)
        .queue_size = 3,
        .flags = SPI_DEVICE_NO_DUMMY,
    };
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &dev_cfg, &spi));

    // GPIOs
    gpio_config_t out = {
        .pin_bit_mask = (1ULL << DC) | (1ULL << RST),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&out));

    gpio_config_t in = {
        .pin_bit_mask = (1ULL << BUSY),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&in));

    // default levels
    gpio_set_level(RST, 1);
    return ESP_OK;
}

static void epd_write_plane(uint8_t cmd, uint8_t fill)
{
    // Writes a whole 800x480 mono plane (48,000 bytes)
    epd_send_command(cmd); // 0x10 (DTM1 old) or 0x13 (DTM2 new)
    const uint32_t bytes_per_line = EPD_WIDTH / 8; // 100
    for (uint32_t y = 0; y < EPD_HEIGHT; ++y) {
        for (uint32_t x = 0; x < bytes_per_line; ++x) {
            uint8_t b = g_invert_pixels ? (uint8_t)~fill : fill;
            epd_send_data(b);
        }
    }
}

static void epd_set_lut_by_host(const uint8_t* lut_vcom, const uint8_t* lut_ww,
                                const uint8_t* lut_bw,   const uint8_t* lut_wb,
                                const uint8_t* lut_bb)
{
    epd_send_command(0x20); // VCOM
    for (int i = 0; i < 42; ++i) epd_send_data(lut_vcom[i]);

    epd_send_command(0x21); // LUT WW
    for (int i = 0; i < 42; ++i) epd_send_data(lut_ww[i]);

    epd_send_command(0x22); // LUT BW
    for (int i = 0; i < 42; ++i) epd_send_data(lut_bw[i]);

    epd_send_command(0x23); // LUT WB
    for (int i = 0; i < 42; ++i) epd_send_data(lut_wb[i]);

    epd_send_command(0x24); // LUT BB
    for (int i = 0; i < 42; ++i) epd_send_data(lut_bb[i]);
}

static inline void epd_set_vcom_dc(uint8_t vcom)
{
    epd_send_command(0x82);
    epd_send_data(vcom);
}

static void epd_write_planes_white(void){
    epd_write_plane(0x10, 0xFF);
    epd_write_plane(0x13, 0xFF);
}

static esp_err_t epd_auto_refresh(void){
    // DSP full frame then AUTO (PON->DRF->POF)
    epd_send_command(0x11); epd_send_data(0x01);
    epd_send_command(0x17); epd_send_data(0xA5);
    return epd_wait_for_idle();
}

esp_err_t epd_refresh(void)
{
    epd_send_command(0x11); // DSP
    epd_send_data(0x01);
    epd_send_command(0x12); // DRF
    esp_err_t r = epd_wait_for_idle();
    LOGI("EPD", "DRF result: %s", (r==ESP_OK)?"OK":"TIMEOUT");
    return r;
}

esp_err_t epd_init_display(void)
{
    epd_reset();
    epd_wait_ready_high();

    // Power Setting
    epd_send_command(0x01);
    epd_send_data(0x07); // BD_EN=0, VSR_EN=1, VS_EN=1, VG_EN=1
    epd_send_data(0x17); // VCOM slew + VG level (typ)
    epd_send_data(0x3A); // VDH
    epd_send_data(0x3A); // VDL
    epd_send_data(0x03); // VDHR (unused for BW but keep default)

    // Optional explicit VCOM DC
    epd_set_vcom_dc(Voltage_Frame_7IN5_V2[4]); // 0x24 typical

    // Booster Soft Start
    epd_send_command(0x06);
    epd_send_data(0x27); epd_send_data(0x27); epd_send_data(0x2F); epd_send_data(0x17);

    // PLL (oscillator)
    epd_send_command(0x30);
    epd_send_data(0x06);

    // Power ON
    epd_send_command(0x04);
    esp_err_t pr = epd_wait_for_idle();
    LOGI("EPD", "PON wait result: %s", (pr==ESP_OK)?"OK":"TIMEOUT");

    LOGI("EPD", "Loading LUT tables");
    // Panel setting / resolution / timing (match Waveshare example)
    epd_send_command(0x00); epd_send_data(0x3F); // PSR
    epd_send_command(0x61); epd_send_data(0x03); epd_send_data(0x20); epd_send_data(0x01); epd_send_data(0xE0); // 800x480
    epd_send_command(0x15); epd_send_data(0x00); // Booster selection per ref code
    epd_send_command(0x50); epd_send_data(0x10); epd_send_data(0x00); // VCOM & data interval
    epd_send_command(0x60); epd_send_data(0x22); // TCON

    // Load full update LUTs (host-provided)
    epd_set_lut_by_host(LUT_VCOM_7IN5_V2, LUT_WW_7IN5_V2, LUT_BW_7IN5_V2,
                        LUT_WB_7IN5_V2, LUT_BB_7IN5_V2);

    // Initialize both SRAM planes to white so the first DRF has a full frame
    epd_write_plane(0x10, 0xFF); // OLD
    epd_write_plane(0x13, 0xFF); // NEW
    epd_refresh();
    return ESP_OK;
}

void app_main(void)
{
    // USB serial (optional for logs)
    usb_serial_jtag_driver_config_t usb_cfg = {
        .rx_buffer_size = 4096,
        .tx_buffer_size = 512,
    };
    usb_serial_jtag_driver_install(&usb_cfg);
    vTaskDelay(pdMS_TO_TICKS(50));

    ESP_ERROR_CHECK(epd_init_bus(0));
    LOGI("EPD", "RST=%d DC=%d BUSY=%d", gpio_get_level(RST), gpio_get_level(DC), gpio_get_level(BUSY));
    epd_init_display();

    // Start with a blank white screen and status bar
    g_line_count = 0;
    g_last_status_line_count = -1;
    g_last_status_minute = -1;
    fb_clear_white();
    fb_update_status_bar(true);

    // Simple terminal -> display: each Enter appends a new line (scrolls at bottom)
    LOGI("EPD", "Type a line and press Enter to render it (no clearing; newline append)");
    char line[TERMINAL_LINE_MAX];
    int pos = 0;
    uint8_t rx_buf[USB_RX_CHUNK];
    bool skip_lf = false;
    int cursor_y = STATUS_BAR_HEIGHT;
    const char* prompt = "> ";
    usb_serial_jtag_write_bytes((const uint8_t*)prompt, (size_t)strlen(prompt), portMAX_DELAY);
    while (1) {
        int n = usb_serial_jtag_read_bytes(rx_buf, sizeof(rx_buf), 10/portTICK_PERIOD_MS);
        if (n <= 0){
            fb_update_status_bar(false);
            vTaskDelay(1);
            continue;
        }

        for (int i = 0; i < n; ++i){
            uint8_t ch = rx_buf[i];
            if (skip_lf && ch == '\n') {
                skip_lf = false;
                continue;
            }
            skip_lf = false;

            if (ch=='\r' || ch=='\n'){
                line[pos] = '\0';
                cursor_y = fb_render_terminal_line(cursor_y, line);
                pos = 0;
                const char* nl = "\r\n> ";
                usb_serial_jtag_write_bytes((const uint8_t*)nl, (size_t)strlen(nl), portMAX_DELAY);
                if (ch == '\r') skip_lf = true;
            } else if ((ch==8 || ch==127) && pos>0){
                pos--;
                const char* bs = "\b \b";
                usb_serial_jtag_write_bytes((const uint8_t*)bs, 3, portMAX_DELAY);
            } else if (ch>=32 && ch<=126){
                if (pos < TERMINAL_LINE_MAX - 1) {
                    line[pos++] = (char)ch;
                }
                usb_serial_jtag_write_bytes(&ch, 1, portMAX_DELAY);
            }
        }
        fb_update_status_bar(false);
    }
}
