#ifndef MAIN_H
#define MAIN_H

#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>

// Pin definitions
#define SPI_SCLK 10
#define SPI_MOSI 7
#define RST 3
#define DC 4
#define BUSY 2

#define EPD_WIDTH       800
#define EPD_HEIGHT      480

// Core e-paper interface functions
esp_err_t epd_init(void);
esp_err_t epd_send_command(uint8_t cmd);
esp_err_t epd_send_data(uint8_t data);
bool epd_is_busy(void);
esp_err_t epd_wait_for_idle(void);
esp_err_t epd_refresh(void);
esp_err_t epd_reset(void);
esp_err_t epd_init_display(void);
esp_err_t epd_display_black(void);
esp_err_t epd_display_white(void);



#endif // MAIN_H
