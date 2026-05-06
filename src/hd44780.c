// This file implements a basic library for 4-bit parallel interfacing 
// with the Adafruit HD44780 2x16 character LCD.

#include "pico/stdlib.h"

#define RW 24
#define EN 25
#define RS 31
#define DB4 27
#define DB5 28
#define DB6 29
#define DB7 30  // valid, we are using rp2350b

static inline void hd44780_pulse_enable(void) {
    gpio_put(EN, 1);
    sleep_us(2);
    gpio_put(EN, 0);
    sleep_us(2);
}

void hd44780_gpio_init() {
    int arr[7] = {EN, RS, RW, DB4, DB5, DB6, DB7};
    for (int i = 0; i < 7; i++) {
        gpio_init(arr[i]);
        gpio_set_dir(arr[i], GPIO_OUT);
        gpio_put(arr[i], 0);
    }
    gpio_put(RW, 0);
}

void hd44780_set_pattern(int val) {
    gpio_put(RS, !!(val & 16));
    gpio_put(RW, 0);
    sleep_us(1);
    
    gpio_put(DB4, !!(val & 1));
    gpio_put(DB5, !!(val & 2));
    gpio_put(DB6, !!(val & 4));
    gpio_put(DB7, !!(val & 8));
    hd44780_pulse_enable();

    sleep_us(50);
}

void hd44780_init() {
    hd44780_gpio_init();

    sleep_ms(50);
    // Force 8-bit mode reset sequence (send high nibble 0x3 three times).
    hd44780_set_pattern(0b00011);
    sleep_ms(5);
    hd44780_set_pattern(0b00011);
    sleep_us(100);
    hd44780_set_pattern(0b00011);

    // Switch to 4-bit mode (send high nibble 0x2).
    hd44780_set_pattern(0b00010);

    // From here on, send full bytes.
    hd44780_send_cmd(0x28); // Function set: 4-bit, 2-line, 5x8 font
    hd44780_send_cmd(0x08); // Display off
    hd44780_send_cmd(0x01); // Clear display
    hd44780_send_cmd(0x06); // Entry mode: increment, no shift
    hd44780_send_cmd(0x0C); // Display on, cursor off, blink off
    sleep_ms(2);
}

static inline void hd44780_write_nibble(uint8_t nibble, bool rs) {
    hd44780_set_pattern(((uint8_t)rs << 4) | (nibble & 0x0F));
    sleep_us(1);
}

static void hd44780_write_byte(uint8_t value, bool rs) {
    hd44780_write_nibble((value >> 4) & 0x0F, rs);
    hd44780_write_nibble(value & 0x0F, rs);
    sleep_us(40);
}

void hd44780_send_cmd(uint8_t cmd) {
    hd44780_write_byte(cmd, false);
    if (cmd == 0x01 || cmd == 0x02) {
        sleep_ms(2);
    }
}

void hd44780_send_data(uint8_t data) {
    hd44780_write_byte(data, true);
}

static void hd44780_write_line(uint8_t ddram_addr, const char *text) {
    hd44780_send_cmd(ddram_addr);
    for (int i = 0; i < 16; i++) {
        char c = ' ';
        if (text && *text) {
            c = *text;
            text++;
        }
        hd44780_send_data((uint8_t)c);
    }
}

void hd44780_display_line1(const char *text) {
    hd44780_write_line(0x80, text);
}

void hd44780_display_line2(const char *text) {
    hd44780_write_line(0xC0, text);
}