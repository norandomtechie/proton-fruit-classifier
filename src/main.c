#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/sync.h"
#include "hardware/i2c.h"
#include "hardware/dma.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/irq.h"
#include "hardware/vreg.h"
#include "tusb.h"
#include "usb_descriptors.h"
#include "ov7670_regs.h"
#include "camera_capture.pio.h"
#include "fruit_classifier.h"

// --- Pin Definitions ---
#define PIN_D0      0
#define PIN_PCLK    8
#define PIN_HREF    9
#define PIN_VSYNC   10
#define PIN_SDA     12
#define PIN_SCL     13
#define PIN_RESET   14
#define PIN_PWDN    15
#define PIN_XCLK    23

#define CAM_I2C     i2c0

// --- Frame buffer ---
#define FRAME_BYTES (FRAME_WIDTH * FRAME_HEIGHT * 2)  // YUY2: 2 bytes/pixel
#define DMA_XFER_COUNT (FRAME_BYTES / 4)              // 32-bit transfers

static uint8_t frame_buf[2][FRAME_BYTES];
static volatile uint8_t capture_buf_idx = 0;  // which buffer DMA is writing to
static volatile bool frame_ready = false;      // a completed frame is available
static volatile uint8_t ready_buf_idx = 0;     // which buffer has the completed frame

// --- PIO / DMA ---
static PIO cam_pio = pio0;
static uint cam_sm = 0;
static int dma_chan = -1;

// --- UVC state ---
static volatile unsigned tx_busy = 0;
static unsigned frame_num = 0;
static unsigned already_sent = 0;
static unsigned interval_ms = 1000 / FRAME_RATE;
static uint32_t tx_start_ms = 0;  // when current transfer started

// --- Debug counters ---
static volatile uint32_t dbg_vsync_fall = 0;
static volatile uint32_t dbg_vsync_rise = 0;
static volatile uint32_t dbg_frames_sent = 0;
static volatile uint32_t dbg_frames_ready = 0;

// --- ML inference state (core 1) ---
static volatile bool ml_frame_pending = false;     // core0 signals core1
static volatile uint8_t ml_buf_idx = 0;            // which buffer to classify
static volatile bool ml_result_ready = false;       // core1 signals result
static volatile fruit_result_t ml_last_result = {0};
static volatile uint32_t ml_inference_count = 0;
static volatile int ml_init_status = 0;            // 0=pending, 1=ok, -1=fail
static volatile uint32_t ml_arena_used = 0;

// =====================================================================
// I2C / SCCB helpers
// =====================================================================
static int cam_write_reg(uint8_t reg, uint8_t val) {
    uint8_t buf[2] = {reg, val};
    int ret = i2c_write_blocking(CAM_I2C, OV7670_ADDR, buf, 2, false);
    if (ret < 0) {
        printf("I2C write error: reg=0x%02X val=0x%02X ret=%d\n", reg, val, ret);
    }
    sleep_us(300);
    return ret;
}

static uint8_t cam_read_reg(uint8_t reg) {
    uint8_t val = 0;
    int ret = i2c_write_blocking(CAM_I2C, OV7670_ADDR, &reg, 1, false);  // SCCB: STOP between write and read
    if (ret < 0) {
        printf("I2C read setup error: reg=0x%02X ret=%d\n", reg, ret);
        return 0;
    }
    ret = i2c_read_blocking(CAM_I2C, OV7670_ADDR, &val, 1, false);
    if (ret < 0) {
        printf("I2C read error: reg=0x%02X ret=%d\n", reg, ret);
        return 0;
    }
    return val;
}

static void i2c_bus_scan(void) {
    printf("I2C bus scan:\n");
    bool found = false;
    for (int addr = 0x08; addr < 0x78; addr++) {
        uint8_t dummy;
        int ret = i2c_read_blocking(CAM_I2C, addr, &dummy, 1, false);
        if (ret >= 0) {
            printf("  Found device at 0x%02X\n", addr);
            found = true;
        }
    }
    if (!found) {
        printf("  No devices found!\n");
    }
}

static void cam_write_list(const ov7670_reg_t *list) {
    for (int i = 0; list[i].reg <= OV7670_REG_LAST; i++) {
        cam_write_reg(list[i].reg, list[i].val);
        sleep_ms(1);
    }
}

// =====================================================================
// OV7670 frame control for QQVGA (from usedbytes driver)
// =====================================================================
static void cam_frame_control_qqvga(void) {
    // QQVGA = SIZE_DIV4 (index 2): vstart=11, hstart=186, edge_offset=2, pclk_delay=2
    uint8_t size = 2; // DIV4 → 160x120
    uint8_t vstart = 11;
    uint16_t hstart = 186;
    uint8_t edge_offset = 2;
    uint8_t pclk_delay = 2;

    // Enable downsampling
    uint8_t com3_val = OV7670_COM3_DCWEN;
    cam_write_reg(OV7670_REG_COM3, com3_val);

    // PCLK division: 0x18 + size for sub-VGA
    uint8_t com14_val = 0x18 + size;
    cam_write_reg(OV7670_REG_COM14, com14_val);

    // Downsample ratio
    uint8_t dcwctr_val = size * 0x11;
    cam_write_reg(OV7670_REG_SCALING_DCWCTR, dcwctr_val);

    // Pixel clock divider
    uint8_t pclk_div_val = 0xF0 + size;
    cam_write_reg(OV7670_REG_SCALING_PCLK_DIV, pclk_div_val);

    // Digital zoom
    uint8_t zoom = 0x48; // for DIV4
    uint8_t xsc = cam_read_reg(OV7670_REG_SCALING_XSC);
    uint8_t ysc = cam_read_reg(OV7670_REG_SCALING_YSC);
    xsc = (xsc & 0x80) | zoom;
    ysc = (ysc & 0x80) | zoom;
    cam_write_reg(OV7670_REG_SCALING_XSC, xsc);
    cam_write_reg(OV7670_REG_SCALING_YSC, ysc);

    // Window registers
    uint16_t vstop = vstart + 480;
    uint16_t hstop = (hstart + 640) % 784;
    cam_write_reg(OV7670_REG_HSTART, hstart >> 3);
    cam_write_reg(OV7670_REG_HSTOP,  hstop >> 3);
    cam_write_reg(OV7670_REG_HREF,   (edge_offset << 6) | ((hstop & 0x07) << 3) | (hstart & 0x07));
    cam_write_reg(OV7670_REG_VSTART, vstart >> 2);
    cam_write_reg(OV7670_REG_VSTOP,  vstop >> 2);
    cam_write_reg(OV7670_REG_VREF,   ((vstop & 0x03) << 2) | (vstart & 0x03));
    cam_write_reg(OV7670_REG_SCALING_PCLK_DELAY, pclk_delay);
}

// =====================================================================
// Camera initialization
// =====================================================================
static bool cam_init(void) {
    // I2C at 100 kHz - init FIRST before XCLK so pins are configured
    i2c_init(CAM_I2C, 100 * 1000);
    gpio_set_function(PIN_SDA, GPIO_FUNC_I2C);
    gpio_set_function(PIN_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(PIN_SDA);
    gpio_pull_up(PIN_SCL);
    printf("I2C0 initialized on SDA=GPIO%d SCL=GPIO%d\n", PIN_SDA, PIN_SCL);

    // PWDN low (enable camera), RESET low (hold in reset)
    gpio_init(PIN_PWDN);
    gpio_set_dir(PIN_PWDN, GPIO_OUT);
    gpio_put(PIN_PWDN, 0);

    gpio_init(PIN_RESET);
    gpio_set_dir(PIN_RESET, GPIO_OUT);
    gpio_put(PIN_RESET, 0);
    sleep_ms(10);

    // Setup XCLK: ~12.5 MHz from sys_clk (150 MHz / 12)
    uint32_t sys_clk = clock_get_hz(clk_sys);
    printf("System clock: %lu Hz\n", sys_clk);
    clock_gpio_init(PIN_XCLK, CLOCKS_CLK_GPOUT0_CTRL_AUXSRC_VALUE_CLK_SYS, 12);
    printf("XCLK started on GPIO%d (~%lu Hz)\n", PIN_XCLK, sys_clk / 12);

    // XCLK must be running before releasing reset
    sleep_ms(10);

    // Release reset
    gpio_put(PIN_RESET, 1);
    sleep_ms(500);  // OV7670 needs time after reset with XCLK running

    printf("Camera power sequence complete (PWDN=0, RESET=1)\n");

    // Scan I2C bus
    i2c_bus_scan();

    // Read PID - expect 0x76
    uint8_t pid = cam_read_reg(OV7670_REG_PID);
    printf("OV7670 PID: 0x%02X (expect 0x76)\n", pid);
    if (pid != 0x76) {
        printf("ERROR: OV7670 not detected!\n");
        printf("Check: XCLK->GPIO%d, SDA->GPIO%d, SCL->GPIO%d, "
               "RESET->GPIO%d, PWDN->GPIO%d, D0-D7->GPIO%d-%d\n",
               PIN_XCLK, PIN_SDA, PIN_SCL, PIN_RESET, PIN_PWDN,
               PIN_D0, PIN_D0+7);
        return false;
    }

    uint8_t midh = cam_read_reg(OV7670_REG_MIDH);
    printf("OV7670 MIDH: 0x%02X (expect 0x7F)\n", midh);

    // Soft reset
    cam_write_reg(OV7670_REG_COM7, OV7670_COM7_RESET);
    sleep_ms(1000);

    // Clock config: internal prescaler
    cam_write_reg(OV7670_REG_CLKRC, 1);        // CLK * 4
    cam_write_reg(OV7670_REG_DBLV, 1 << 6);    // PLL x4

    // Set YUV output format
    cam_write_list(ov7670_yuv_regs);

    // Write main init table
    cam_write_list(ov7670_init_regs);

    // Set QQVGA frame size (160x120)
    cam_frame_control_qqvga();

    sleep_ms(300); // settling time

    printf("OV7670 configured for QVGA YUV422\n");
    return true;
}

// =====================================================================
// DMA restart - called on VSYNC to begin capturing next frame
// =====================================================================
static void dma_restart(void) {
    dma_channel_abort(dma_chan);

    // Drain any stale data from PIO RX FIFO
    while (!pio_sm_is_rx_fifo_empty(cam_pio, cam_sm)) {
        pio_sm_get(cam_pio, cam_sm);
    }

    dma_channel_set_write_addr(dma_chan, frame_buf[capture_buf_idx], false);
    dma_channel_set_trans_count(dma_chan, DMA_XFER_COUNT, true);
}

// =====================================================================
// VSYNC GPIO interrupt handler
// =====================================================================
static void vsync_isr(uint gpio, uint32_t events) {
    if (gpio != PIN_VSYNC) return;

    if (events & GPIO_IRQ_EDGE_FALL) {
        dbg_vsync_fall++;
        dma_restart();
    }
    if (events & GPIO_IRQ_EDGE_RISE) {
        dbg_vsync_rise++;
        if (!tx_busy) {
            // Only swap when USB isn't reading the other buffer
            ready_buf_idx = capture_buf_idx;
            frame_ready = true;
            capture_buf_idx ^= 1;
            dbg_frames_ready++;
        }
        // If tx_busy, keep capturing into the same buffer (drop this frame)
    }
}

// =====================================================================
// PIO + DMA setup
// =====================================================================
static void capture_init(void) {
    // Load PIO program
    uint offset = pio_add_program(cam_pio, &camera_capture_program);
    camera_capture_program_init(cam_pio, cam_sm, offset, PIN_D0, PIN_PCLK, PIN_HREF);

    // DMA channel
    dma_chan = dma_claim_unused_channel(true);
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_32);
    channel_config_set_read_increment(&cfg, false);
    channel_config_set_write_increment(&cfg, true);
    channel_config_set_dreq(&cfg, pio_get_dreq(cam_pio, cam_sm, false));

    dma_channel_configure(dma_chan, &cfg,
        frame_buf[0],                        // write address
        &cam_pio->rxf[cam_sm],               // read address (PIO RX FIFO)
        DMA_XFER_COUNT,                      // transfer count
        false                                // don't start yet
    );

    // VSYNC interrupt
    gpio_init(PIN_VSYNC);
    gpio_set_dir(PIN_VSYNC, GPIO_IN);
    gpio_set_irq_enabled_with_callback(PIN_VSYNC,
        GPIO_IRQ_EDGE_FALL | GPIO_IRQ_EDGE_RISE, true, vsync_isr);

    // Start PIO state machine
    pio_sm_set_enabled(cam_pio, cam_sm, true);
    printf("PIO + DMA capture initialized\n");
}

// =====================================================================
// UVC video streaming
// =====================================================================

// Pick a frame buffer to send — prefer the latest ready buffer,
// but always have *something* even if no new frame arrived yet.
static uint8_t pick_send_buf(void) {
    uint32_t save = save_and_disable_interrupts();
    uint8_t idx = ready_buf_idx;
    frame_ready = false;
    restore_interrupts(save);
    return idx;
}

static void video_send_frame(void) {
    if (!tud_video_n_streaming(0, 0)) {
        already_sent = 0;
        frame_num = 0;
        tx_busy = 0;
        return;
    }

    // Timeout: if tx_busy stuck for >500ms, force-clear it
    if (tx_busy) {
        uint32_t now = to_ms_since_boot(get_absolute_time());
        if (now - tx_start_ms > 500) {
            tx_busy = 0;
        }
    }

    // First frame: send immediately when streaming starts
    if (!already_sent) {
        already_sent = 1;
        tx_busy = 1;
        tx_start_ms = to_ms_since_boot(get_absolute_time());
        uint8_t idx = pick_send_buf();
        dbg_frames_sent++;
        if (!tud_video_n_frame_xfer(0, 0, (void*)frame_buf[idx], FRAME_BYTES)) {
            tx_busy = 0;
        }
        return;
    }

    if (tx_busy) return;
    if (!frame_ready) return;

    tx_busy = 1;
    tx_start_ms = to_ms_since_boot(get_absolute_time());
    uint8_t idx = pick_send_buf();
    dbg_frames_sent++;
    if (!tud_video_n_frame_xfer(0, 0, (void*)frame_buf[idx], FRAME_BYTES)) {
        tx_busy = 0;
    }
}

void tud_video_frame_xfer_complete_cb(uint_fast8_t ctl_idx, uint_fast8_t stm_idx) {
    (void)ctl_idx;
    (void)stm_idx;
    tx_busy = 0;
    frame_num++;
}

int tud_video_commit_cb(uint_fast8_t ctl_idx, uint_fast8_t stm_idx,
                        video_probe_and_commit_control_t const *parameters) {
    (void)ctl_idx;
    (void)stm_idx;
    // Read the negotiated frame interval (in 100ns units) and convert to ms
    interval_ms = parameters->dwFrameInterval / 10000;
    if (interval_ms < 10) interval_ms = 10;  // Clamp to reasonable minimum
    // Reset streaming state for new session
    frame_num = 0;
    tx_busy = 0;
    already_sent = 0;
    return VIDEO_ERROR_NONE;
}

// =====================================================================
// USB callbacks
// =====================================================================
void tud_mount_cb(void) {
    printf("USB mounted\n");
}

void tud_umount_cb(void) {
    printf("USB unmounted\n");
}

void tud_suspend_cb(bool remote_wakeup_en) {
    (void)remote_wakeup_en;
}

void tud_resume_cb(void) {
}

// =====================================================================
// Core 1: ML inference loop
// =====================================================================
static void core1_ml_entry(void) {
    // No printf on Core 1 — it interleaves with Core 0 CDC output.
    // Use shared flags to report status back to Core 0.

    if (!fruit_classifier_init()) {
        ml_init_status = -1;  // signal failure to Core 0
        while (1) { tight_loop_contents(); }
    }

    ml_arena_used = 0; // could read from interpreter if exposed
    ml_init_status = 1;  // signal success to Core 0

    while (1) {
        if (!ml_frame_pending) {
            tight_loop_contents();
            continue;
        }

        // Run classification on the indicated frame buffer
        fruit_result_t result;
        uint8_t idx = ml_buf_idx;
        ml_frame_pending = false;

        if (fruit_classifier_run(frame_buf[idx], FRAME_WIDTH, FRAME_HEIGHT, &result)) {
            ml_last_result = result;
            ml_result_ready = true;
            ml_inference_count++;
        }
    }
}

void hd44780_init();
void hd44780_display_line1(const char *text);
void hd44780_display_line2(const char *text);

// =====================================================================
// Main (Core 0: camera + USB + dispatch to Core 1)
// =====================================================================
int main(void) {
    // Overclock to 250 MHz for faster ML inference
    vreg_set_voltage(VREG_VOLTAGE_1_20);
    sleep_ms(10);
    set_sys_clock_khz(250000, true);

    // Initialize LCD
    sleep_ms(100);
    hd44780_init();
    hd44780_display_line1("Hi! I'm Proton!");
    hd44780_display_line2("I identify fruit");

    stdio_init_all();
    sleep_ms(2000); // let CDC connect

    printf("\n=== OV7670 UVC Camera + Fruit Classifier ===\n");
    printf("System clock: %lu MHz\n", clock_get_hz(clk_sys) / 1000000);

    // Initialize camera
    if (!cam_init()) {
        printf("Camera init failed, halting.\n");
        while (1) { tight_loop_contents(); }
    }

    hd44780_display_line1("My camera is");
    hd44780_display_line2("initialized!");
    sleep_ms(1000);

    // Initialize PIO + DMA capture
    capture_init();

    // Launch Core 1 for ML inference
    multicore_launch_core1(core1_ml_entry);
    printf("Core 1 launched for ML inference\n");

    printf("Entering main loop...\n"); 

    uint32_t last_status = 0;
    uint32_t last_ml_frame = 0;
    uint32_t last_lcd_update = 0;
    bool ml_init_reported = false;
    while (1) {
        tud_task();
        video_send_frame();

        // Keep calling tud_task() frequently — do NOT let printf or
        // other work starve the USB stack.
        tud_task();

        uint32_t now = to_ms_since_boot(get_absolute_time());

        // Report Core 1 ML init status once
        if (!ml_init_reported && ml_init_status != 0) {
            if (ml_init_status == 1) {
                printf("[Core1] Fruit classifier initialized OK\n");
                hd44780_display_line1("Fruit classifier");
                hd44780_display_line2("has started!");
            } else {
                printf("[Core1] ERROR: classifier init FAILED\n");
            }
            ml_init_reported = true;
        }

        // Dispatch a frame to Core 1 for classification every ~500ms.
        if (now - last_ml_frame >= 500 && !ml_frame_pending
            && ml_init_status == 1 && dbg_frames_ready > 0) {
            ml_buf_idx = ready_buf_idx;
            ml_frame_pending = true;
            last_ml_frame = now;
        }

        // Only print status when CDC is connected, and less frequently
        // during active streaming to avoid stalling the main loop.
        uint32_t status_interval = tud_video_n_streaming(0, 0) ? 5000 : 2000;
        if (now - last_status >= status_interval) {
            last_status = now;
            if (tud_cdc_connected()) {
                uint32_t dma_remaining = dma_channel_hw_addr(dma_chan)->transfer_count;
                printf("VSYNC f/r=%lu/%lu ready=%lu sent=%lu "
                       "tx_busy=%u streaming=%d dma_rem=%lu\n",
                       dbg_vsync_fall, dbg_vsync_rise, dbg_frames_ready, dbg_frames_sent,
                       tx_busy, tud_video_n_streaming(0, 0), dma_remaining);
            }
        }

        if (ml_result_ready && (now - last_lcd_update >= 300)) {
            char buf[16];
            sprintf(buf, "%s (%d)",
                ml_last_result.class_name,
                ml_last_result.confidence);
            hd44780_display_line1(buf);
            printf("%s\n", buf);
            sprintf(buf, "[%d,%d,%d,%d]",
                ml_last_result.scores[0], ml_last_result.scores[1],
                ml_last_result.scores[2], ml_last_result.scores[3]);
            hd44780_display_line2(buf);
            printf("%s\n", buf);

            ml_result_ready = false;
            last_lcd_update = now;
        }
    }
}