#ifndef OV7670_REGS_H_
#define OV7670_REGS_H_

#include <stdint.h>

// I2C address
#define OV7670_ADDR 0x21

// Register addresses
#define OV7670_REG_GAIN            0x00
#define OV7670_REG_BLUE            0x01
#define OV7670_REG_RED             0x02
#define OV7670_REG_VREF            0x03
#define OV7670_REG_COM1            0x04
#define OV7670_REG_BAVE            0x05
#define OV7670_REG_GbAVE           0x06
#define OV7670_REG_AECHH           0x07
#define OV7670_REG_RAVE            0x08
#define OV7670_REG_COM2            0x09
#define OV7670_REG_PID             0x0A
#define OV7670_REG_VER             0x0B
#define OV7670_REG_COM3            0x0C
#define OV7670_REG_COM4            0x0D
#define OV7670_REG_COM5            0x0E
#define OV7670_REG_COM6            0x0F
#define OV7670_REG_AECH            0x10
#define OV7670_REG_CLKRC           0x11
#define OV7670_REG_COM7            0x12
#define OV7670_REG_COM8            0x13
#define OV7670_REG_COM9            0x14
#define OV7670_REG_COM10           0x15
#define OV7670_REG_HSTART          0x17
#define OV7670_REG_HSTOP           0x18
#define OV7670_REG_VSTART          0x19
#define OV7670_REG_VSTOP           0x1A
#define OV7670_REG_PSHFT           0x1B
#define OV7670_REG_MIDH            0x1C
#define OV7670_REG_MIDL            0x1D
#define OV7670_REG_MVFP            0x1E
#define OV7670_REG_ADCCTR0        0x20
#define OV7670_REG_ADCCTR1        0x21
#define OV7670_REG_ADCCTR2        0x22
#define OV7670_REG_AEW             0x24
#define OV7670_REG_AEB             0x25
#define OV7670_REG_VPT             0x26
#define OV7670_REG_BBIAS           0x27
#define OV7670_REG_GbBIAS          0x28
#define OV7670_REG_EXHCH           0x2A
#define OV7670_REG_EXHCL           0x2B
#define OV7670_REG_RBIAS           0x2C
#define OV7670_REG_ADVFL           0x2D
#define OV7670_REG_ADVFH           0x2E
#define OV7670_REG_YAVE            0x2F
#define OV7670_REG_HSYST           0x30
#define OV7670_REG_HSYEN           0x31
#define OV7670_REG_HREF            0x32
#define OV7670_REG_CHLF            0x33
#define OV7670_REG_ARBLM           0x34
#define OV7670_REG_ADC             0x37
#define OV7670_REG_ACOM            0x38
#define OV7670_REG_OFON            0x39
#define OV7670_REG_TSLB            0x3A
#define OV7670_REG_COM11           0x3B
#define OV7670_REG_COM12           0x3C
#define OV7670_REG_COM13           0x3D
#define OV7670_REG_COM14           0x3E
#define OV7670_REG_EDGE            0x3F
#define OV7670_REG_COM15           0x40
#define OV7670_REG_COM16           0x41
#define OV7670_REG_COM17           0x42
#define OV7670_REG_AWBC1           0x43
#define OV7670_REG_AWBC2           0x44
#define OV7670_REG_AWBC3           0x45
#define OV7670_REG_AWBC4           0x46
#define OV7670_REG_AWBC5           0x47
#define OV7670_REG_AWBC6           0x48
#define OV7670_REG_REG4B           0x4B
#define OV7670_REG_DNSTH           0x4C
#define OV7670_REG_MTX1            0x4F
#define OV7670_REG_MTX2            0x50
#define OV7670_REG_MTX3            0x51
#define OV7670_REG_MTX4            0x52
#define OV7670_REG_MTX5            0x53
#define OV7670_REG_MTX6            0x54
#define OV7670_REG_BRIGHT          0x55
#define OV7670_REG_CONTRAS         0x56
#define OV7670_REG_CONTRAS_CENTER  0x57
#define OV7670_REG_MTXS            0x58
#define OV7670_REG_LCC1            0x62
#define OV7670_REG_LCC2            0x63
#define OV7670_REG_LCC3            0x64
#define OV7670_REG_LCC4            0x65
#define OV7670_REG_LCC5            0x66
#define OV7670_REG_MANU            0x67
#define OV7670_REG_MANV            0x68
#define OV7670_REG_GFIX            0x69
#define OV7670_REG_GGAIN           0x6A
#define OV7670_REG_DBLV            0x6B
#define OV7670_REG_AWBCTR3         0x6C
#define OV7670_REG_AWBCTR2         0x6D
#define OV7670_REG_AWBCTR1         0x6E
#define OV7670_REG_AWBCTR0         0x6F
#define OV7670_REG_SCALING_XSC     0x70
#define OV7670_REG_SCALING_YSC     0x71
#define OV7670_REG_SCALING_DCWCTR  0x72
#define OV7670_REG_SCALING_PCLK_DIV 0x73
#define OV7670_REG_REG74           0x74
#define OV7670_REG_REG76           0x76
#define OV7670_REG_SLOP            0x7A
#define OV7670_REG_GAM_BASE        0x7B
#define OV7670_REG_RGB444          0x8C
#define OV7670_REG_DM_LNL          0x92
#define OV7670_REG_LCC6            0x94
#define OV7670_REG_LCC7            0x95
#define OV7670_REG_HAECC1          0x9F
#define OV7670_REG_HAECC2          0xA0
#define OV7670_REG_SCALING_PCLK_DELAY 0xA2
#define OV7670_REG_BD50MAX         0xA5
#define OV7670_REG_HAECC3          0xA6
#define OV7670_REG_HAECC4          0xA7
#define OV7670_REG_HAECC5          0xA8
#define OV7670_REG_HAECC6          0xA9
#define OV7670_REG_HAECC7          0xAA
#define OV7670_REG_BD60MAX         0xAB
#define OV7670_REG_ABLC1           0xB1
#define OV7670_REG_THL_ST          0xB3
#define OV7670_REG_SATCTR          0xC9

#define OV7670_REG_LAST            OV7670_REG_SATCTR

// Bit defines
#define OV7670_COM7_RESET          0x80
#define OV7670_COM7_SIZE_VGA       0x00
#define OV7670_COM7_RGB            0x04
#define OV7670_COM7_YUV            0x00
#define OV7670_COM3_DCWEN          0x04
#define OV7670_COM3_SCALEEN        0x08
#define OV7670_COM8_FASTAEC        0x80
#define OV7670_COM8_AECSTEP        0x40
#define OV7670_COM8_BANDING        0x20
#define OV7670_COM8_AGC            0x04
#define OV7670_COM8_AWB            0x02
#define OV7670_COM8_AEC            0x01
#define OV7670_COM10_VS_NEG        0x02
#define OV7670_COM15_R00FF         0xC0
#define OV7670_COM15_RGB565        0x10
#define OV7670_TSLB_YLAST          0x04
#define OV7670_MVFP_MIRROR         0x20
#define OV7670_MVFP_VFLIP          0x10
#define OV7670_CLK_EXT             0x40

// Register address/value pair for init sequences
typedef struct {
    uint8_t reg;
    uint8_t val;
} ov7670_reg_t;

// YUV format setup (write before init table)
static const ov7670_reg_t ov7670_yuv_regs[] = {
    {OV7670_REG_COM7,  OV7670_COM7_YUV},
    {OV7670_REG_COM15, OV7670_COM15_R00FF},
    {0xFF, 0xFF}, // end marker
};

// Main init table (from Adafruit/usedbytes OV7670 driver)
static const ov7670_reg_t ov7670_init_regs[] = {
    {OV7670_REG_TSLB,  0x04},  // TSLB: YUYV output order (bit3=0 for YUYV/YUY2)
    {OV7670_REG_COM13, 0xC1},  // COM13: gamma enable + UV auto + UV swap (fix blue tint)
    {OV7670_REG_SLOP,  0x20},
    // Gamma curve
    {OV7670_REG_GAM_BASE + 0,  0x1C},
    {OV7670_REG_GAM_BASE + 1,  0x28},
    {OV7670_REG_GAM_BASE + 2,  0x3C},
    {OV7670_REG_GAM_BASE + 3,  0x55},
    {OV7670_REG_GAM_BASE + 4,  0x68},
    {OV7670_REG_GAM_BASE + 5,  0x76},
    {OV7670_REG_GAM_BASE + 6,  0x80},
    {OV7670_REG_GAM_BASE + 7,  0x88},
    {OV7670_REG_GAM_BASE + 8,  0x8F},
    {OV7670_REG_GAM_BASE + 9,  0x96},
    {OV7670_REG_GAM_BASE + 10, 0xA3},
    {OV7670_REG_GAM_BASE + 11, 0xAF},
    {OV7670_REG_GAM_BASE + 12, 0xC4},
    {OV7670_REG_GAM_BASE + 13, 0xD7},
    {OV7670_REG_GAM_BASE + 14, 0xE8},
    // AEC/AGC
    {OV7670_REG_COM8,  OV7670_COM8_FASTAEC | OV7670_COM8_AECSTEP | OV7670_COM8_BANDING},
    {OV7670_REG_GAIN,  0x00},
    {OV7670_REG_COM2,  0x00},
    {OV7670_REG_COM4,  0x00},
    {OV7670_REG_COM9,  0x20},
    {OV7670_REG_COM11, (1 << 3)},
    {0x9D, 89},  // Banding filter 50 Hz at ~14 MHz
    {OV7670_REG_BD50MAX, 0x05},
    {OV7670_REG_BD60MAX, 0x07},
    {OV7670_REG_AEW,  0x75},
    {OV7670_REG_AEB,  0x63},
    {OV7670_REG_VPT,  0xA5},
    {OV7670_REG_HAECC1, 0x78},
    {OV7670_REG_HAECC2, 0x68},
    {0xA1, 0x03},
    {OV7670_REG_HAECC3, 0xDF},
    {OV7670_REG_HAECC4, 0xDF},
    {OV7670_REG_HAECC5, 0xF0},
    {OV7670_REG_HAECC6, 0x90},
    {OV7670_REG_HAECC7, 0x94},
    {OV7670_REG_COM8,  OV7670_COM8_FASTAEC | OV7670_COM8_AECSTEP |
                        OV7670_COM8_BANDING | OV7670_COM8_AGC |
                        OV7670_COM8_AEC | OV7670_COM8_AWB},
    // Misc
    {OV7670_REG_COM5,  0x61},
    {OV7670_REG_COM6,  0x4B},
    {0x16, 0x02},
    {OV7670_REG_MVFP,  0x07},
    {OV7670_REG_ADCCTR1, 0x02},
    {OV7670_REG_ADCCTR2, 0x91},
    {0x29, 0x07},
    {OV7670_REG_CHLF,  0x0B},
    {0x35, 0x0B},
    {OV7670_REG_ADC,   0x1D},
    {OV7670_REG_ACOM,  0x71},
    {OV7670_REG_OFON,  0x2A},
    {OV7670_REG_COM12, 0x78},
    {0x4D, 0x40},
    {0x4E, 0x20},
    {OV7670_REG_GFIX,  0x5D},
    {OV7670_REG_REG74, 0x19},
    {0x8D, 0x4F},
    {0x8E, 0x00},
    {0x8F, 0x00},
    {0x90, 0x00},
    {0x91, 0x00},
    {OV7670_REG_DM_LNL, 0x00},
    {0x96, 0x00},
    {0x9A, 0x80},
    {0xB0, 0x84},
    {OV7670_REG_ABLC1, 0x0C},
    {0xB2, 0x0E},
    {OV7670_REG_THL_ST, 0x82},
    {0xB8, 0x0A},
    // AWB
    {OV7670_REG_AWBC1, 0x14},
    {OV7670_REG_AWBC2, 0xF0},
    {OV7670_REG_AWBC3, 0x34},
    {OV7670_REG_AWBC4, 0x58},
    {OV7670_REG_AWBC5, 0x28},
    {OV7670_REG_AWBC6, 0x3A},
    {0x59, 0x88},
    {0x5A, 0x88},
    {0x5B, 0x44},
    {0x5C, 0x67},
    {0x5D, 0x49},
    {0x5E, 0x0E},
    {OV7670_REG_LCC3,  0x04},
    {OV7670_REG_LCC4,  0x20},
    {OV7670_REG_LCC5,  0x05},
    {OV7670_REG_LCC6,  0x04},
    {OV7670_REG_LCC7,  0x08},
    {OV7670_REG_AWBCTR3, 0x0A},
    {OV7670_REG_AWBCTR2, 0x55},
    {OV7670_REG_AWBCTR1, 0x11},
    {OV7670_REG_AWBCTR0, 0x9E},
    // Brightness / contrast
    {OV7670_REG_BRIGHT, 0x00},
    {OV7670_REG_CONTRAS, 0x40},
    {OV7670_REG_CONTRAS_CENTER, 0x80},
    // End
    {OV7670_REG_LAST + 1, 0x00},
};

#endif
