#include "tusb.h"
#include "usb_descriptors.h"

//--------------------------------------------------------------------
// Device Descriptor
//--------------------------------------------------------------------
tusb_desc_device_t const desc_device = {
    .bLength            = sizeof(tusb_desc_device_t),
    .bDescriptorType    = TUSB_DESC_DEVICE,
    .bcdUSB             = 0x0200,
    .bDeviceClass       = TUSB_CLASS_MISC,
    .bDeviceSubClass    = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol    = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0    = CFG_TUD_ENDPOINT0_SIZE,
    .idVendor           = 0xCAFE,
    .idProduct          = 0x4007,
    .bcdDevice          = 0x0100,
    .iManufacturer      = 0x01,
    .iProduct           = 0x02,
    .iSerialNumber      = 0x03,
    .bNumConfigurations = 0x01,
};

uint8_t const* tud_descriptor_device_cb(void) {
    return (uint8_t const*)&desc_device;
}

//--------------------------------------------------------------------
// Configuration Descriptor
//--------------------------------------------------------------------

#define TUD_VIDEO_CAPTURE_DESCRIPTOR_UNCOMPR_BULK(_stridx, _epin, _width, _height, _fps, _epsize) \
  TUD_VIDEO_DESC_IAD(ITF_NUM_VIDEO_CONTROL, 0x02, _stridx), \
  TUD_VIDEO_DESC_STD_VC(ITF_NUM_VIDEO_CONTROL, 0, _stridx), \
    TUD_VIDEO_DESC_CS_VC(0x0150, TUD_VIDEO_DESC_CAMERA_TERM_LEN + TUD_VIDEO_DESC_OUTPUT_TERM_LEN, UVC_CLOCK_FREQUENCY, ITF_NUM_VIDEO_STREAMING), \
      TUD_VIDEO_DESC_CAMERA_TERM(UVC_ENTITY_CAP_INPUT_TERMINAL, 0, 0, 0, 0, 0, 0), \
      TUD_VIDEO_DESC_OUTPUT_TERM(UVC_ENTITY_CAP_OUTPUT_TERMINAL, VIDEO_TT_STREAMING, 0, 1, 0), \
  TUD_VIDEO_DESC_STD_VS(ITF_NUM_VIDEO_STREAMING, 0, 1, _stridx), \
    TUD_VIDEO_DESC_CS_VS_INPUT(1, \
        TUD_VIDEO_DESC_CS_VS_FMT_UNCOMPR_LEN + TUD_VIDEO_DESC_CS_VS_FRM_UNCOMPR_CONT_LEN + TUD_VIDEO_DESC_CS_VS_COLOR_MATCHING_LEN, \
        _epin, 0, UVC_ENTITY_CAP_OUTPUT_TERMINAL, \
        0, 0, 0, \
        0), \
      TUD_VIDEO_DESC_CS_VS_FMT_UNCOMPR(1, 1, TUD_VIDEO_GUID_YUY2, 16, 1, 0, 0, 0, 0), \
        TUD_VIDEO_DESC_CS_VS_FRM_UNCOMPR_CONT(1, 0, _width, _height, \
            _width * _height * 16, _width * _height * 16 * _fps, \
            _width * _height * 16 / 8, \
            (10000000/_fps), (10000000/_fps), 10000000, (10000000/_fps)), \
        TUD_VIDEO_DESC_CS_VS_COLOR_MATCHING(VIDEO_COLOR_PRIMARIES_BT709, VIDEO_COLOR_XFER_CH_BT709, VIDEO_COLOR_COEF_SMPTE170M), \
        TUD_VIDEO_DESC_EP_BULK(_epin, _epsize, 1)

uint8_t const desc_fs_configuration[] = {
    TUD_CONFIG_DESCRIPTOR(1, ITF_NUM_TOTAL, 0, CONFIG_TOTAL_LEN, 0x80, 500),

    // CDC Interface
    TUD_CDC_DESCRIPTOR(ITF_NUM_CDC, 4, EPNUM_CDC_NOTIF, 8, EPNUM_CDC_OUT, EPNUM_CDC_IN, 64),

    // UVC Interface
    TUD_VIDEO_CAPTURE_DESCRIPTOR_UNCOMPR_BULK(5, EPNUM_VIDEO_IN, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE, 64),
};

uint8_t const* tud_descriptor_configuration_cb(uint8_t index) {
    (void)index;
    return desc_fs_configuration;
}

//--------------------------------------------------------------------
// String Descriptors
//--------------------------------------------------------------------
static char const* string_desc_arr[] = {
    (const char[]){0x09, 0x04}, // 0: Language (English)
    "Proton",                   // 1: Manufacturer
    "OV7670 UVC Camera",        // 2: Product
    "123456",                   // 3: Serial
    "Proton CDC",               // 4: CDC Interface
    "Proton UVC",               // 5: UVC Interface
};

static uint16_t _desc_str[32 + 1];

uint16_t const* tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
    (void)langid;
    size_t chr_count;

    switch (index) {
        case 0:
            memcpy(&_desc_str[1], string_desc_arr[0], 2);
            chr_count = 1;
            break;
        default:
            if (index >= sizeof(string_desc_arr) / sizeof(string_desc_arr[0])) return NULL;
            const char* str = string_desc_arr[index];
            chr_count = strlen(str);
            if (chr_count > 31) chr_count = 31;
            for (size_t i = 0; i < chr_count; i++) {
                _desc_str[1 + i] = str[i];
            }
            break;
    }

    _desc_str[0] = (uint16_t)((TUSB_DESC_STRING << 8) | (2 * chr_count + 2));
    return _desc_str;
}
