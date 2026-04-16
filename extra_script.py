Import("env")

# Add our include/ dir globally so framework sources (pico_stdio_usb, etc.)
# can find tusb_config.h
# Disable SDK's default USB descriptors so we can provide our own composite ones
env.Append(
    CPPPATH=[env.subst("$PROJECT_DIR") + "/include"],
    CPPDEFINES=[
        ("PICO_STDIO_USB_USE_DEFAULT_DESCRIPTORS", "0"),
    ]
)
