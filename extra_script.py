Import("env")
import os

project_dir = env.subst("$PROJECT_DIR")

# Add our include/ dir globally so framework sources (pico_stdio_usb, etc.)
# can find tusb_config.h
# Disable SDK's default USB descriptors so we can provide our own composite ones
env.Append(
    CPPPATH=[os.path.join(project_dir, "include")],
    CPPDEFINES=[
        ("PICO_STDIO_USB_USE_DEFAULT_DESCRIPTORS", "0"),
    ]
)

# Link pre-compiled pico-tflmicro static library
tflm_lib_path = os.path.join(project_dir, "lib", "pico-tflmicro", "build")
tflm_lib_file = os.path.join(tflm_lib_path, "libpico-tflmicro.a")
if os.path.exists(tflm_lib_file):
    env.Append(
        LIBPATH=[tflm_lib_path],
        LIBS=["pico-tflmicro"],
    )
    print(f"TFLM: Linked {tflm_lib_file}")
else:
    print(f"WARNING: TFLM library not found at {tflm_lib_file}")
    print("  Build it first: cd lib/pico-tflmicro/build && make -j4")
