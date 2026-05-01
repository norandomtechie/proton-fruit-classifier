import cv2
import usb.core
import usb.util
import time
import logging
import sys
import os
import glob

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Constants for the custom camera
VENDOR_ID = 0xcafe
PRODUCT_ID = 0x4007

def find_video_device():
    """Find candidate /dev/videoX nodes for our USB VID:PID."""
    matches = []
    for devpath in sorted(glob.glob('/sys/class/video4linux/video*')):
        devname = os.path.basename(devpath)
        # Follow the device symlink to find the USB device
        try:
            uevent_path = os.path.join(devpath, 'device', '..', '..', 'uevent')
            uevent_path = os.path.realpath(uevent_path)
            if not os.path.exists(uevent_path):
                # Try alternate path for different sysfs layouts
                uevent_path = os.path.join(devpath, 'device', 'uevent')
            with open(uevent_path) as f:
                uevent = f.read()
            # Look through parent directories for idVendor/idProduct
        except Exception:
            pass

        # Check idVendor/idProduct in parent USB device
        try:
            realpath = os.path.realpath(devpath)
            # Walk up to find the USB device with idVendor/idProduct
            parts = realpath.split('/')
            for i in range(len(parts), 2, -1):
                parent = '/'.join(parts[:i])
                vid_path = os.path.join(parent, 'idVendor')
                pid_path = os.path.join(parent, 'idProduct')
                if os.path.exists(vid_path) and os.path.exists(pid_path):
                    with open(vid_path) as f:
                        vid = int(f.read().strip(), 16)
                    with open(pid_path) as f:
                        pid = int(f.read().strip(), 16)
                    if vid == VENDOR_ID and pid == PRODUCT_ID:
                        node = f"/dev/{devname}"
                        matches.append(node)
        except Exception:
            continue

    return matches


def open_matching_device():
    """Open the first working V4L2 device node for our VID:PID."""
    candidates = find_video_device()
    if not candidates:
        return None, None

    for node in candidates:
        logging.info(f"Trying camera node {node}")
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        if cap.isOpened():
            logging.info(f"Stream opened successfully on {node}.")
            return cap, node
        cap.release()

    return None, None

def is_device_present():
    """Checks the USB bus for the specific VID/PID."""
    device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    return device is not None

def main():
    cap = None
    device_active = False

    logging.info(f"Starting monitor for device {hex(VENDOR_ID)}:{hex(PRODUCT_ID)}...")

    try:
        while True:
            present = is_device_present()

            # Case 1: Device was just reconnected
            if present and not device_active:
                logging.info("Device detected! Waiting for OS to initialize driver...")
                time.sleep(2.0)  # Buffer for the OS to create the device node
                
                candidates = find_video_device()
                if not candidates:
                    logging.error("USB device present but no /dev/videoX found for it. Retrying...")
                    time.sleep(1.0)
                    continue
                logging.info(f"Found candidate nodes: {', '.join(candidates)}")

                cap, opened_node = open_matching_device()
                if cap and cap.isOpened():
                    device_active = True
                else:
                    logging.error("Failed to open any matching V4L2 camera node. Retrying...")
                    if cap:
                        cap.release()

            # Case 2: Device was disconnected
            elif not present and device_active:
                logging.warning("Device lost. Closing stream and cleaning up...")
                if cap:
                    cap.release()
                cv2.destroyAllWindows()
                device_active = False

            # Case 3: Streaming mode
            if device_active and cap:
                ret, frame = cap.read()
                if ret:
                    larger_img = cv2.resize(frame, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
                    cv2.imshow(f"OV7670 UVC - {hex(VENDOR_ID)}:{hex(PRODUCT_ID)}", larger_img)
                    # Press 'q' in the window to exit the script
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("User requested exit.")
                        break
                else:
                    logging.error("Failed to grab frame. Checking hardware status...")
                    # This often happens during "soft" disconnects
                    device_active = False
                    cap.release()

            # Idle polling interval to save CPU
            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user.")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    main()
