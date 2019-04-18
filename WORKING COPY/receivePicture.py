#!/usr/bin/env python3
'''
Modified version of jpegstream.py

'''

import time
import serial
import cv2
from sys import stdout

# from helpers import *

# Modifiable params --------------------------------------------------------------------

PORT = 'COM5'         # Bluetooth

BAUD = 38400       # Change to 115200 for Due

# main ------------------------------------------------------------------------------

if __name__ == '__main__':

    # Open connection to Arduino with a timeout of two seconds
    port = serial.Serial(PORT, BAUD, timeout=2)

    # Report acknowledgment from camera
    stdout.write(port.readline().decode())

    # Wait a spell
    time.sleep(1)

    # Send start flag
    port.write(bytearray([1]))

    # Counter for image naming
    i = -1

    # Loop over bytes from Arduino for a single image
    while True:
        # Increment counter
        i = i + 1

        # Open image file to begin writing to
        tmpfile = open("img" + str(i) + ".jpg", "wb")

        # Initialize start variables
        written = False
        prevbyte = None

        # Loop over bytes from Arduino for a single image
        while True:

            # Read a byte from Arduino
            currbyte = port.read(1)

            # If we've already read one byte, we can check pairs of bytes
            if prevbyte:
                # Start-of-image sentinel bytes: write previous byte to temp file
                if ord(currbyte) == 0xd8 and ord(prevbyte) == 0xff:
                    print("Picture start!")
                    tmpfile.write(prevbyte)
                    written = True

                # Inside image, write current byte to file
                if written:
                    tmpfile.write(currbyte)

                # End-of-image sentinel bytes: close temp file and display its contents
                if ord(currbyte) == 0xd9 and ord(prevbyte) == 0xff:
                    tmpfile.close()
                    img = cv2.imread("img" + str(i) + ".jpg")
                    print("Picture taken!")
                    break  

            # Track previous byte
            prevbyte = currbyte

    # Send stop flag
    port.write(bytearray([0]))

