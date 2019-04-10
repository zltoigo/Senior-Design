# reorganized Bluetooth tests into a class

import time, serial
from sys import stdout
import cv2

class BluetoothManager:

    def __init__(self, portName, baudRate):
        # Open connection to Arduino with a timeout of two seconds
        self. port = serial.Serial(portName, baudRate, timeout=2)

        # Report acknowledgment from camera
        stdout.write(self.port.readline().decode())

        # Wait a spell
        time.sleep(1)

        # Send start flag
        self.port.write(bytearray([1]))

        self.imageCounter = -2

        self.prepareForNewImage()

    def prepareForNewImage(self):
        # Increment counter
        self.imageCounter = self.imageCounter + 1

        # Open image file to begin writing to
        self.tmpfile = open("img" + str(self.imageCounter) + ".jpg", "wb")

        # Initialize start variables
        self.prevbyte = None

    def waitForNewImage(self):
        # Read a byte from Arduino
        currbyte = self.port.read(1)
        if self.prevbyte:
            if ord(currbyte) == 0xd8 and ord(self.prevbyte) == 0xff:
                print("Picture start!")
                self.tmpfile.write(self.prevbyte)
                return True
        self.prevByte = currbyte

    def createImageFromStream(self):
          while True:
            # Read a byte from Arduino
            currbyte = self.port.read(1)

            self.tmpfile.write(currbyte)

            # End-of-image sentinel bytes: close temp file and display its contents
            if ord(currbyte) == 0xd9 and ord(self.prevbyte) == 0xff:
                self.tmpfile.close()
                print("Picture taken!")
                imagePath = "image" + str(self.imageCounter) + ".jpg" 
                self.prepareForNewImage()
                return (imagePath)

            # Track previous byte
            self.prevbyte = currbyte

   



