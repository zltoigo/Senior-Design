'''
Main file for product demo. Designed for single runs
'''

#import python libraries
import time, serial
from sys import stdout

#import 3rd party libraries
import pyttsx3          #text to speech

#import classes
from BluetoothManager import BluetoothManager

if __name__ == '__main__':
    bluetoothManager = BluetoothManager(portName = 'COM5', baudRate = 38400)
    dnnModel = setupModel()
    ttsEngine = pyttsx3.init() 

    #infinite loop waits for bytes from Arduino
    while True:
        didReceiveStartBit = bluetoothManager.waitForNewImage()

        if didReceiveStartBit:
            imagePath = bluetoothManager.createImageFromStream()
            
            #infer returns None if no text found in image
            recognizedText = dnnModel.infer()
            
            if recognizedText:
                ttsEngine.say("You are facing " + recognizedText)
            else:
                ttsEngine.say("Could not read text, please take another picture.")


