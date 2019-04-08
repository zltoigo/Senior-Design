import serial
import time
import cv2

# get serial
serOut=serial.Serial("COM6", 9600, timeout=2)
#ser = serial.Serial("COM2", 9600, timeout=2)

#print(serIn.inWaiting())

#ser.flushInput()
serOut.flushInput()
time.sleep(0.2)
while True:
    # data = ser.read()
    # print(data)

    output_data=serOut.read()
    print(output_data)
    # while True:
    #     ch = serIn.read(1).decode('utf8')
    #     print(ch)
    # print(chstring)
    #print(input_data)
    time.sleep(0.1)
    #serIn.close()