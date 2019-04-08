
//#include <SoftwareSerial.h>
//SoftwareSerial Bluetooth(9, 10); // RX | TX

#include <Wire.h>
#include <BreezyArduCAM.h>
#include <SPI.h>

static const int CS = 7;
int switchPin = 4;
int oldSwitchState = HIGH;

Serial_ArduCAM_FrameGrabber fg;

ArduCAM_Mini_2MP myCam(CS, &fg);

void setup(void) 
{
    pinMode(switchPin, INPUT_PULLUP);
    // ArduCAM Mini uses both I^2C and SPI buses
    Wire.begin();
    SPI.begin();
    
    // Baud rate
    Serial.begin(38400);    
    // Start the camera in JPEG mode with a specific image size
    myCam.beginJpeg800x600();
}

void loop(void) 
{
    //Serial.print("HELLO!");
    myCam.capture();
    int switchState = digitalRead(switchPin);
    if (switchState != oldSwitchState)
    {
      oldSwitchState = switchState;
      if (switchState == HIGH)
      {
        myCam.capture();   
      }
    }
}
