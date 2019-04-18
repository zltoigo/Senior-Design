#include <SoftwareSerial.h>
SoftwareSerial BTSerial(10, 11); // RX | TX

void setup()
{
  Serial.begin(9600);
  BTSerial.begin(38400);
}

void loop()
{
  BTSerial.println("Hello world!");
  //BTSerial.println("Hello World");
  //BTSerial.write(0x45);
  delay(5000);



}
