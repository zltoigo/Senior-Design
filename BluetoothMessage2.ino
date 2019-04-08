#include <SoftwareSerial.h>
SoftwareSerial EEBlue(9, 10); // RX | TX

void setup() {
  // put your setup code here, to run once:
  EEBlue.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  EEBlue.print("test");
  //EEBlue.flush();
  //delay(500);
}
