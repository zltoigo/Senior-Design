#include <SoftwareSerial.h>
SoftwareSerial EEBlue(9, 10); 

void setup()  
{
  // set digital pin to control as an output
  pinMode(6, OUTPUT);
  // set the data rate for the SoftwareSerial port
  EEBlue.begin(9600);
  // Send test message to other device
  EEBlue.println("Hello from Arduino");
}
char a; // stores incoming character from other device
void loop() 
{
  if (EEBlue.available())
  // if text arrived in from BT serial...
  {
    a=(EEBlue.read());
    if (a=='1')
    {
      digitalWrite(13, HIGH);
      EEBlue.println("LED on");
    }
    if (a=='2')
    {
      digitalWrite(13, LOW);
      EEBlue.println("LED off");
    }
    if (a=='?')
    {
      EEBlue.println("Send '1' to turn LED on");
      EEBlue.println("Send '2' to turn LED on");
    }   
    // you can add more "if" statements with other characters to add more commands
  }
}
