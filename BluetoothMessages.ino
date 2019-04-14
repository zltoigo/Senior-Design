float flow = 500.06;

void setup() {
  Serial.begin(9600);
}
void loop() {
    if (Serial.available()>0) {
        //if (Serial.read() == 'R') {
             Serial.print("q");
             Serial.write("q");
        //}
    }
}
