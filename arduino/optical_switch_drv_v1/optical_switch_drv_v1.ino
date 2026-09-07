// Arduino Nano 33 loT
// define the serial pins
// 1,9 GND
// 12, vcc1, external 5V, 1000mA
// 15, vcc2, arduino 5V, 50mA 
const int ReadyPin = 7;
const int ErrorPin = 8;
const int ResetPin = 11;
const int ChannelPin[6] = {2, 3, 4, 5, 6, 10};

// status value
int ReadyState = LOW;
int ErrorState = LOW;

char read[20];

void setup() {
  Serial.begin(9600);
  pinMode(ReadyPin, INPUT);
  pinMode(ErrorPin, INPUT);
  pinMode(ResetPin, OUTPUT);
  for (int i = 0; i <= 5; i = i + 1) {
    pinMode(ChannelPin[i], OUTPUT);
  }
  for (int i = 0; i < 5; i++) {
    digitalWrite(ChannelPin[i], LOW); delay(100);
  }
} // end of setup


void loop() {
  delay(1000);
  if ( Serial.available() > 0 ) {
    // return input 
    Serial.readBytesUntil('\n', read, 20); 
    Serial.print("Your input:");
    Serial.println(read);
    // char channel[6];

    // set port D0-D5
    if (read[0] == 's'){
      digitalWrite(ResetPin, HIGH);
      for (int i = 0; i <= 5; i = i + 1) {
        if (read[i+1] == '0') {
          digitalWrite(ChannelPin[i], LOW);
        }
        else if (read[i+1] == '1') {
          digitalWrite(ChannelPin[i], HIGH);
          // COM += bit(i);
        }
      }
    }
  
    // query the status
    if (read[0]  == '?') {
      ReadyState = digitalRead(ReadyPin);
      ErrorState = digitalRead(ErrorPin);
      Serial.print(ReadyState);
      Serial.print('\t');
      Serial.println(ErrorState);
      // Serial.println(ErrorState+ReadyState);
    }

    // reset
    if (read[0]  == 'r') {
      digitalWrite(ResetPin, LOW);
      delay(1000);
      // digitalWrite(ResetPin, HIGH);
      // Serial.println('Reset\n');
      // Serial.flush();
    }

    
  } // end of SPI
} // end of loop
