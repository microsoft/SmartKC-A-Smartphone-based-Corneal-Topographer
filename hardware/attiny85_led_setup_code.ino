// SmartKC dev board code
#include <Adafruit_NeoPixel.h>

// Pin on the ATtiny85 board connected to the LED
#define PIN            1

// Number of LED's connected to the ATtiny85 board
#define NUMPIXELS      12

// To setup the LED ring library (NeoPixel), we pass number of LED's and pin number to send signals to as parameters.
// Note: For older NeoPixel LED rings, we might need to change the third parameter.
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_RGB + NEO_KHZ800);

// Delay for half a second
int delayval = 500; 

void setup() {
  // Initializes the NeoPixel library
  pixels.begin(); 
}

void loop() {
  // For a set of NeoPixel LEDs, the first NeoPixel is 0, second is 1, all the way up to the count of pixels minus one.
  for(int i=0; i<NUMPIXELS; i++){
    // pixels.Color takes RGB values, from 0,0,0 up to 255,255,255 in the GRB order
    pixels.setPixelColor(i, pixels.Color(255,200,255));
    // Send the updated pixel color to the hardware.
    pixels.show();
    // Delay for a period of time (in milliseconds).
    delay(delayval); 
  }
}
