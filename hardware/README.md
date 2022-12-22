# Phone Attachment

## Requirement:
1. **Placido head**: 3D-print `placido_head.stl` by following the 3D-print related instructions [below](#placido-head)
1. **Placido base**: 3D-print `placido_base.stl` by following the 3D-print related instructions [below](#placido-base)
1. **Diffuser**: 3D-print `diffuser.stl` by following the 3D-print related instructions [below](#diffuser)
1. **Circular LED**: Use WS2812/WS2812B 12-bit RGB LED Ring (available for Rs 250 or 3$ on DigiKey, [Robocraze](https://robocraze.com/products/ws2812-12-bit-rgb-led-round), [ThinkRobotics](https://thinkrobotics.in/products/ws2812-5050-rgb-led-ring), etc.)
1. **Development board**: Use ATtiny85 (Digispark) USB Development Board (available for Rs 325 or 4$ on DigiKey, [Robocraze](https://robocraze.com/products/attiny85-usb-development-board), [Robu](https://robu.in/product/attiny85-usb-development-board/), etc.)
1. **Source code for Development board**: Use `attiny85_led_setup_code.ino` code and upload it in the ATtiny85 development board by following the instructions [below](#attiny85-development-board-related-instructions)
1. **OTG cable**: To connect ATtiny85 Development Board with the smartphone charging input, use a USB-A Female to USB-B/C Male cable (available for Rs 200 or 2.5$ on [Amazon](https://www.amazon.in/gp/product/B012V56C8K))
1. **Double-sided tape**: Use 3M Scotch Double Sided Foam Tape (available for Rs 100 or 1.2$ on [Amazon](https://www.amazon.in/gp/product/B00N1U9AJS/))
1. **Smartphone Case**: Specific to your smartphone
1. **Smartphone**: Please check the smartphone specification related requirement [below](#smartphone-specifications)

Note: Kindly follow the guidelines mentioned in the `guidelines_for_data_collector.pdf` document to collect high-quality data from the SmartKC device.

## Ring Distribution
File `./ring_distribution.txt` contains the radius of each ring and its height from the base.

## 3D-print Instructions:

### Placido head:
MJF (MultiJet Fusion) print with PA-12 material in Stone Grey color (Matt finish). We used the default settings.

### Placido base:
MJF (MultiJet Fusion) print with PA-12 material in Stone Grey color (Matt finish). We used the default settings.

### Diffuser:
Print using PLA material (white filament): [1.75mm PLA_3D Filament White 1KG](https://robu.in/product/esun-pla-1-75mm-3d-printing-filament-1kg-white/). White-colored filament is required to achieve translucency. Note: Convert the `.stl` file to `.gx` file before printing.

We used the FlashForge Finder 3D printer with [FlashPrint](https://flashforge-usa.com/pages/download) software (version 4.20, 64 Bits), with the following settings: 
* First Layer Height: 0.25 mm, 
* Layer Height: 0.20 mm, 
* Infill: 30%, 
* Fill pattern: Hexagonal, 
* Temperature: 225C,
* Resolution: High.

## ATtiny85 Development Board-related Instructions:
1. Download Arduino IDE (.exe file) from [here](https://www.arduino.cc/en/software) on Windows 7 or newer, and install it.
1. To program Digispark ATtiny85 development board, follow the instructions from [here](http://digistump.com/wiki/digispark/tutorials/connecting)
    1. Download device drivers from [here](https://github.com/digistump/DigistumpArduino/releases/download/1.6.7/Digistump.Drivers.zip). Unzip and run `DPInst64.exe`
    1. Open the installed Arduino IDE. Go to the `File`→`Preferences` menu. In the box labeled `Additional Boards Manager URLs` enter: http://digistump.com/package_digistump_index.json.
    1. Go to the `Tools`→`Board`→`Boards Manager` menu. From the `Type` drop-down, select `Contributed`. Select the `Digistump AVR Boards` package and click the `Install` button. After the installation completes, close the `Boards Manager` window.
    1. Go to the `Tools`→`Boards`→`Digistump AVR Boards` and select `Digispark (Default - 16.5mhz)` board.
    1. DO NOT CONNECT THE USB BOARD TO YOUR COMPUTER.
    1. Copy the [test code](http://digistump.com/wiki/digispark/tutorials/connecting) into Arduino IDE. Click the `Upload` button. When asked connect the USB board, and see it working.
1. Copy the `attiny85_led_setup_code.ino` source code into Arduino IDE. Click the `Upload` button. 

## Smartphone Specifications:
* Android smartphone with 12 MP (or more) back camera with a minimum focusing distance of 75mm (or less).
* Devices we have tested SmartKC to work well on: OnePlus 7T, OnePlus 6T, Samsung Galaxy A52s 5G, Xiaomi Pocophone F1.
* Devices we have tested SmartKC to not work on: OnePlus 9 5G, Redmi 9 Power, Pixel 4a. Reason: minimum focusing distance too high.

## Assembly Instructions (putting it all together):
1. Upload the `attiny85_led_setup_code.ino` code to the `Development board`.
1. Connect the `Circular LED` to the `Development board` by soldering ATtiny85 GND to WS2812 GND, ATtiny85 5V to WS2812 5V, and ATtiny85 P1 to WS2812 D1.
1. Use `Double-sided tape` to attach the `Circular LED` on the `Placido base` (aligning it with the center).
1. Use `Double-sided tape` to attach the `Development board` on the `Placido base` (inside the igloo shaped enclosure)
1. Use `Double-sided tape` to attach the `Placido base` over the `Smartphone Case`.
   * Important: Make sure to align the hole on the `Placido base` with the smartphone's main back camera.
   * Also, orient the `Placido base` such that the `Development board` enclosure aligns vertically with the phone.
1. Connect the `Development board` to the smartphone using the `OTG cable`.
1. Place the `Diffuser` inside the `Placido head`.
1. Attach the `Placido head` (with `Diffuser` inside it) to the `Placido base` by placing the three pegs and rotating it.