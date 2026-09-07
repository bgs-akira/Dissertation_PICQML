#include <SPI.h>

#define USB_RATE 19200//usb transfer rate
#define SPI_CLOCK 18e6//spi clock frequency
#define ADC_RES 262144.0//adc bits 2^18
#define VREF 4.096// from ADS8698

// teensy pin definitions
#define _CS0 10
#define _CS1 9
#define _CS0S 14
#define _CS1S 15
#define _RST 4

SPISettings ADC_SPI(SPI_CLOCK, MSBFIRST, SPI_MODE1);

// global variable definitions
double VRANGE = 1.25 * VREF;
double v_meas[8];
char prefix;
int cs_map[4] = {10, 9, 14, 15};
int order[8] = {2,  3,  4,  5,  1,  0,  7,  6};
double v_meas_sorted[8];

void setup()
{
  /*
  Configure the lightGRID for operation. Executed before all functions.
  */

  // prepare serial protocols
  Serial.begin(USB_RATE);
  SPI.begin();

  // define pin polarities
  pinMode(_CS0, OUTPUT);
  pinMode(_CS1, OUTPUT);
  pinMode(_CS0S, OUTPUT);
  pinMode(_CS1S, OUTPUT);
  pinMode(_RST, OUTPUT);

  // set starting pin states
  digitalWrite(_CS0, HIGH);
  digitalWrite(_CS1, HIGH);
  digitalWrite(_CS0S, HIGH);
  digitalWrite(_CS1S, HIGH);

  // reset the chip
  digitalWrite(_RST, HIGH);
  digitalWrite(_RST, LOW);
  digitalWrite(_RST, HIGH);

  // set voltage range registers
  for(int i = 0; i < 4; i++)
    configure_range(i, 4);
}

void startup()
{
  /*
  Reset the lightGRID board to its initial state.
  */

  // set starting pin states
  digitalWrite(_CS0, HIGH);
  digitalWrite(_CS1, HIGH);
  digitalWrite(_CS0S, HIGH);
  digitalWrite(_CS1S, HIGH);

  // reset the chip
  digitalWrite(_RST, HIGH);
  digitalWrite(_RST, LOW);
  digitalWrite(_RST, HIGH);

  // set voltage range registers
  for(int i = 0; i < 4; i++)
    configure_range(i, 4);
}

void configure_range(int chip_num, int mode)
{
  /*Configure the ADC input range. This sets programmable gain
  amplifiers within the ADC chip and modifies the minimum and
  maximum voltages that can be sensed.

  reading material:
  send receive program commands page 48
  program register map page 50.

  // mode definitions
  0: +/- 2.5 * VREF
  1: +/- 1.25 * VREF
  2: +/- 0.625 * VREF
  3: 0->2.5 * VREF
  4: 0->1.25 * VREF
  */

  byte range_code;
  byte read_range_code;

  // change VRANGE variable
  if (mode == 4)
    VRANGE = 1.25 * VREF;
  else if (mode == 3)
    VRANGE = 2.5 * VREF;
  else
    VRANGE = 1.25 * VREF; // FIX ME, this is temporary

  // look up the range setting

  if(mode == 0)
    range_code = 0x00;
  else if(mode == 1)
    range_code = 0x01;
  else if(mode == 2)
    range_code = 0x02;
  else if(mode == 3)
    range_code = 0x05;
  else if(mode == 4)
    range_code = 0x06;
  else
    range_code = 0x00;

  // ADC channel address definitions
  byte addrs[8];
  addrs[0] = 0b00001011;// 0000, 101, 1
  addrs[1] = 0b00001101;// 0000, 110, 1
  addrs[2] = 0b00001111;// 0000, 111, 1
  addrs[3] = 0b00010001;// 0001, 000, 1
  addrs[4] = 0b00010011;// 0001, 001, 1
  addrs[5] = 0b00010101;// 0001, 010, 1
  addrs[6] = 0b00010111;// 0001, 011, 1
  addrs[7] = 0b00011001;// 0001, 100, 1

  // program the range registers
  SPI.beginTransaction(ADC_SPI);
  for(int i = 0; i < 8; i++)
  {
    digitalWrite(cs_map[chip_num], LOW);
    SPI.transfer(addrs[i]); // send the address and write cmd
    SPI.transfer(range_code); // send the range register value
    read_range_code = SPI.transfer(0x00);// device reads back what was written
    digitalWrite(cs_map[chip_num], HIGH);// end the transaction
  }
  SPI.endTransaction();
}

double code_to_volts(int32_t code, int mode)
{
  if(mode == 0)
    return 0.;
  else if(mode == 1)
    return 0.;
  else if(mode == 2)
    return 0.;
  else if(mode == 3)
    return 0.;
  else if(mode == 4)
    return 0.;
  else
    return 0.;
}

double* read_ADC(int chip_num, int N)
{
  /*
  Read the voltage and current for every one of the ADC channels.
  chip_num: which of the two ADCs do you want to read? [0, 1]
  N: how many averages do you want to take? in [Z]
  */

  // initialize v_meas variable
  for(int i = 0; i < 8; i++)
  {
    v_meas[i] = 0;
    v_meas_sorted[i] = 0;
  }

  // enter the auto channel enable with reset mode
  SPI.beginTransaction(ADC_SPI);
  digitalWrite(cs_map[chip_num], LOW);// slave select disable
  SPI.transfer(0xA0);// auto_rst cmd (1st byte)
  SPI.transfer(0x00);// auto_rst cmd (2nd byte)
  SPI.transfer(0x00);// noop
  SPI.transfer(0x00);// noop
  digitalWrite(cs_map[chip_num], HIGH);// slave select enable

  // read out the data
  for(int i = 0; i < 8 * N; i++)
  {
    // wait for the conversion for 16 clock cycles
    digitalWrite(cs_map[chip_num], LOW);
    SPI.transfer(0x00);// noop
    SPI.transfer(0x00);// noop

    // get the data (see page 54)
    int32_t code = 0;
    code = (int32_t)(SPI.transfer(0x00)) << 10; // bits 17-10
    code += (int32_t)(SPI.transfer(0x00)) << 2; // bits 9-2
    code += SPI.transfer(0x00) >> 6; // bits 1-0 and 6 trash bits

    // convert the binary code to voltage units
    v_meas[i%8] += (double) VRANGE * code/ADC_RES;

    // end the current channel read
    digitalWrite(cs_map[chip_num], HIGH);
  }

  SPI.endTransaction();

  for(int i = 0; i < 8; i++)
    v_meas_sorted[i] = v_meas[order[i]] / N;

  return v_meas_sorted;
}

void process_input()
{
  /* interpret user serial commands */

  if (Serial.available() > 0)
  {
    prefix = Serial.read();

    if (prefix == '*')
    {
      Serial.println("lightGRID 2.0");
    }

    if (prefix == 'p')
    {
      int avgs = Serial.parseInt();

      for(int i = 0; i < 4; i++)
      {
        double* adc = read_ADC(i, avgs);

        for(int j = 0; j < 8; j++)
        {
          Serial.print(adc[j], DEC);
          Serial.print(',');
        }
      }
      Serial.print("\n");
      Serial.flush();
    }

    if (prefix == 'z')
      startup();

    if (prefix == 'g')
    {
      int setting = Serial.parseInt();
      if (setting == 4 || setting == 3 || setting == 2 || setting == 1 || setting == 0)
      {
        if (setting == 4)

        configure_range(0, setting);
        configure_range(1, setting);
      }
    }

  }
}

void loop()
{
  process_input();
  // delay(100);
  //
  // for(int i = 0; i < 4; i++)
  // {
  //   double* v_meas_adc = read_ADC(i, 100);
  //   for(int j = 0; j < 8; j++)
  //   {
  //     char data_str[32];
  //     sprintf(data_str, "ADC (%d) Channel (%d): %.6f", i, j, v_meas_adc[j]);
  //     Serial.println(data_str);
  //   }
  // }
  // Serial.println();
}
