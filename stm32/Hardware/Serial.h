#ifndef __SERIAL_H
#define __SERIAL_H

#define RX_PACKET_SIZE 32

extern  uint8_t Serial_RxFlag;
extern char Serial_RxPacket[RX_PACKET_SIZE];


void Serial_Init(void);
void USART1_IRQHandler(void);


#endif