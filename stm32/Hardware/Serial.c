#include "stm32f10x.h"                  // Device header
#include "Serial.h"


uint8_t Serial_RxFlag = 0;
char Serial_RxPacket[RX_PACKET_SIZE];


void Serial_Init(void)
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	USART_InitTypeDef USART_InitStructure;
	USART_InitStructure.USART_BaudRate = 9600;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
	USART_InitStructure.USART_Parity = USART_Parity_No;
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	USART_Init(USART1, &USART_InitStructure);
	
	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
	
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	
	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
	NVIC_Init(&NVIC_InitStructure);
	
	USART_Cmd(USART1, ENABLE);
}

  void USART1_IRQHandler(void)
  {
	  static uint8_t index = 0;
	if (USART_GetITStatus(USART1, USART_IT_RXNE) == SET)
	{
		char data = (char)USART_ReceiveData(USART1);
			if (data == '\n')
			{
				Serial_RxPacket[index] = '\0';  //换行符代表这一包结束，发送结束字符串
				Serial_RxFlag = 1;				//标志位置一告诉主循环这一包数据结束
				index = 0;
			}
			else if (Serial_RxFlag == 0)       //如果标志位没有置一，就没有结束，继续接受
			{
				if (index < RX_PACKET_SIZE - 1)     //-1 是为了给‘\0’留位置
					{
						Serial_RxPacket[index ++] = data;
					}
				else
				{
					index = 0;							//数据包太长，丢弃
				}
			
			}
			
		else
		{
			
		}
	USART_ClearITPendingBit(USART1, USART_IT_RXNE);	
	}
  }



