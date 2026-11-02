#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "OLED.h"
#include "Serial.h"
#include <string.h>
int main(void)
{		
	OLED_Init();
	Serial_Init();
	
	
		while(1)
		{
		  if (Serial_RxFlag == 1)
		  {
			if (strcmp(Serial_RxPacket, "COIN,1Y") == 0)
			{
				OLED_Clear();
				OLED_ShowString(1, 1, "Coin: 1 Yuan");
			}
			else if (strcmp(Serial_RxPacket, "COIN,1J") == 0)
			{
				OLED_Clear();
				OLED_ShowString(1, 1, "Coin: 1 JIAO");
			}
			else if (strcmp(Serial_RxPacket, "COIN,5J") == 0)
			{
				OLED_Clear();
				OLED_ShowString(1, 1, "Coin: 5 JIAO");
			}
			Serial_RxFlag = 0;
		  }
		}

}