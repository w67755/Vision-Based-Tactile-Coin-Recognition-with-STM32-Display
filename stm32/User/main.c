#include "stm32f10x.h"                  // Device header
#include "OLED.h"
#include "Serial.h"
#include <string.h>

int main(void)
{
	OLED_Init();
	Serial_Init();

	OLED_ShowString(1, 1, "Place a shape");

	while (1)
	{
		if (Serial_RxFlag == 1)
		{
			if (strcmp(Serial_RxPacket, "SHAPE,SPHERE") == 0)
			{
				OLED_Clear();
				OLED_ShowString(1, 1, "Shape: Sphere");
			}
			else if (strcmp(Serial_RxPacket, "SHAPE,CYLINDER") == 0)
			{
				OLED_Clear();
				OLED_ShowString(1, 1, "Shape: Cylinder");
			}
			else if (strcmp(Serial_RxPacket, "SHAPE,CUBE") == 0)
			{
				OLED_Clear();
				OLED_ShowString(1, 1, "Shape: Cube");
			}
			else
			{
				OLED_Clear();
				OLED_ShowString(1, 1, "Unknown shape");
			}

			Serial_RxFlag = 0;
		}
	}
}
