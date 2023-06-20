#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	srand(time(NULL));
	float input[3];
	float weight[2][3][3];
	float output[3] = { 0, };
	printf("----------Input-----------\n");
	for (int i = 0;i < 3;i++)
	{
		input[i] = float(rand())/RAND_MAX;
		
		printf("%f ", input[i]);
	}
	printf("\n\n");
	for (int k = 0;k < 2;k++)
	{
		printf("----------Layer%d Weight-----------\n", k + 1);
		for (int i = 0;i < 3;i++)
		{
			for (int j = 0;j < 3;j++)
			{
				weight[k][i][j] = float(rand()) / RAND_MAX;
				printf("%f ", weight[k][i][j]);
			}
			printf("\n\n");
		}
	}
	for (int k = 0;k < 2;k++)
	{
		printf("----------Layer%d Output-----------\n", k + 1);
		for (int i = 0;i < 3;i++)
		{
			for (int j = 0;j < 3;j++)
			{
				output[i] += input[j] * weight[k][i][j];
			}
			output[i] = 1 / (1 + expf(-output[i]));
			printf("%f\n\n", output[i]);
		}
		for (int i = 0;i < 3;i++)
		{
			input[i] = output[i];
			output[i] = 0;
		}
	}
}
