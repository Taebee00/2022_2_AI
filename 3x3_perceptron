#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	float input[3] = { 0.9,0.1,0.8 };
	float weight[2][3][3] = { { {0.9,0.3,0.4},{0.2,0.8,0.2},{0.1,0.5,0.6} },
							{ {0.3,0.7,0.5},{0.6,0.5,0.2},{0.8,0.1,0.9} } };
	float output[3] = { 0, };
	for (int k=0;k<2;k++)
	{
		printf("----------layer%d output-----------\n", k + 1);
		for (int i = 0;i < 3;i++)
		{
			for (int j = 0;j < 3;j++)
			{
				output[i] += input[j] * weight[k][i][j];
			}
			output[i] = 1 / (1 + expf(-output[i]));
			printf("%f\n", output[i]);
		}		
		for (int i = 0;i < 3;i++)
		{
			input[i] = output[i];
			output[i] = 0;
		}
	}
}
