#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	float input[2] = { 1.0,0.5 };
	float weight[2][2] = { {0.9,0.3},{0.2,0.8} };
	float x[2] = { 0, };
	float output[2];
	for (int i = 0;i < 2;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			x[i] += input[j] * weight[i][j];
		}
		output[i] = 1 / (1 + expf(-x[i]));
		printf("%f\n", output[i]);
	}
}
