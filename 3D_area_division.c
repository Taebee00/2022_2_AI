#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// 상수값 정의 
#define INPUT_LAYER 3
#define HIDDEN_LAYER_1 5
#define HIDDEN_LAYER_2 3
#define OUTPUT_LAYER 3
#define LEARN_RATE 0.5
#define LEARN_DATA 300
#define RIGHT 0.6
#define WRONG 0.4
#define TIMES 5
#define TEST_CASE 20

// 각 노드의 입력, 출력 배열, 목표 배열 선언
float input_arr[LEARN_DATA][3];
float target_arr[LEARN_DATA][3];
float* i_output;
float j_input[HIDDEN_LAYER_1], j_output[HIDDEN_LAYER_1], j_error[HIDDEN_LAYER_1];
float k_input[HIDDEN_LAYER_2], k_output[HIDDEN_LAYER_2], k_error[HIDDEN_LAYER_2];
float l_input[OUTPUT_LAYER], l_output[OUTPUT_LAYER], l_error[OUTPUT_LAYER];
float weight_ij[INPUT_LAYER][HIDDEN_LAYER_1], weight_jk[HIDDEN_LAYER_1][HIDDEN_LAYER_2], weight_kl[HIDDEN_LAYER_2][OUTPUT_LAYER], weight_lm[HIDDEN_LAYER_2][OUTPUT_LAYER];

float sum_error = 0;
float sum_sum_error = 0;

// 가장 높은 확률의 색깔을 반환해주는 함수
int find_maxidx(float a, float b, float c)
{
	float max = a;
	int maxidx = 0;
	max = b > a ? b : max;
	maxidx = b > a ? 1 : 0;
	maxidx = c > max ? 2 : maxidx;
	return maxidx;
}

// 가중치 행렬 랜덤 초기화 함수
void rand_weight(int layer1, int layer2, float* weight)
{
	for (int i = 0;i < layer1;i++)
	{
		for (int j = 0;j < layer2;j++)
		{
			weight[(i * layer2) + j] = -1 + (rand() % 2) + (float)rand() / RAND_MAX;
		}
	}
}

// 순전파 함수
void forward(int layer1_num, int layer2_num, float* layer1_output, float* layer2_input, float* layer2_output, float* weight)
{
	for (int i = 0;i < layer2_num;i++)
	{
		layer2_input[i] = 0;
		for (int j = 0;j < layer1_num;j++)
		{
			layer2_input[i] += layer1_output[j] * weight[j * layer2_num + i];
		}
		layer2_output[i] = 1 / (1 + expf(-layer2_input[i]));
	}
}

// 역전파 함수
void backward(int layer1_num, int layer2_num, float* error_1, float* error_2, float* weight, float* output)
{
	for (int i = 0;i < layer2_num;i++)
	{
		error_2[i] = 0;
		for (int j = 0;j < layer1_num;j++)
		{
			error_2[i] += error_1[j] * weight[i * layer2_num + j] * output[i] * (1 - output[i]);
		}
	}
}

int main()
{
	srand(time(NULL));

	// 파이썬 파일 시각화를 위한 csv 파일 만들기
	FILE* fp = fopen("input.csv", "w");
	fprintf(fp, "input_x,input_y,input_z\n");
	FILE* fp2 = fopen("target.csv", "w");
	fprintf(fp2, "target_x,target_y,target_z\n");
	FILE* fp3 = fopen("error.csv", "w");
	fprintf(fp3, "sum_idx,sum_sum_error\n");
	FILE* fp4 = fopen("correct.csv", "w");
	fprintf(fp4, "correct\n");
	FILE* fp5 = fopen("wgraph_ij.csv","w");
	fprintf(fp5, "ij,wij\n");
	FILE* fp6 = fopen("wgraph_jk.csv", "w");
	fprintf(fp6, "jk,wjk\n");
	FILE* fp7 = fopen("wgraph_kl.csv", "w");
	fprintf(fp7, "kl,wkl\n");

	// 가중치 행렬 임의 초기화
	rand_weight(INPUT_LAYER, HIDDEN_LAYER_1, weight_ij);
	rand_weight(HIDDEN_LAYER_1, HIDDEN_LAYER_2, weight_jk);
	rand_weight(HIDDEN_LAYER_2, OUTPUT_LAYER, weight_kl);

	// 300개의 학습 데이터와 목표값 선언
	for (int i = 0;i < LEARN_DATA;i++)
	{
		if (i % 3 == 0)
		{
			input_arr[i][0] = (rand() % 2) + -1 + (float)rand() / RAND_MAX;
			input_arr[i][1] = (rand() % 2) + -1 + (float)rand() / RAND_MAX;
			input_arr[i][2] = (rand() % 2) + -1 + (float)rand() / RAND_MAX;
			target_arr[i][0] = RIGHT;
			target_arr[i][1] = WRONG;
			target_arr[i][2] = WRONG;
		}
		else if (i % 3 == 1)
		{
			input_arr[i][0] = (rand() % 2) - 4 + (float)rand() / RAND_MAX;
			input_arr[i][1] = (rand() % 2) - 4 + (float)rand() / RAND_MAX;
			input_arr[i][2] = (rand() % 2) - 4 + (float)rand() / RAND_MAX;
			target_arr[i][0] = WRONG;
			target_arr[i][1] = RIGHT;
			target_arr[i][2] = WRONG;
		}
		else
		{
			input_arr[i][0] = (rand() % 2) + 2 + (float)rand() / RAND_MAX;
			input_arr[i][1] = (rand() % 2) + 2 + (float)rand() / RAND_MAX;
			input_arr[i][2] = (rand() % 2) + 2 + (float)rand() / RAND_MAX;
			target_arr[i][0] = WRONG;
			target_arr[i][1] = WRONG;
			target_arr[i][2] = RIGHT;
		}
		// 목표값와 출력값도 파이썬 시각화를 위해 csv파일로 저장
		fprintf(fp, "%f,%f,%f\n", input_arr[i][0], input_arr[i][1], input_arr[i][2]);
		fprintf(fp2, "%f,%f,%f\n", target_arr[i][0], target_arr[i][1], target_arr[i][2]);
	}

	// TEST_CASE 20개 선언 
	float test_input[TEST_CASE][3] =
	{
		{0, 0, 0}, { -0.6,0.4,0.5 }, { 0.6,-0.4,0.4 }, { 0.6,-0.6,0.4 }, { 0.8,0.8,-0.8 }, { 0.5,0.5,-0.5 }, { 0.5,0.5,0.5 },
		{-3,-3,-3}, {-2.5,-3.8,-3.4},{-2,-2,-2},{-4,-4,-4},{-2.6,-3.8,-3.8},{-2.5,-3.6,-2.7},
		{3,3,3}, {3.5,3.8,3.8},{3.4,3.4,3.7},{2.6,2.5,2.6},{2.6,3.8,3.8},{2.5,3.6,2.7},{2.3,2.4,2.5}
	};
	float test_target[TEST_CASE][3] =
	{
		{RIGHT,WRONG,WRONG}, {RIGHT,WRONG,WRONG}, {RIGHT,WRONG,WRONG}, {RIGHT,WRONG,WRONG}, {RIGHT,WRONG,WRONG}, {RIGHT,WRONG,WRONG}, {RIGHT,WRONG,WRONG},
		{WRONG,RIGHT,WRONG}, {WRONG,RIGHT,WRONG}, {WRONG,RIGHT,WRONG}, {WRONG,RIGHT,WRONG}, {WRONG,RIGHT,WRONG}, {WRONG,RIGHT,WRONG},
		{WRONG,WRONG,RIGHT}, {WRONG,WRONG,RIGHT}, {WRONG,WRONG,RIGHT}, {WRONG,WRONG,RIGHT}, {WRONG,WRONG,RIGHT}, {WRONG,WRONG,RIGHT}, {WRONG,WRONG,RIGHT}
	};

	int wcnt = 0;
	// 학습데이터 300개를 TIMES만큼 학습시킴
	printf("n만큼 학습시켰을 때의 오차값, 20개의 TEST_CASE 중 맞은 개수\n\n");
	for (int num = 0;num < TIMES;num++)
	{
		//순전파
		sum_sum_error = 0;
		for (int testnum = 0;testnum < LEARN_DATA;testnum++)
		{
			sum_error = 0;
			i_output = input_arr[testnum];
			forward(INPUT_LAYER, HIDDEN_LAYER_1, i_output, j_input, j_output, weight_ij);
			forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
			forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);

			//에러값 확인
			for (int i = 0;i < OUTPUT_LAYER;i++)
			{
				l_error[i] = -(target_arr[testnum][i] - l_output[i]);
			}

			// 역전파
			backward(OUTPUT_LAYER, HIDDEN_LAYER_2, l_error, k_error, weight_kl, k_output);
			backward(HIDDEN_LAYER_2, HIDDEN_LAYER_1, k_error, j_error, weight_jk, j_output);

			//가중치 업데이트
			for (int i = 0;i < INPUT_LAYER;i++)
			{
				for (int j = 0;j < HIDDEN_LAYER_1;j++)
				{
					weight_ij[i][j] -= LEARN_RATE * (j_error[j]) * j_output[j] * (1 - j_output[j]) * i_output[i];
					fprintf(fp5, "%d, %f\n", ++wcnt, weight_ij[i][j]);
				}
			}
			wcnt = 0;
			for (int i = 0;i < HIDDEN_LAYER_1;i++)
			{
				for (int j = 0;j < HIDDEN_LAYER_2;j++)
				{
					weight_jk[i][j] -= LEARN_RATE * (k_error[j]) * k_output[j] * (1 - k_output[j]) * j_output[i];
					fprintf(fp6, "%d, %f\n", ++wcnt, weight_jk[i][j]);
				}
			}
			wcnt = 0;
			for (int i = 0;i < HIDDEN_LAYER_2;i++)
			{
				for (int j = 0;j < OUTPUT_LAYER;j++)
				{
					weight_kl[i][j] -= LEARN_RATE * (l_error[j]) * l_output[j] * (1 - l_output[j]) * k_output[i];
					fprintf(fp7, "%d, %f\n", ++wcnt, weight_kl[i][j]);
				}
			}
			for (int i = 0;i < 3;i++)
			{
				sum_error += (pow((target_arr[testnum][i] - l_output[i]), 2)) / 2;
			}
			sum_sum_error += sum_error;
		}
		fprintf(fp3, "%d, %f\n", num + 1, sum_sum_error / LEARN_DATA);
		int correct = 0;
		for (int i = 0;i < TEST_CASE;i++)
		{
			//순전파
			forward(INPUT_LAYER, HIDDEN_LAYER_1, test_input[i], j_input, j_output, weight_ij);
			forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
			forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);
			sum_error = 0;

			// 실제 색깔과 비교
			if (find_maxidx(test_target[i][0], test_target[i][1], test_target[i][2]) == find_maxidx(l_output[0], l_output[1], l_output[2]))
			{
				correct++;
			}
		}
		fprintf(fp4, "%d\n", correct);
		printf("%d번째 맞은 갯수: %d / 오차값: %f\n", num+1, correct, sum_sum_error/LEARN_DATA);
	}
	printf("\n");
	// 최종결과에 영향을 크게 미치는 가중치 3개 찾기
	printf("오차를 각 가중치에 대해 미분한 결과(결과에 가장 영향을 미치는 가중치 3개 찾기)\n\n");
	float dweight_ij[INPUT_LAYER][HIDDEN_LAYER_1];
	float dweight_jk[HIDDEN_LAYER_1][HIDDEN_LAYER_2];
	float dweight_kl[HIDDEN_LAYER_2][OUTPUT_LAYER];
	printf("-----Layer1-2의 미분된 가중치 행렬-----\n");
	for (int i = 0;i < INPUT_LAYER;i++)
	{
		for (int j = 0;j < HIDDEN_LAYER_1;j++)
		{
			dweight_ij[i][j] = (j_error[j]) * j_output[j] * (1 - j_output[j]) * i_output[i];
			printf("%f ", dweight_ij[i][j]);
		}
		printf("\n");
	}
	printf("-----Layer2-3의 미분된 가중치 행렬-----\n");
	for (int i = 0;i < HIDDEN_LAYER_1;i++)
	{
		for (int j = 0;j < HIDDEN_LAYER_2;j++)
		{
			dweight_jk[i][j] = (k_error[j]) * k_output[j] * (1 - k_output[j]) * j_output[i];
			printf("%f ", dweight_jk[i][j]);
		}
		printf("\n");
	}
	printf("-----Layer3-4의 미분된 가중치 행렬-----\n");
	for (int i = 0;i < HIDDEN_LAYER_2;i++)
	{
		for (int j = 0;j < OUTPUT_LAYER;j++)
		{
			dweight_kl[i][j] = (l_error[j]) * l_output[j] * (1 - l_output[j]) * k_output[i];
			printf("%f ", dweight_kl[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	// 학습 다 시킨 후, TEST_CASE 넣어서 실험
	printf("TIMES * LEARN_DATA만큼 학습 후, 20개의 TEST_CASE 실험\n\n");
	int correct=0;
	for (int i = 0;i < TEST_CASE;i++)
	{
		forward(INPUT_LAYER, HIDDEN_LAYER_1, test_input[i], j_input, j_output, weight_ij);
		forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
		forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);
		//foward(HIDDEN_LAYER_3, OUTPUT_LAYER, l_output,m_input, m_output, weight_lm);
		printf("Test_Case_%d 좌표: %f, %f, %f\n", i + 1, test_input[i][0], test_input[i][1], test_input[i][2]);
		printf("출력값: %f, %f, %f\n", l_output[0], l_output[1], l_output[2]);
		sum_error = 0;
		for (int j = 0;j < 3;j++)
		{
			sum_error += (pow((test_target[i][j] - l_output[j]), 2)) / 2;
		}
		printf("최종 오차: %f\n", sum_error);
		if (find_maxidx(test_target[i][0], test_target[i][1], test_target[i][2]) == find_maxidx(l_output[0], l_output[1], l_output[2]))
		{
			correct++;
			char color[10];
			switch (find_maxidx(l_output[0], l_output[1], l_output[2]))
			{
			case 0:
				strcpy(color, "Red");
				break;
			case 1:
				strcpy(color, "Green");
				break;
			case 2:
				strcpy(color, "Blue");
				break;
			}
			printf("%f %f %f 공간의 색깔은 %s 고 신경망은 색깔을 맞췄습니다\n", test_input[i][0], test_input[i][1], test_input[i][2], color);
		}
		else
		{
			char color[10];
			char real_color[10];
			switch (find_maxidx(l_output[0], l_output[1], l_output[2]))
			{
			case 0:
				strcpy(color, "Red");
				break;
			case 1:
				strcpy(color, "Green");
				break;
			case 2:
				strcpy(color, "Blue");
				break;
			}
			switch (find_maxidx(test_target[i][0], test_target[i][1], test_target[i][2]))
			{
			case 0:
				strcpy(real_color, "Red");
				break;
			case 1:
				strcpy(real_color, "Green");
				break;
			case 2:
				strcpy(real_color, "Blue");
				break;
			}
			printf("공간의 색깔은 %s 고 신경망은 %s 로 예측해 틀렸습니다\n", real_color, color);
		}
		printf("------------------------------------\n\n");
	}
	printf("최종결과\n맞춘색깔: %d\n", correct);
	printf("최종오차: %f", sum_sum_error / LEARN_DATA);
}
