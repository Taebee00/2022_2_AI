#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// ����� ���� 
#define INPUT_LAYER 3
#define HIDDEN_LAYER_1 5
#define HIDDEN_LAYER_2 3
#define OUTPUT_LAYER 3
#define LEARN_RATE 0.5
#define LEARN_DATA 300
#define RIGHT 0.8
#define WRONG 0.2
#define TIMES 20
#define TEST_CASE 20

// �� ����� �Է�, ��� �迭, ��ǥ �迭 ����
float input_arr[LEARN_DATA][3];
float target_arr[LEARN_DATA][3];
float* i_output;
float j_input[HIDDEN_LAYER_1], j_output[HIDDEN_LAYER_1], j_error[HIDDEN_LAYER_1];
float k_input[HIDDEN_LAYER_2], k_output[HIDDEN_LAYER_2], k_error[HIDDEN_LAYER_2];
float l_input[OUTPUT_LAYER], l_output[OUTPUT_LAYER], l_error[OUTPUT_LAYER];
float weight_ij[INPUT_LAYER][HIDDEN_LAYER_1], weight_jk[HIDDEN_LAYER_1][HIDDEN_LAYER_2], weight_kl[HIDDEN_LAYER_2][OUTPUT_LAYER], weight_lm[HIDDEN_LAYER_2][OUTPUT_LAYER];
float sum_error = 0;
float sum_sum_error = 0;

// ���� ���� Ȯ���� ������ ��ȯ���ִ� �Լ�
int find_maxidx(float a, float b, float c)
{
	float max = a;
	int maxidx = 0;
	max = b > a ? b : max;
	maxidx = b > a ? 1 : 0;
	maxidx = c > max ? 2 : maxidx;
	return maxidx;
}

// ����ġ ��� ���� �ʱ�ȭ �Լ�
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

// ������ �Լ�
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

// ������ �Լ�
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

// �����Լ�
int main()
{
	srand(time(NULL));

	// ���̽� ���� �ð�ȭ�� ���� csv ���� �����
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
	FILE* fp8 = fopen("error_graph.csv", "w");
	fprintf(fp8, "idx,error\n");

	// ����ġ ��� ���� �ʱ�ȭ
	rand_weight(INPUT_LAYER, HIDDEN_LAYER_1, weight_ij);
	rand_weight(HIDDEN_LAYER_1, HIDDEN_LAYER_2, weight_jk);
	rand_weight(HIDDEN_LAYER_2, OUTPUT_LAYER, weight_kl);

	// 300���� �н� �����Ϳ� ��ǥ�� ����
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
		// ��ǥ���� ��°��� ���̽� �ð�ȭ�� ���� csv���Ϸ� ����
		fprintf(fp, "%f,%f,%f\n", input_arr[i][0], input_arr[i][1], input_arr[i][2]);
		fprintf(fp2, "%f,%f,%f\n", target_arr[i][0], target_arr[i][1], target_arr[i][2]);
	}

	// TEST_CASE 20�� ���� 
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

	int cnt = 0;
	int wcnt = 0;
	// �н������� 300���� TIMES��ŭ �н���Ŵ
	printf("n��ŭ �н������� ���� ������, 20���� TEST_CASE �� ���� ����\n\n");
	for (int num = 0;num < TIMES;num++)
	{
		//������
		sum_sum_error = 0;
		for (int testnum = 0;testnum < LEARN_DATA;testnum++)
		{
			sum_error = 0;
			i_output = input_arr[testnum];
			forward(INPUT_LAYER, HIDDEN_LAYER_1, i_output, j_input, j_output, weight_ij);
			forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
			forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);

			//������ Ȯ��
			for (int i = 0;i < OUTPUT_LAYER;i++)
			{
				l_error[i] = -(target_arr[testnum][i] - l_output[i]);
			}

			// ������
			backward(OUTPUT_LAYER, HIDDEN_LAYER_2, l_error, k_error, weight_kl, k_output);
			backward(HIDDEN_LAYER_2, HIDDEN_LAYER_1, k_error, j_error, weight_jk, j_output);

			//����ġ ������Ʈ
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
			fprintf(fp8, "%d, %f\n", cnt, sum_error);
			cnt++;
		}
		fprintf(fp3, "%d, %f\n", num + 1, sum_sum_error / LEARN_DATA);
		int correct = 0;
		for (int i = 0;i < TEST_CASE;i++)
		{
			//������
			forward(INPUT_LAYER, HIDDEN_LAYER_1, test_input[i], j_input, j_output, weight_ij);
			forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
			forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);
			sum_error = 0;

			// ���� ����� ��
			if (find_maxidx(test_target[i][0], test_target[i][1], test_target[i][2]) == find_maxidx(l_output[0], l_output[1], l_output[2]))
			{
				correct++;
			}
		}
		fprintf(fp4, "%d\n", correct);
		printf("%d��° ���� ����: %d / ������: %f\n", num+1, correct, sum_sum_error/LEARN_DATA);
	}
	printf("\n");
	// ��������� ������ ũ�� ��ġ�� ����ġ 3�� ã��
	printf("������ �� ����ġ�� ���� �̺��� ���(����� ���� ������ ��ġ�� ����ġ 3�� ã��)\n\n");
	float dweight_ij[INPUT_LAYER][HIDDEN_LAYER_1];
	float dweight_jk[HIDDEN_LAYER_1][HIDDEN_LAYER_2];
	float dweight_kl[HIDDEN_LAYER_2][OUTPUT_LAYER];
	printf("-----Layer1-2�� �̺е� ����ġ ���-----\n");
	for (int i = 0;i < INPUT_LAYER;i++)
	{
		for (int j = 0;j < HIDDEN_LAYER_1;j++)
		{
			dweight_ij[i][j] = (j_error[j]) * j_output[j] * (1 - j_output[j]) * i_output[i];
			printf("%f ", dweight_ij[i][j]);
		}
		printf("\n");
	}
	printf("-----Layer2-3�� �̺е� ����ġ ���-----\n");
	for (int i = 0;i < HIDDEN_LAYER_1;i++)
	{
		for (int j = 0;j < HIDDEN_LAYER_2;j++)
		{
			dweight_jk[i][j] = (k_error[j]) * k_output[j] * (1 - k_output[j]) * j_output[i];
			printf("%f ", dweight_jk[i][j]);
		}
		printf("\n");
	}
	printf("-----Layer3-4�� �̺е� ����ġ ���-----\n");
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

	// �н� �� ��Ų ��, TEST_CASE �־ ����
	printf("TIMES * LEARN_DATA��ŭ �н� ��, 20���� TEST_CASE ����\n\n");
	int correct=0;
	for (int i = 0;i < TEST_CASE;i++)
	{
		forward(INPUT_LAYER, HIDDEN_LAYER_1, test_input[i], j_input, j_output, weight_ij);
		forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
		forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);
		//foward(HIDDEN_LAYER_3, OUTPUT_LAYER, l_output,m_input, m_output, weight_lm);
		printf("Test_Case_%d ��ǥ: %f, %f, %f\n", i + 1, test_input[i][0], test_input[i][1], test_input[i][2]);
		printf("��°�: %f, %f, %f\n", l_output[0], l_output[1], l_output[2]);
		sum_error = 0;
		for (int j = 0;j < 3;j++)
		{
			sum_error += (pow((test_target[i][j] - l_output[j]), 2)) / 2;
		}
		printf("���� ����: %f\n", sum_error);
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
			printf("%f %f %f ������ ������ %s �� �Ű���� ������ ������ϴ�\n", test_input[i][0], test_input[i][1], test_input[i][2], color);
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
			printf("������ ������ %s �� �Ű���� %s �� ������ Ʋ�Ƚ��ϴ�\n", real_color, color);
		}
		printf("------------------------------------\n\n");
	}
	printf("�������\n�������: %d\n", correct);
	printf("��������: %f", sum_sum_error / LEARN_DATA);
}