# Area-Classification
3차원 공간에서 영역을 3가지로 나눈다. 한 영역에서는 한 색만 들어갈 수 있으며 색깔은 R, G, B 3가지로 한다. 영역과 색의 조합과 구성은 자유롭게 정할 수 있다. 
그렇게 임의로 정한 영역과 색에서 좌표와 색 데이터를 통해 학습하고, 좌표 입력을 받아 그 좌표(영역)애서의 색을 판단하는 신경망을 구성한다.

## 프로젝트 기간
2022.10.18 ~ 2021.11.08

## 조건
- 신경망 구성
  - 입력층 ~ 은닉층 ~ 출력층까지 노드의 갯수: 3 -> 5 > 3 -> 3
  - 입력층 노드: x,y,z 좌표값
  - 출력층 노드: r,g,b 색
- 학습 데이터: 200개 이상
- 테스트 데이터: 20개 

## 개발 환경
- C
- Python
- IDE: Visual Studio

## 코드
### 정의 및 선언
#### 사용한 헤더 파일, 매크로 상수 정의
```C
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
#define RIGHT 0.8
#define WRONG 0.2
#define TIMES 20
#define TEST_CASE 20
```
- 필요한 헤더 파일 Include
- 신경망의 변인이 달라졌을 때 일반화하여 대처할 수 있도록 매크로 상수 정의
    - 입력계층, 은닉계층, 출력계층 노드 수
    - 학습률
    - 학습 데이터 갯수
    - 출력 노드의 목표값
    - 에포크 횟수
    - 테스트 케이스 횟수
- 이렇게 매크로 상수를 통해 쉽게 변인을 바꿔가면서 최적의 결과를 찾기 위해 실험할 수 있었음

#### 변수 선언

```c
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
```

- 전에 선언한 매크로 상수를 이용하여 입력계층, 은닉계층, 출력계층의 출력 배열과 계층 사이의 가중치 배열을 선언
- 한번의 미니배치에서의 오차와 한번의 에포크 오차를 측정하기 위한 `sum_error` / `sum_sum_error` 선언

### 함수
#### find_maxidx()

```c
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
```

- 가장 출력값이 높은 노드의 인덱스를 반환해주는 함수
- 학습의 결과와 실제 목표값을 비교할 때 사용

#### rand_weight()

```c
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
```

- 계층 두개의 노드 갯수를 받아서 그 크기에 맞는 가중치 행렬을 만들어 주는 함수
- 가중치를 다양하게 주고 싶어서 사용

#### forward()

```c
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
```

- 계층 두개의 노드 갯수, 가중치 행렬, 입출력값 등을 받고 순전파를 수행하여 다음 계층의 노드에 출력을 만들어주는 함수
- 일반적인 행렬곱이 아닌 행렬곱을 전치시켜서 사용

#### backward()

```c
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
```

- 오차를 가중치에 대해 미분한 값:
    - $\delta w_{jk}=E_k*O_k(1-O_k)*O_j$
- 오차값을 모든 각각의 노드에 역전파시켜서 각 노드의 오차값을 구한 후에 이 공식을 따라서 전체 오차에 가중치에 대한 미분값을 구하는 방식을 사용
- 일반적으로 오차를 역전파할 때 사용했던 가중치 행렬과 오차값 행렬만을 곱하는 것이 아닌, 그 값에 가중치가 들어오는 노드의 출력값, 1에서 가중치가 들어오는 노드의 출력값을 뺀 값을 곱하면 체인룰 미분에서의 노드의 에러값이 나옴
- backward 함수를 통해 은닉층에 대한 오차값을 구하고 그렇게 은닉층의 오차값을 모두 구한 후, 그 오차값에 대해 $E_k*O_k(1-O_k)*O_j$ 공식을 적용하여 가중치를 업데이트

### 메인 함수
#### csv 파일 생성 및 칼럼 정의
```c
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
	FILE* fp8 = fopen("error_graph.csv", "w");
	fprintf(fp8, "idx,error\n");
```

- 파이썬에서 시각화에 쓰일 csv파일 생성
- 입력값, 목표값, 에러값, Test_Case 정확성, 가중치 행렬 등에 대한 csv 파일 생성

#### 가중치 행렬 초기화

```c
// 가중치 행렬 임의 초기화
rand_weight(INPUT_LAYER, HIDDEN_LAYER_1, weight_ij);
rand_weight(HIDDEN_LAYER_1, HIDDEN_LAYER_2, weight_jk);
rand_weight(HIDDEN_LAYER_2, OUTPUT_LAYER, weight_kl);
```

- 계층 간 가중치 행렬을 전에 만들어둔 `rand_weight` 함수를 통해 초기화

#### 학습 데이터 생성

```c
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
}
```

![데이터 영역과 그에 따른 색깔](https://github.com/Taebee00/2022_2_AI/assets/104549849/3729acfd-7838-4c35-bb12-e4bc8ff2b64a)

- 학습 데이터가 코드에서 정한 일정 범위 안에서 랜덤한 값을 가지게 하여 한 영역에 100개씩 300개의 학습 데이터를 생성
- 한 영역에 대해 편향적으로 학습이 되지 않도록 학습 데이터 순서를 영역별로 돌면서 골고루 하도록 함
- 그에 따른 학습 목표값 배열도 생성

#### TEST_CASE 생성

```c
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
```

- TEST_CASE 20개 생성
- 입력값 배열 20개와 그에 따른 목표값 배열 20개 생성
- 입력값은 빨간영역에 7, 초록영역에 6, 파란영역에 7개 배정
- 출력값은 매크로 상수로 정한 RIGTH,WRONG값 기준으로 정해줌

#### 순전파

```c
i_output = input_arr[testnum];
forward(INPUT_LAYER, HIDDEN_LAYER_1, i_output, j_input, j_output, weight_ij);
forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);
```

- input_arr의 학습 데이터를 받아서 순전파 진행

#### 에러값 확인

```c
for (int i = 0;i < OUTPUT_LAYER;i++)
{
	l_error[i] = -(target_arr[testnum][i] - l_output[i]);
}
```

- 해당 학습 데이터의 목표값과 최종 결과를 빼서 에러값을 구함
- 이후에 오차값 미분 과정에서 부호가 바뀌는 것을 한번 미리 처리해주기 위해 부호를 바꿔줌

#### 역전파와 가중치 업데이트

```c
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
```

- 전에 선언한 `backward` 역전파 함수를 통해 각 노드의 오차값이 역전파되어있는 상태
- $\delta w_{jk}=E_k*O_k(1-O_k)*O_j$고
    - 그렇게 역전파된 에러값은 $E_K$라고 한다면 위의 공식을 통해 손쉽게 전체 에러값을 해당 가중치로 미분한 값을 얻을 수 있음
- 기존 가중치에서 미분값*학습률 값의 차를 구해서 가중치 행렬을 업데이트해줌

#### 평균제곱오차

```c
for (int i = 0;i < 3;i++)
{
	sum_error += (pow((target_arr[testnum][i] - l_output[i]), 2)) / 2;
}
sum_sum_error += sum_error;
```

- 3개의 출력 노드를 거치면 평균 제곱오차를 구함
- 한 미니배치를 돌때마다 평균제곱오차를 sum_sum_error에 더해주고 한번의 에포크가 다 돌면 한번의 에포크에 대한 오차인 sum_sum_error를 학습데이터로 나누어 한 에포크의 평균적인 오차를 측정
    
    → 한 에포크씩 학습시킬 때마다 오차가 줄어드는지 측정할 수 있는 일관적인 정보
    
#### TEST_CASE 결과 중간 확인

```c
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
```

- 한번의 에포크가 돌때마다 테스트 케이스 20개 중 몇개를 맞췄는지, 한번의 에포크에 평균 오차값은 무엇인지 출력해줌
- 전에 만들었던 `find_maxidx` 함수를 이용해 정답 판별

#### TEST_CASE 결과 최종 확인

```c
/ 학습 다 시킨 후, TEST_CASE 넣어서 실험
	printf("TIMES * LEARN_DATA만큼 학습 후, 20개의 TEST_CASE 실험\n\n");
	int correct=0;
	for (int i = 0;i < TEST_CASE;i++)
	{
		forward(INPUT_LAYER, HIDDEN_LAYER_1, test_input[i], j_input, j_output, weight_ij);
		forward(HIDDEN_LAYER_1, HIDDEN_LAYER_2, j_output, k_input, k_output, weight_jk);
		forward(HIDDEN_LAYER_2, OUTPUT_LAYER, k_output, l_input, l_output, weight_kl);
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
```

- 모든 에포크에 대해 학습을 완료한 후, 각각의 테스트 케이스에 대해 하나씩 출력값, 오차값, 예측한 색깔, 실제 색깔 출력
- 최종결과로 맞춘색깔과 최종오차값 출력

## Flow Chart
![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/c137be4f-029d-4cf1-96d5-3cb5bcc7e87c)

## 실행 결과

### TEST_CASE 중간 오차값, 맞은 갯수 확인

#### 5번의 에포크 동안의 중간 결과 출력

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/fce6da9d-770f-4db8-888e-d428cfa7891b)

- 5번의 에포크를 학습시키는 동안 한번의 에포크에 대해 20개의 TEST CASE 중 신경망이 맞힌 공간의 갯수와 평균오차값 출력
- 에포크가 증가하는 동안 맞추는 TEST CASE의 갯수는 늘어나고 오차값은 0에 수렴하게 되는 것을 알 수 있음

#### 20번의 에포크 동안의 중간 결과 출력

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/a96e247f-390e-4aa2-844e-44189d087f23)

- 5번의 에포크가 아닌 20번까지 에포크를 돌려봤을 때, 6번째 에포크에서부터는 20개의 TEST_CASE 모두를 맞춘 것을 알 수 있고, 에러값은 더욱 0에 수렴하는 것을 알 수 있음

### TEST_CASE 최종 오차값, 맞은 갯수 확인

- 20번의 에포크를 모두 학습시킨 후 각각의 TEST_CASE에 대해 오차값, 출력값 확인
- 출력 노드 1번: 빨간색 확률 / 출력 노드 2번: 초록색 확률 / 출력 노드 3번: 파란색 확률
- 맞는 색깔 목표값:  0.8 / 틀린 색깔 목표값: 0.2

#### 빨간색 영역에 대한 TEST

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/ad649252-3839-4259-be9a-02bf4d1925a8)

- TEST CASE 1~7은 빨간색 영역의 좌표에 대한 TEST
- 출력 노드 1번은 0.8의 값에 수렴, 출력 노드 2,3번은 0.2의 값에 수렴한 것을 알 수 있음

#### 초록색 영역에 대한 TEST

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/69132454-607e-4679-a554-6f5f1b62637f)

- TEST CASE 8~13은 초록색 영역의 좌표에 대한 TEST
- 출력 노드 2번은 0.8의 값에 수렴, 출력 노드 1,3번은 0.2의 값에 수렴한 것을 알 수 있음

#### 파란색 영역에 대한 TEST

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/2328eb9a-9b51-439b-b29f-aaa77089177d)

- TEST CASE 14~20은 파란색 영역의 좌표에 대한 TEST
- 출력 노드 3번은 0.8의 값에 수렴, 출력 노드 1,2번은 0.2의 값에 수렴한 것을 알 수 있음


### 최종 결과

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/2b63208a-b91e-47ab-902a-81a4fec8ab08)

최종적으로 20번의 에포크를 학습시켰을 때 20개의 TEST_CASE를 모두 신경망이 맞췄으며 마지막 에포크 동안의 평균 오차값은 0.000762로 거의 0에 수렴하는 결과를 얻었음


## 시각화

### 한번씩의 미니배치에 대한 에러 그래프

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/27e45c44-5c90-460c-bef1-7288a9e6e1c1)

- 평균제곱오차를 사용
- 3000번째의 미니배치까지는 수렴하면서 중간에 튀는 값들이 있지만 그 이후에는 크게 튀는 값 없이 각각의 미니배치에서 거의 0의 값에 수렴하는 것을 알 수 있음

### 한번의 에포크에 대한 에러, 테스트 케이스 그래프

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/65e11452-6481-40f2-a223-052009b90d14)

- 빨간 그래프: n번째의 에포크 동안의 평균제곱오차값 그래프
- 파란 그래프: n번의 에포크 학습 후 신경망이 맞춘 TEST_CASE 갯수
- 오차값은 10번째 에포크에서, TEST_CASE는 7번째 에포크쯤에서부터 정확성을 가지는 것을 알 수 있음

### 가중치 그래프 변화율

![image](https://github.com/Taebee00/2022_2_AI/assets/104549849/1e266f1a-e470-4048-abf3-a44ab45e431d)

- 20번의 에포크 6000개의 학습 동안 가중치 변화량
- 왼쪽부터 1 ~ 2번 계층, 2 ~ 3번 계층, 3 ~ 4번 계층의 가중치
- 가중치 역시도 오차값이 수렴하는 것처럼 대략 3000번째 미니배치에서부터 각자의 일정한 값으로 수렴하기 시작하는 것을 알 수 있음
- 출력에 가까운 계층일수록 가중치의 변화율이 크다는 것을 알 수 있음

## 결론

- 순전파, 역전파, 경사하강법 등을 통해 3개의 입력에 대해 3개의 분류 학습이 실제로 거의 정확하게 가능하다는 것을 알게 되었음
- 노드의 수, 초기 데이터값, 학습률에 따라 학습의 과정이 매우 달라짐
- 가중치의 변화율은 출력에 가까운 노드일수록 크며, 오차값이 수렴하는 속도와 매우 일치하고, 모든 가중치의 변화율이 수렴하는 타이밍도 거의 같음


