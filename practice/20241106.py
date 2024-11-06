"""
pip install torch, matplotlib
"""

def TODO():
    """
    이 함수가 쓰인 곳을 지우고 올바른 코드를 적어주세요
    """
    return

import torch
import matplotlib.pyplot as plt

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 입력층: 입력 크기 1, 출력 크기 1
        self.input_layer = TODO()
        # 출력층: 입력 크기 1, 출력 크기 1
        self.output_layer = TODO()

    def forward(self, x):
        x = self.input_layer(x)  # 입력층을 통과
        x = self.output_layer(x)  # 출력층을 통과
        return x

def f(x: int):
    # y = 2x + 1 로 설정
    return 2*x + 1

def generate_test_data(count: int, dataRange: tuple, noiseRange: tuple):
    import random

    x_values = []
    y_values = []
    
    # count 개의 랜덤한 데이터 생성
    for _ in range(count):
        x = random.uniform(*dataRange)
        # 불규칙적인 테스트 데이터 생성을 위해 노이즈 추가
        noise = random.uniform(*noiseRange)
        
        x_values.append(x)
        y_values.append(f(x) + noise)
    
    return x_values, y_values

def log_data(epoch: int, total_epochs: int, loss: float, log_step: int):
    if epoch % log_step == 0:
        print(f'Epoch [{epoch}/{total_epochs}], Loss: {loss:.4f}')

def show_data(X: list, y: list, predict: list, delay: float = 0.1):
    # 활성화된 요소 삭제
    plt.cla()
    plt.clf()
    # 실제값을 산점도로 표시
    plt.scatter(X, y)
    # 예측값을 그래프로 표시
    plt.plot(X, predict, color='red')
    # 업데이트 딜레이
    plt.pause(delay)

def train_data(X: list, y: list, lr: float = 0.001, epochs: int = 100, log_step: int = 10):
    # 모델은 2차원 tensor 데이터를 받을 수 있음
    # list를 tensor로 변환 후 1차원에 차원 추가 (unsqueeze)
    X = TODO()
    y = TODO()
    # **HINT**: tensor()의 파라미터로 list를 넘기면 변환됨

    # 모델 설정
    model = LinearRegressionModel()
    # 평균 제곱 오차 (Mean Squared Error, MSE) 적용
    # 예측값과 실제값 차이의 제곱 평균
    criterion = TODO()
    # 확률적 경사 하강법 (Stochastic Gradient Descent, SGD) 적용
    # 새 가중치 = 기존 가중치 - 학습률 * 기울기
    # 학습률은 다음 가중치를 결정하는 중요한 하이퍼파라미터로, 기울기를 얼마나 반영할지를 정함
    # 학습률이 지나치게 높은 경우에 가중치가 수치적 한계를 넘어감 (nan return)
    # 모델 파라미터는 가중치, 편향 등 미분이 가능한 텐서들로 구성됨
    optimizer = TODO()
    # **HINT**: 모델 파라미터와 학습률 넘기기

    # 모델을 훈련 모드로 설정
    # 모드에 따라 레이어의 동작 방식이 달라지기에 구분함
    model.train()

    for epoch in range(epochs):
        # 이전 단계에서 계산된 기울기 초기화
        optimizer.zero_grad()

        # **HINT**: 이 부분은 ANN에서 가장 중요한 부분이기에 직접 서칭해서 작성해보세요

        # 순전파 (예측값 계산)
        outputs = TODO()
        # 손실함수 (예측값과 실제값의 차이 계산)
        loss = TODO()
        # 역전파 (기울기 계산, pytorch가 알아서 해줌)
        TODO()
        # 최적화 (계산된 기울기로 파라미터 업데이트)
        TODO()

        # 로그 출력
        log_data(epoch+1, epochs, loss.item(), log_step)

        # 데이터 시각화
        show_data(X, y, outputs.tolist())
    
    # 모델을 평가 모드로 설정
    model.eval()

    # 모델이 설정한 파라미터 확인
    # 일반적인 상황에서는 로깅해도 별 의미가 없고 사람이 해석하기에 어려움이 있기에 하지 않음
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values: {param[:5]}")

    # 최종적인 예측값을 list로 반환
    return model(X).tolist()

def run():
    # 1~10 사이의 랜덤한 값을 100개 생성. 노이즈는 -1.5~1.5 범위의 랜덤한 값
    X, y = generate_test_data(100, (1, 10), (-1.5, 1.5))

    plt.ion()
    predict = train_data(X, y)
    plt.ioff()
    plt.show()

run()
