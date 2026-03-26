# Assignment 1 Rules

기준 문서: `assignment1.pdf`  
원문 대조용: `rule_audit.md`

## 핵심

- 대상 데이터셋: `MNIST`, `CIFAR-10`
- 구현 공격: targeted FGSM, untargeted FGSM, targeted PGD, untargeted PGD
- 실행 파일: `test.py`
- 최종 제출물: `requirements.txt`, `test.py`, `results/`, `report.pdf`, `README.md`

## 일정 / 제출

- 배포일: `Wednesday, March 25, 2026`
- 마감일: `Wednesday, April 1, 2026, 11:59 PM`
- 제출 방식: GitHub 업로드 후 Uclass 링크 제출
- 지각 제출: 시스템 / 이메일 모두 불가

## 프로젝트 구성

- `requirements.txt` 포함
- 외부 모듈 / Python dependency 전부 기록
- 코드 이해용 주석 포함
- `test.py` 포함
- `test.py` 실행 시 순차 학습 + adversarial attack 수행
- 테스트 형식 자유, 기능과 타당성 확인 가능 상태

## 데이터셋 / 모델

### MNIST

- 분류기 직접 구현
- 아키텍처 자유
- simple CNN 사용 가능

### CIFAR-10

- 직접 학습 모델 사용 가능
- pretrained model 사용 가능
- pretrained model / open-source code 사용 시 citation 필요

### 공격 전 확인

- clean test accuracy 확인
- 권장 목표:
  - `MNIST >= 95%`
  - `CIFAR-10 >= 80%`

## 구현 함수

안전한 구현 기준상 아래 함수명 사용 권장

```python
fgsm_targeted(model, x, target, eps)
fgsm_untargeted(model, x, label, eps)
pgd_targeted(model, x, target, k, eps, eps_step)
pgd_untargeted(model, x, label, k, eps, eps_step)
```

- return: adversarial image `x_adv`

## 공격 구현 규칙

### Targeted FGSM

- target label 기준 loss
- update 식: `x_adv = x - eps * sign(grad)`
- 결과 범위: `[0, 1]` clamp

### Untargeted FGSM

- correct label 기준 loss
- update 식: `x_adv = x + eps * sign(grad)`
- 결과 범위: `[0, 1]` clamp

### PGD

- targeted / untargeted 모두 구현
- iterative FGSM 방식
- step 수: `k`
- step 크기: `eps_step`
- 전체 perturbation budget: `eps`
- 매 step 후 `eps-ball` projection
- 매 step 후 `[0, 1]` clamp
- 방향:
  - targeted: minus
  - untargeted: plus
- 일반적 초기값: clean input
- 권장 구현: FGSM gradient 로직 재사용
- MNIST 예시 하이퍼파라미터: `eps=0.3`, `eps_step=0.01`, `k=40`

## PyTorch 팁

- 필요 시 forward 전 `x.requires_grad_(True)`
- `loss.backward()` 후 `x.grad` 사용

## test.py 출력 요구사항

### 실행 범위

- 네 가지 공격 전부 실행
- 두 데이터셋 모두 실행:
  - `MNIST`
  - `CIFAR-10`

### 1. Attack Success Rate

- 각 공격 / 각 데이터셋별 success rate 출력
- 평가 샘플 수: 최소 `100개`
- 성공 기준:
  - targeted attack: target class 예측
  - untargeted attack: 정답이 아닌 다른 class 예측

### 2. Visualization

- 최소 `5개` sample 시각화
- side-by-side 구성
- 포함 요소:
  - original image + predicted label
  - adversarial image + incorrect prediction
  - magnified perturbation
- 저장 위치: `results/`
- 저장 형식 예시: PNG

## 보고서

- 파일명: `report.pdf`
- 분량: `1-2 pages`
- 포함 내용:
  - epsilon별 attack success rate 표
  - 공격별 효과 차이 논의
  - epsilon과 perturbation visibility trade-off 논의
- 예시 epsilon:
  - `0.05`
  - `0.1`
  - `0.2`
  - `0.3`

## AI 사용 / Git 히스토리

- AI 도구 사용 가능
- 전제: 실제 이해 기반 작업
- 저장소 요구:
  - incremental commit history
  - single bulk commit 지양
- 보고서 요구:
  - 본인 해석과 reasoning
  - 실험 중 관찰한 구체적 behavior
  - unexpected result 포함

## 최종 제출물

- `requirements.txt`
- `test.py`
- `results/`
- `report.pdf`
- `README.md`

## README 요구사항

- 실행 방법 설명 포함
- 간단한 사용 안내 포함

## 해석 주의

- 함수명: 원문상 `signature similar to`, 실무상 예시 함수명 그대로 사용 권장
- 정확도 기준: 절대 컷이 아닌 rough guideline
- simple CNN: 예시 아키텍처
- PGD clean input 시작: common choice, 절대 강제 조항 아님
