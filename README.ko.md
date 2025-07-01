# auto-evol-instruct
Auto Instruction Evolution는 LLM 파인튜닝에 사용되는 instruction의 complexity를 인간의 개입 없이 자동으로 향상시키는 프레임워크입니다.

[English](./README.md) | [한국어](./README.ko.md)

## 주요 키워드
- Method: Instruction의 복잡성을 높이기 위한 프롬프트
- Trajectory: Metohd에 따라 진화된 instruction의 모든 단계

## 주요 특징
- 인간의 개입 없이 Instruction의 복잡성 자동 향상
- 다양한 유형의 데이터 지원
- Method 최적화 프로세스에 대한 유연한 설정

## 소개
이 저장소는 Automatic Instruction Evolving 논문(https://arxiv.org/abs/2406.00770)의 자동화된 알고리즘을 기반으로, 한국어 데이터 증강을 위하여 직접 구현한 프레임워크를 포함하고 있습니다.

이전 버전(https://github.com/joonavel/evol-instruct)에서는 instruction의 진화 방법을 직접 작성해야했습니다. 이 때문에 기존에 작성한 진화 방법과 호환되지 않는 데이터 유형이 입력되었을 경우 진화가 잘 이루어지지 않았습니다.

본 프레임워크는 Fully Automated Method Optimization과정을 추가하여 진화를 진행하기 전 어떠한 Method가 입력받은 instruction들을 진화시키는데 있어 최적인지를 LLM이 결정하도록 만들어 이러한 문제를 해결하였습니다.

## 설치 방법
```
pip install -r requirements.txt
```

## 사용법

### 환경 변수 설정
프로젝트를 실행하기 전에 .env 파일에 다음 환경 변수를 설정해야 합니다:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
MY_HF_TOKEN=your_huggingface_api_key
```

### 필수 파라미터
- ```data_path <hf_dataset_name>```: Huggingface 데이터셋 경로. 데이터셋에는 "instruction"과 "response" 필드가 있어야 합니다.
- ```train_size <int>```: Method Optimization Process에 사용할 데이터 크기.
- ```dev_size <int>```: Method Validation에 사용할 데이터 크기.
- ```seed <int>```: 재현성을 위한 랜덤 시드.
- ```batch_size <int>```: 전체 프로세스의 배치 크기.
- ```max_steps <int>```: Method Optimization Process의 최대 스텝 수.
- ```loop <int>```: 전체 진화 세대 수(논문의 l).
- ```candidate_size <int>```: 최적화된 Method의 수(논문의 m).
- ```test_run <int>```: 테스트 실행 여부. 1이면 10개의 Instruction만 진화합니다.
- ```save_path <str>```: 진화 결과를 저장할 경로.

### 선택적 파라미터
- ```evol_llm_config <str>```: 진화에 사용할 LLM 설정.
- ```optim_llm_config <str>```: 최적화에 사용할 LLM 설정.

### 모델
현재 Deepseek R1만이 API로 제공되는 LLM 중 MIT 라이선스(상업적 사용 가능) 하에 있기에 이 프레임워크는 진화 및 최적화에 DeepSeek R1 모델을 사용했습니다.
(https://deepseeklicense.github.io/)
다른 모델을 사용하고 싶다면 코드에서 모델을 변경하시면 됩니다(utils/utils.py 참고).

### 사용 예시
```bash
sh main.sh
```
스크립트 내부는 다음과 같습니다:
```bash
python main.py\
    --data-path joonavel/seed_for_evolving\
    --train-size 20\
    --dev-size 50\
    --seed 42\
    --batch-size 10\
    --max-step 3\
    --loop 3\
    --candidate-size 5\
    --test-run 0\
    --save-path evolution_result.json
```
이 명령어는 다음을 수행합니다:
1. "joonavel/seed_for_evolving" 데이터셋을 Huggingface에서 불러옵니다.
2. Method Optimization Process에 20개의 샘플을 사용합니다.
3. Method Validation에 50개의 샘플을 사용합니다.
4. Method Optimization Process를 3회 반복합니다(Method가 3번 진화).
5. 각 Instruction 진화 궤적의 크기는 3(loop)입니다.
6. Optimized Method는 5개의 Method(candidate_size) 중에서 선택됩니다.
7. 최종 진화 결과는 "evolution_result.json" 파일에 저장됩니다.

## 구성 요소
이 프레임워크는 5개의 주요 컴포넌트로 구성됩니다.
- Evolver: Method에 따라 Instruction을 진화시킴
- Analyzer: Instruction 진화 궤적을 분석하고 피드백 제공
- Optimizer: 피드백을 바탕으로 Method를 최적화
- Validator: dev 데이터셋으로 Method를 검증
- Generator: 최적화된 Method로 Instruction을 생성

## 출력
스크립트는 결과를 지정한 출력 파일에 JSON 형식으로 저장합니다.

아래 링크에서 본 프레임워크로 생성된 2K 크기의 데이터셋을 확인할 수 있습니다:
(https://huggingface.co/datasets/joonavel/ko-auto-evol-instruct)

## Appendix

### Is This Framework useful?
이 레포의 방식으로 생성된 데이터가 실제로 유용한지 확인하기 위해서 Unsloth/Phi-4-bnb-4bit 모델에 다음 세가지 데이터셋을 활용한 fine-tuning을 진행하여 성능을 비교했습니다.

1. LIMA 데이터셋을 한국어로 번역한 데이터셋(https://huggingface.co/datasets/taeshahn/ko-lima)
2. Evol-Instruct 에서 생성한 데이터셋(https://huggingface.co/datasets/joonavel/ko-evol-instruct)
3. 이 레포(Auto-Evol-Instruct)에서 생성한 데이터셋(https://huggingface.co/datasets/joonavel/ko-auto-evol-instruct)

KMMLU 데이터셋을 활용하여 모델의 성능을 평가한 결과 이 레포에서 생성한 데이터셋을 활용한 모델은 LIMA 한국어 번역 데이터셋을 활용한 모델보다 더 높은 성능을 보였고 Evol-Instruct 데이터셋을 활용한 모델과 비교해 조금 낮은 성능을 보였습니다.

base model: 0.468

with kolima(epoch8): 0.494 https://huggingface.co/joonavel/Phi-4-kolima-adapter

with koevol(epoch8): 0.499 https://huggingface.co/joonavel/Phi-4-koevol-adapter

with koautoevol(epoch8): 0.497 https://huggingface.co/joonavel/Phi-4-koautoevol-adapter

### Limitations
Auto-Evol-Instruct 방식은 Evol-Instruct 방식과 비교해 다음 두가지 장점이 있습니다.
1. 데이터의 유형에 따라 Evolving Prompt(Method)를 사람의 손으로 직접 만들어내지 않아도 됩니다.
2. Seed Data 보존율이 높습니다.
- koevol : Evolving 과정 평균 손실율 : 6.48%, Responsing 과정 평균 손실율 : 55%(1722 -> 922) (generation이 높아질 수록 크게 증가)
- koautoevol : Evolving 과정 평균 손실율 : 0~1%, Responsing 과정 평균 손실율 : 3.5%(2117 -> 2042)

이런 장점에도 불구하고 실제로 데이터셋을 사용하여 평가한 결과 Evol-Instruct 방식보다 KMMLU 평가 지표가 낮았던 이유를 추측해보면 다음과 같습니다.
- Evol-Instruct 방식은 instruction의 complexity 뿐만 아니라 diversity를 증가시켜주는 프롬프트가 같이 사용되었다.
- 반면 Auto-Evol-Instruct 방식은 diversity를 증가시켜주는 과정이 명시되지 않아 최종 결과 데이터 셋의 diversity가 부족했다.
-> 학습 데이터셋의 diversity 부족으로인한 성능 차이

이 외에도 성능에 부정적인 영향를 끼칠 수 있는 요인들로 가능한 것들은 다음과 같습니다.
1. Method Optimization 결과의 부적합성
2. Evolving 과정에서 instruction의 품질 저하
3. LLM이 instruction에 대한 적절한 답변을 생성하지 못함

이 요소들이 정말로 파인튜닝에 부정적인 영향을 끼쳤는지 확인하기 위해 추가적인 실험이 필요합니다.