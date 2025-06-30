# auto-evol-instruct
Auto Instruction Evolution는 LLM 파인튜닝에 사용되는 instruction의 complexity를 인간의 개입 없이 자동으로 향상시키는 프레임워크입니다.

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