# Sentence_Generation-model-with-LSTM Model
- 단타 카운터 잡이용 하루만에 뚜들겨본, LSTM을 응용한 프로젝트 입니다.
- 문장을 예측하고 생성하는 모듈로, 매우 간단한 코드로 이루어져 있어 입문용으로 적합할 것 같아 만들어 보았습니다.
- 정확도 상승을 위해서는 형태소 분석을 비롯하여, 부수적인 부분을 추가하여 차츰 발전시켜 나가면 됩니다.
- 저도 물론 여기서 더 발전시켜 만들어지는 프로젝트(예정)들을 추가하는 형식으로 올려보겠습니다.

## Architecture
<img src="https://user-images.githubusercontent.com/79067558/108952434-c6ff3a80-76ac-11eb-9421-5ac4c3114386.png" width="70%" height="70%"><br/>

# Sentence_Generation-model-with-RNN Model
- 이 친구는 좀 정석적으로 꾸려봤습니다, 제목에 명시된대로 RNN을 응용한 프로젝트입니다.
- 제가... 자연어 처리에 관심이 많아서 참 이것저것 손벌린 것은 많습니다.
- 그래도 이거 참고하신 후에 LSTM 모델을 살펴보신다면, 어떤 부분을 추가하면 좋고 어떻게 진행이 이루어져야 깔끔한지
- 오히려 비교에 수월하실 것 같아 이것 또한 올려보았습니다 :) 좋게 봐주시면 감사하겠습니다 헤헤

## How To Work 
<img src="https://user-images.githubusercontent.com/79067558/108954303-f794a380-76af-11eb-8bbc-a2d223bf8e5f.png" width="70%" height="70%"><br/>

## Predict Loop
<img src="https://user-images.githubusercontent.com/79067558/108955670-ec427780-76b1-11eb-8221-7ec0802a8dbf.png" width="70%" height="70%"><br/>

## Models
|Title|Contents|Explanation|From|
|:------:|:---:|:------:|:--------:|
|Data Name|concateText.py|concatenate texts, of course It's free to use which if u don't want, keep steady on your mind eheh(반드시 사용할 필요는 없습니다, 자유입니다)|My Brain(나의 머리)
|Data Name|TextMaker.py|Sentence Generation model with LSTM(LSTM을 응용한 문장 생성)|My Brain(나의 머리)|
|Data Name|RNNTextMaker.py|Sentence Generation model with RNN(RNN을 응용한 문장 생성)|My Brain(나의 머리)|
