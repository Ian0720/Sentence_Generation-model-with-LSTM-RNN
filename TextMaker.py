import codecs # 이 모듈은 표준 파이썬 코덱의 베이스 클래스를 정의하고, 코덱과 에러 처리 조회 프로세스를 관리하는 내부 파이썬 코덱 레지스트리에 대한 엑세스를 제공합니다.
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout 
from keras.layers import LSTM 
from keras.optimizers import RMSprop # RMSprop는 과거의 모든 기울기를 균일하게 더하지 않고, 
                                     # 새로운 기울기의 정보만 반영하도록 해서 학습률이 크게 떨어져 0에 가까워 지는 것을 방지하는 방법입니다.
from keras.utils.data_utils import get_file
import numpy as np 
import random, sys 

# 파일을 읽어오는 부분입니다, 저는 국립국어원의 토지를 가져왔습니다.
fileOpen = codecs.open("./concatenate_texts.txt", "r", encoding="utf-16") # 문자열 앞에 r이 붙으면 해당 문자열이 구성된 그대로 문자열로 반환됩니다.
# UTF-16은 유니코드 문자 인코딩 방식의 하나입니다, 주로 사용되는 기본 다국어 평면에 속하는 문자들은 그대로 16비트 값으로 인코딩이 되고
# 그 이상의 문자는 특별히 정해진 방식으로 32비트로 인코딩이 됩니다.
soup = BeautifulSoup(fileOpen, "html.parser") # HTML와 XHTML 형식의 텍스트 파일을 구문 분석하기 위한 기초로 사용되는 클래스라고 보시면 됩니다.
body = soup.select_one("body")
text = body.getText() + " "
print('Corpus Length: ', len(text))

# 문자를 하나하나 읽어들인 후, ID를 붙여주는 부분입니다.
charset = sorted(list(set(text)))
print('Number of Characters in use:',len(charset))
charset_indices = dict((c, i)for i, c in enumerate(charset)) # 문자를 ID로 
indices_charset = dict((i, c)for i, c in enumerate(charset)) # ID를 문자로

# 텍스트를 MaxLength개의 문자로 잘라내고, 다음에 오는 문자를 등록해주는 부분입니다.
MaxLength = 20
step = 3
sentences = []
next_charset = []
for i in range(0, len(text) - MaxLength, step):
    sentences.append(text[i: i+MaxLength])
    next_charset.append(text[i + MaxLength]) # 텍스트를 MaxLength개의 문자로 자릅니다.
print('Number of phrases to learn:',len(sentences))

print('Converts text to ID vector....')
X = np.zeros((len(sentences), MaxLength, len(charset)), dtype=np.bool) # np.zeros : 0으로 초기화 된, shape 차원의 np.array배열의 객체를 반환해줍니다.
y = np.zeros((len(sentences), len(charset)), dtype=np.bool) # np.bool : Boolean array 혹은 마스크라고 합니다.
for i, sentences in enumerate(sentences): # enumerate : return 값으로 index를 포함하는 enumerate 객체를 반환합니다.
    for t, char in enumerate(sentences):
        X[i, t, charset_indices[char]] = 1
    y[i, charset_indices[next_charset[i]]] = 1

# 모델 구축(LSTM)하는 부분입니다.
print('Build up model...')
model = Sequential()  #모델은 Sequential로 하겠습니다.
model.add(LSTM(128, input_shape=(MaxLength, len(charset))))
model.add(Dense(len(charset)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate=0.01) # RMSprop는 과거의 모든 기울기를 균일하게 더하지 않고, 
                                        # 새로운 기울기의 정보만 반영하도록 해서 학습률이 크게 떨어져 0에 가까워 지는 것을 방지하는 방법입니다.
model.compile(loss='categorical_crossentropy', optimizer=optimizer) # 모델의 학습 과정을 설정합니다.

# 후보를 배열에서 꺼내줍니다.
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 학습을 시키고, 텍스트 생성을 반복해줍니다.
for iteration in range(1, 60): # iteration : '되풀이'를 의미합니다, 혹시나 해서 적어봅니다.
    print()
    print('-' * 50)
    print('Repetition = ', iteration)
    model.fit(X, y, batch_size=128, epochs=1)
    # 임의의 시작 텍스트를 선택합니다.
    start_index = random.randint(0, len(text) - MaxLength - 1)
    # 다양성을 가진 문장을 생성해줍니다.
    for diversity in [0.2, 0.5, 1.0, 1.2]: # diversity : '다양성'을 의미합니다, 이것도 혹시나 해서 적어봅니다.
        print()
        print('--- 다양성 = ', diversity)
        generated = ''
        sentences = text[start_index: start_index + MaxLength]
        generated += sentences
        print('--- 시드 = "' + sentences + '"')
        # 시드를 기반으로 하여, 텍스트를 자동으로 생성해주는 구간입니다.
        for i in range(400):
            x = np.zeros((1, MaxLength, len(charset)))
            for t, char in enumerate(sentences):
                x[0,t,charset_indices[char]] = 1.
                # 다음에 올 문자를 예측하는 부분입니다.
                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_charset = indices_charset[next_index]
                # 출력해주는 부분입니다.
                generated += next_charset
                sentences = sentences[1:] + next_charset
                sys.stdout.write(next_charset)
                sys.stdout.flush()
            print()

# 그 다음, 모델을 저장해줍니다.
from keras.models import load_model
model.save('Successful_data.h5')