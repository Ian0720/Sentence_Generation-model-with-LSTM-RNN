import tensorflow as tf

import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 읽은 다음 파이썬 2와 호환되도록 디코딩합니다.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# 텍스트의 길이는 그 안에 있는 문자의 수입니다.
print ('텍스트의 길이: {}자'.format(len(text)))

# 텍스트의 처음 250자를 살펴봅니다
print(text[:250])

# 파일의 고유 문자수를 출력합니다.
vocab = sorted(set(text))
print ('고유 문자수 {}개'.format(len(vocab)))

# 고유 문자에서 인덱스로 매핑 생성
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# 텍스트에서 처음 13개의 문자가 숫자로 어떻게 매핑되었는지를 보여줍니다
print ('{} ---- 문자들이 다음의 정수로 매핑되었습니다 ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# 단일 입력에 대해 원하는 문장의 최대 길이
seq_length = 100
examples_per_epoch = len(text)//seq_length

# 훈련 샘플/타깃 만들기
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))

# 각 시퀀스에서, map 메서드를 사용하여 각 배치에 간단한 함수를 적용한 후 입력 테스트와 타깃 텍스트를 복사 및 이동합니다.
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# 첫 번째 샘플의 타깃 값을 출력해줍니다.
for input_example, target_example in  dataset.take(1):
  print ('입력 데이터: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('타깃 데이터: ', repr(''.join(idx2char[target_example.numpy()])))

# 이 벡터의 각 인덱스는 하나의 타임스텝으로 처리됩니다.
# 타임 스텝 0의 입력으로 모델은 "F"의 인덱스를 받고 다음 문자로 "i"의 인덱스를 예측합니다.
# 다음 타임 스텝에서도 같은 일을 하지만, 현재 입력 문자 외에 이전 타임 스텝의 컨텍스트(Context)를 고려합니다.

# 배치 크기
BATCH_SIZE = 64
# 데이터셋을 섞을 버퍼 크기
# (TF 데이터는 무한한 시퀀스와 함께 작동이 가능하도록 설계되었으며,
# 따라서 전체 시퀀스를 메모리에 섞지 않습니다. 대신에,
# 요소를 섞는 버퍼를 유지합니다).
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 문자로 된 어휘 사전의 크기
vocab_size = len(vocab)
# 임베딩 차원
embedding_dim = 256
# RNN 유닛(unit) 개수입니다, 1024로 설정했습니다.
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

# 여기서부터는 모델을 사용하는 구간입니다.
# 모델을 실행하여 원하는대로 동작하는지 확인하려 합니다, 이전에 먼저 출력 형태를 확인하도록 하겠습니다.
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (배치 크기, 시퀀스 길이, 어휘 사전 크기)")

# 모델로부터 실측을 얻기 위해서, 출력 배열에서 샘플링하고 실제 문자 인덱스를 얻어야 합니다.
# 이 분포는 문자 어휘에 대한 로직에 의해 정의됩니다.
# 배열에 Argmax를 취하면, 모델이 쉽게 루프에 걸릴 수 있습니다 ㅠㅠ 따라서 배열에서 샘플링하는 것이 중요합니다.
model.summary()
# 배치의 첫 번째 샘플링을 시도해보는 구간입니다.
# 이렇게하면 각 타임스텝에서 다음 문자 인덱스에 대한 예측을 제공합니다.
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print("입력: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("예측된 다음 문자: \n", repr("".join(idx2char[sampled_indices ])))

# 이제는 모델 훈련 파트입니다, 옵티마이저를 붙입니다.
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("예측 배열 크기(shape): ", example_batch_predictions.shape, " # (배치 크기, 시퀀스 길이, 어휘 사전 크기")
print("스칼라 손실:          ", example_batch_loss.numpy().mean())

# tf.keras.Model.compile 메서드를 사용하여 훈련 절차를 설정합니다.
# 기본 매개변수의 tf.keras.optimizers.Adam과 손실 함수를 사용합니다.
model.compile(optimizer='adam', loss=loss)

# 다음은 체크포인트를 구성해줍니다.
# tf.keras.callbacks.ModelCheckpoint를 이용하여 훈련 중 체크포인트가 저장되도록 합니다.
# 체크포인트가 저장될 디렉토리
checkpoint_dir = './training_checkpoints'
# 체크포인트 파일 이름
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# 다음은 훈련 실행파트 입니다.
# 저같은 경우는 너무 길지 않도록 10회만 에포크를 사용하겠습니다, Colab을 이용하실 경우 런타임을 GPU로 설정하여 보다 빠르게 진행시킬 수 있습니다!
EPOCHS = 10 
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# 텍스트 생성
# 최근 체크 포인트를 복원하고, 예측 단계에 돌입해보려 합니다.
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_string):
  # 평가 단계 (학습된 모델을 사용하여 텍스트 생성)

  # 생성할 문자의 수
  num_generate = 1000

  # 시작 문자열을 숫자로 변환(벡터화)시킵니다.
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 결과를 저장할 빈 문자열
  text_generated = []

  # 온도가 낮으면 더 예측 가능한 텍스트가 됩니다.
  # 온도가 높으면 더 의외의 텍스트가 됩니다.
  # 최적의 세팅을 찾기 위한 실험
  temperature = 1.0

  # 여기에서 배치 크기 == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # 배치 차원 제거
      predictions = tf.squeeze(predictions, 0)

      # 범주형 분포를 사용하여 모델에서 리턴한 단어 예측
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # 예측된 단어를 다음 입력으로 모델에 전달
      # 이전 은닉 상태와 함께
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))