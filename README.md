# KodLandAI: Библиотека для Упрощенной Работы с Нейросетями и Машинным Обучением.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://example.com/kodlandai_logo.gif" alt="KodLandAI Logo" width="400"/>
</p>

**KodLandAI** – это библиотека Python, разработанная для упрощения и ускорения процесса создания, обучения и развертывания моделей машинного обучения и нейронных сетей. Она предлагает набор инструментов и оптимизированных алгоритмов, которые позволяют как начинающим, так и опытным специалистам эффективно работать с ML&DL связанных с машинным обучением.

## Особенности библиотеки

*   **Простота использования:** Интуитивно понятный синтаксис , понятные команды + объяснение.
*   **Оптимизация:** Включает в себя набор оптимизированных алгоритмов, разработанных для достижения высокой производительности и эффективности при обучении моделей.
*   **Гибкость:** Легко интегрируется с другими популярными библиотеками машинного обучения, такими как PyTorch и TensorFlow, предоставляя возможность использования в различных сценариях.

## Установка

Для установки достатточно скачать данный архив и начать работу используя стандартные импорты.


## Модули и Функциональность

### 1. KodLand\_Optimizer

Пользовательский оптимизатор `Kod`, созданный на основе `torch.optim.Optimizer`.

*   **Функциональность:** Реализует оптимизацию параметров модели на основе momentum.

*   **Пример использования:**

import torch
from KodLandAI import KodLand_Optimizer

model = ВашаМодель()
optimizer = KodLand_Optimizer(model.parameters(), lr=0.001, beta=0.9)


### 2. HIGGS

Оптимизатор `HiggsOptimizer`, предназначенный для трансформерных моделей.

*   **Функциональность:** Адаптирует параметры модели, используя нормализацию градиента и параметров для повышения стабильности и скорости обучения.

*   **Пример использования:**


### 2. HIGGS

Оптимизатор `HiggsOptimizer`, предназначенный для трансформерных моделей.

*   **Функциональность:** Адаптирует параметры модели, используя нормализацию градиента и параметров для повышения стабильности и скорости обучения.

*   **Пример использования:**

from KodLandAI import HIGGS
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = HIGGS.HiggsOptimizer(model.parameters(), lr=2e-5, beta=0.95)


### 3. RNN

Класс `RNN` для создания рекуррентных нейронных сетей с различными типами ячеек (LSTM, GRU, RNN).

*   **Функциональность:** Позволяет создавать и обучать рекуррентные сети для обработки последовательностей с механизмом внимания.

*   **Пример использования:**

import torch
from KodLandAI import RNN

model = RNN.RNN(input_size=10, hidden_size=64, output_size=1, num_layers=2, rnn_type='lstm')
input_seq = torch.randn(32, 20, 10) # batch_size, seq_len, input_size
output = model(input_seq)

### 4. MLP

Класс `MLP` для создания многослойных персептронов.

*   **Функциональность:** Предоставляет простую структуру MLP для решения задач классификации и регрессии. Включает функцию `train_mlp` для обучения модели с использованием оптимизатора AdamW.

*   **Пример использования:**

import torch
from KodLandAI import MLP

X = torch.randn(100, 10)
y = torch.randn(100, 1)

model = MLP.MLP(input_size=10, hidden_size=50, output_size=1)
MLP.train_mlp(model, X, y, optim_name='adamw', lr=0.001)


### 5. optim

Набор оптимизаторов, включая `Lion`, `AdamW` и `AdaGrad`.

*   **Функциональность:** Предоставляет различные алгоритмы оптимизации для обучения моделей.

    *   `Lion`: Альтернативный оптимизатор, использующий momentum и знак градиента для обновления параметров.
    *   `AdamW`: Улучшенная версия Adam, включающая регуляризацию весов.
    *   `AdaGrad`: Оптимизатор с адаптивной скоростью обучения для каждого параметра.

*   **Пример использования:**

import torch
from KodLandAI import optim

model = YourModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


