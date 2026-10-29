# Капчерка

Распознавание капчи на сайтах судов общей юрисдикции и мировых судей (ГАС «Правосудие», `*.sudrf.ru`, `*.msudrf.ru`). Одна модель для всех поддерживаемых типов, тип капчи указывать не нужно. Модель в формате ONNX весит 2,2 МБ и работает на CPU.

## Установка

```bash
pip install git+ssh://git@github.com/cluster-help/kapcherka.git
```

Python 3.9+. Зависимости: `onnxruntime`, `numpy`, `pillow`.

## Использование

```python
from kapcherka import Solver

solver = Solver()
result = solver.solve("captcha.png")
result.text                              # '25457'
```

`solve()` принимает путь к файлу, `bytes`, `PIL.Image` или `numpy.ndarray` и возвращает `Result(text, confidence)`. Экземпляр `Solver` потокобезопасен.

Ответ возвращается в нижнем регистре. Сайты судов принимают его в cp1251:

```python
from urllib.parse import quote

quote(result.text.encode("cp1251"))
```

### Командная строка

```bash
kapcherka captcha.png
kapcherka *.png --json
kapcherka bench
```

### HTTP

```bash
kapcherka serve --host 0.0.0.0 --port 8000
curl --data-binary @captcha.png http://localhost:8000/solve
```

```json
{"text": "25457", "confidence": 0.872, "ms": 2.1}
```

`POST /solve` принимает картинку в теле запроса либо JSON `{"image": "<base64>"}`. `GET /health` — проверка состояния.

### Docker

```bash
docker build -t kapcherka .
docker run -d -p 8000:8000 kapcherka
```

## Поддерживаемые капчи

| Источник | Размер | Ответ | Точность |
|---|---|---|---|
| `sudrf.ru`, цифровая | 100×30 | 5 цифр | 98,3 % |
| `sudrf.ru`, цифры и кириллица | 100×30 | 4–6 знаков | 98,0 % |
| Краснодарский краевой суд | 140×30 | 4–6 знаков | 98,3 % |
| Суды Карачаево-Черкесской Республики | 105×33 | 5 цифр | 86,7 % |
| `msudrf.ru`, мировые судьи | 250×100 | 5–7 букв | 99,3 % |

Точность — доля полностью верных ответов на отложенной выборке из 300 капч каждого типа.

## Производительность

onnxruntime, один поток: медиана 2,1 мс на капчу, около 470 капч в секунду на ядро.

## Обучение

Код обучения — в каталоге `training/`.

## Лицензия

[PolyForm Noncommercial 1.0.0](LICENSE.md). Коммерческое использование запрещено.
