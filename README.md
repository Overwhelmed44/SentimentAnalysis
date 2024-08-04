# Sentiment Analysis
Console app that analyse the sentiment of reviews\
Knows 5000 russian words

- Python 3.12 used
- RNN model built with keras
- Yandex [dataset](https://github.com/yandex/geo-reviews-dataset-2023) used

I haven\`t included that dataset in the project cuz it\`s too big
<!--хочу работать у вас-->

# Try it out
Clone the repository
```
git clone https://github.com/Overwhelmed44/SentimentAnalysis.git
```
Install all required packages
```
pip install -r requirements.txt
```
Run the console app
```python
from app.eval import evaluate_text

def main():
    while True:
        print(evaluate_text(input('>>> ')), end='\n\n')

if __name__ == '__main__':
    main()
```

# Example
```
>>> Суперское место! Обязательно посещу еще раз!
Положительный отзыв

>>> Больше не приеду
Отрицательный отзыв
```
