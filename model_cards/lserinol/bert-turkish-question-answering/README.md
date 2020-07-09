---
language: turkish
---

# bert-turkish-question-answering

## Usage

```python
from transformers import pipeline
nlp = pipeline('question-answering', model='lserinol/bert-turkish-question-answering', tokenizer='lserinol/bert-turkish-question-answering')
nlp({
    'question': "Ankara'da kaç ilçe vardır?",
    'context': r"""Türkiye'nin başkenti Ankara'dır. Ülkenin en büyük idari birimleri illerdir ve 81 il vardır. Bu iller ilçelere ayrılmıştır, toplamda 973 ilçe mevcuttur."""
})
```

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("lserinol/bert-turkish-question-answering")
model = AutoModelForQuestionAnswering.from_pretrained("lserinol/bert-turkish-question-answering")
text = r"""
Ankara'nın başkent ilan edilmesinin ardından (13 Ekim 1923) şehir hızla gelişmiş ve Türkiye'nin ikinci en kalabalık ili olmuştur.
Türkiye Cumhuriyeti'nin ilk yıllarında ekonomisi tarım ve hayvancılığa dayanan ilin topraklarının yarısı hâlâ tarım amaçlı 
kullanılmaktadır. Ekonomik etkinlik büyük oranda ticaret ve sanayiye dayalıdır. Tarım ve hayvancılığın ağırlığı ise giderek 
azalmaktadır. Ankara ve civarındaki gerek kamu sektörü gerek özel sektör yatırımları, başka illerden büyük bir nüfus göçünü 
teşvik etmiştir. Cumhuriyetin kuruluşundan günümüze, nüfusu ülke nüfusunun iki katı hızda artmıştır. Nüfusun yaklaşık dörtte 
üçü hizmet sektörü olarak tanımlanabilecek memuriyet, ulaşım, haberleşme ve ticaret benzeri işlerde, dörtte biri sanayide, 
%2'si ise tarım alanında çalışır. Sanayi, özellikle tekstil, gıda ve inşaat sektörlerinde yoğunlaşmıştır. Günümüzde ise en çok 
savunma, metal ve motor sektörlerinde yatırım yapılmaktadır. Türkiye'nin en çok sayıda üniversiteye sahip ili olan Ankara'da 
ayrıca, üniversite diplomalı kişi oranı ülke ortalamasının iki katıdır. Bu eğitimli nüfus, teknoloji ağırlıklı yatırımların 
gereksinim duyduğu iş gücünü oluşturur. Ankara'dan otoyollar, demir yolu ve hava yoluyla Türkiye'nin diğer şehirlerine ulaşılır.
Ankara aynı zamanda başkent olarak Türkiye Büyük Millet Meclisi (TBMM)'ye de ev sahipliği yapmaktadır.
"""

questions = [
    "Ankara kaç yılında başkent oldu?",
    "Ankara ne zaman başkent oldu?",
    "Ankara'dan başka şehirlere nasıl ulaşılır?",
    "TBMM neyin kısaltmasıdır?"
]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
  ```
