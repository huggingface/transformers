---
language: ar
---
# Arabic Named Entity Recognition Model

Pretrained BERT-based ([arabic-bert-base](https://huggingface.co/asafaya/bert-base-arabic)) Named Entity Recognition model for Arabic.

The pre-trained model can recognize the following entities:
1. **PERSON**

-  و هذا ما نفاه المعاون السياسي للرئيس ***نبيه بري*** ، النائب ***علي حسن خليل***   

- لكن أوساط ***الحريري*** تعتبر أنه ضحى كثيرا في سبيل البلد 

- و ستفقد الملكة ***إليزابيث الثانية*** بذلك سيادتها على واحدة من آخر ممالك الكومنولث 

2. **ORGANIZATION**

- حسب أرقام ***البنك الدولي*** 

-  أعلن ***الجيش العراقي*** 

-  و نقلت وكالة ***رويترز*** عن ثلاثة دبلوماسيين في ***الاتحاد الأوروبي*** ، أن ***بلجيكا*** و ***إيرلندا*** و ***لوكسمبورغ*** تريد أيضاً مناقشة 

-  ***الحكومة الاتحادية*** و ***حكومة إقليم كردستان*** 

- و هو ما يثير الشكوك حول مشاركة النجم البرتغالي في المباراة المرتقبة أمام ***برشلونة*** الإسباني في 


3. ***LOCATION***

-  الجديد هو تمكين اللاجئين من “ مغادرة الجزيرة تدريجياً و بهدوء إلى ***أثينا*** ” 

-  ***جزيرة ساكيز*** تبعد 1 كم عن ***إزمير*** 


4. **DATE**

-  ***غدا الجمعة*** 

-  ***06 أكتوبر 2020*** 

- ***العام السابق*** 


5. **PRODUCT**

-  عبر حسابه ب ***تطبيق “ إنستغرام ”*** 

-  الجيل الثاني من ***نظارة الواقع الافتراضي أوكولوس كويست*** تحت اسم " ***أوكولوس كويست 2*** " 


6. **COMPETITION**

-  عدم المشاركة في ***بطولة فرنسا المفتوحة للتنس*** 

-  في مباراة ***كأس السوبر الأوروبي*** 

7. **PRIZE**

-  ***جائزة نوبل ل لآداب***

-  الذي فاز ب ***جائزة “ إيمي ” لأفضل دور مساند***

8. **EVENT**

-  تسجّل أغنية جديدة خاصة ب ***العيد الوطني السعودي***

- ***مهرجان المرأة يافوية*** في دورته الرابعة 

9. **DISEASE**

-  في مكافحة فيروس ***كورونا*** و عدد من الأمراض 

-  الأزمات المشابهة مثل “ ***انفلونزا الطيور*** ” و ” ***انفلونزا الخنازير*** 

## Example

[Find here a complete example to use this model](https://github.com/hatmimoha/arabic-ner)

Here is the map from index to label:

```
id2label = {
    "0": "B-PERSON",
    "1": "I-PERSON",
    "2": "B-ORGANIZATION",
    "3": "I-ORGANIZATION",
    "4": "B-LOCATION",
    "5": "I-LOCATION",
    "6": "B-DATE",
    "7": "I-DATE"",
    "8": "B-COMPETITION",
    "9": "I-COMPETITION",
    "10": "B-PRIZE",
    "11": "I-PRIZE",
    "12": "O",
    "13": "B-PRODUCT",
    "14": "I-PRODUCT",
    "15": "B-EVENT",
    "16": "I-EVENT",
    "17": "B-DISEASE",
    "18": "I-DISEASE",
}

```

## Training Corpus

The training corpus is made of 378.000 tokens (14.000 sentences) collected from the Web and annotated manually.

## Results

The results on a valid corpus made of 30.000 tokens shows an F-measure of ~87%.
