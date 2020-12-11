---
language: id
---

# Indonesian T5 Summarization Small Model

Finetuned T5 small summarization model for Indonesian. 

## Finetuning Corpus

`t5-small-indonesian-summarization-cased` model is based on `t5-small-bahasa-summarization-cased` by [huseinzol05](https://huggingface.co/huseinzol05), finetuned using [indosum](https://github.com/kata-ai/indosum) dataset.

## Load Finetuned Model

```python
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("panggi/t5-small-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained("panggi/t5-small-indonesian-summarization-cased")
```

## Code Sample

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("panggi/t5-small-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained("panggi/t5-small-indonesian-summarization-cased")

# https://www.sehatq.com/artikel/apa-itu-dispepsia-fungsional-ketahui-gejala-dan-faktor-risikonya
ARTICLE_TO_SUMMARIZE = "Secara umum, dispepsia adalah kumpulan gejala pada saluran pencernaan seperti nyeri, sensasi terbakar, dan rasa tidak nyaman pada perut bagian atas. Pada beberapa kasus, dispepsia yang dialami seseorang tidak dapat diketahui penyebabnya. Jenis dispepsia ini disebut dengan dispepsia fungsional. Apa saja gejala dispepsia fungsional? Apa itu dispepsia fungsional? Dispepsia fungsional adalah kumpulan gejala tanpa sebab pada saluran pencernaan bagian atas. Gejala tersebut dapat berupa rasa sakit, nyeri, dan tak nyaman pada perut bagian atas atau ulu hati. Penderita dispepsia fungsional juga akan merasakan kenyang lebih cepat dan sensasi perut penuh berkepanjangan. Gejala-gejala tersebut bisa berlangsung selama sebulan atau lebih. Dispepsia ini memiliki nama “fungsional” karena kumpulan gejalanya tidak memiliki penyebab yang jelas. Dilihat dari fungsi dan struktur saluran pencernaan, dokter tidak menemukan hal yang salah. Namun, gejalanya bisa sangat mengganggu dan menyiksa. Dispepsia fungsional disebut juga dengan dispepsia nonulkus. Diperkirakan bahwa 20% masyarakat dunia menderita dispepsia fungsional. Kondisi ini berisiko tinggi dialami oleh wanita, perokok, dan orang yang mengonsumsi obat anti-peradangan nonsteroid (NSAID). Dispepsia fungsional bisa bersifat kronis dan mengganggu kehidupan penderitanya. Namun beruntung, ada beberapa strategi yang bisa diterapkan untuk mengendalikan gejala dispepsia ini. Strategi tersebut termasuk perubahan gaya hidup, obat-obatan, dan terapi.Ragam gejala dispepsia fungsional Gejala dispepsia fungsional dapat bervariasi antara satu pasien dengan pasien lain. Beberapa tanda yang bisa dirasakan seseorang, yaitu:  Sensasi terbakar atau nyeri di saluran pencernaan bagian atas  Perut kembung  Cepat merasa kenyang walau baru makan sedikit  Mual  Muntah  Bersendawa  Rasa asam di mulut  Penurunan berat badan  Tekanan psikologis terkait dengan kondisi yang dialami  Apa sebenarnya penyebab dispepsia fungsional?  Sebagai penyakit fungsional, dokter mengkategorikan dispepsia ini sebagai penyakit yang tidak diketahui penyebabnya.   Hanya saja, beberapa faktor bisa meningkatkan risiko seseorang terkena dispepsia fungsional. Faktor risiko tersebut, termasuk:  Alergi terhadap zat tertentu  Perubahan mikrobioma usus  Infeksi, seperti yang dipicu oleh bakteriHelicobacter pylori  Sekresi asam lambung yang tidak normal  Peradangan pada saluran pencernaan bagian atas  Gangguan pada fungsi lambung untuk mencerna makanan  Pola makan tertentu  Gaya hidup tidak sehat  Stres  Kecemasan atau depresi  Efek samping pemakaian obat seperti obat antiinflamasi nonsteroid  Penanganan untuk dispepsia fungsional  Ada banyak pilihan pengobatan untuk dispepsia fungsional. Seperti yang disampaikan di atas, tidak ada penyebab tunggal dispepsia ini yang bisa diketahui. Gejala yang dialami antara satu pasien juga mungkin amat berbeda dari orang lain.  Dengan demikian, jenis pengobatan dispepsia fungsional juga akan bervariasi. Beberapa pilihan strategi penanganan untuk dispepsia fungsional, meliputi:  1. Obat-obatan  Ada beberapa jenis obat yang mungkin akan diberikan dokter, seperti  Obat penetral asam lambung yang disebut penghambat reseptor H2  Obat penghambat produksi asam lambung yang disebut proton pump inhibitors  Obat untuk mengendalikan gas di perut yang mengandung simetikon  Antidepresan seperti amitriptyline  Obat penguat kerongkongan yang disebut agen prokinetik  Obat untuk pengosongan isi lambung seperti metoclopramide  Antibiotik jika dokter mendeteksi adanya infeksi bakteri H. pylori 2. Anjuran terkait perubahan gaya hidup  Selain obat-obatan, dokter akan memberikan rekomendasi perubahan gaya hidup yang harus diterapkan pasien. Tips terkait perubahan gaya hidup termasuk:  Makan lebih sering namun dengan porsi yang lebih sedikit  Menjauhi makanan berlemak karena memperlambat pengosongan makanan di lambung  Menjauhi jenis makanan lain yang memicu gejala dispepsia, seperti makanan pedas, makanan tinggi asam, produk susu, dan produk kafein  Menjauhi rokok  Dokter juga akan meminta pasien untuk mencari cara untuk mengendalikan stres, tidur dengan kepala lebih tinggi, dan menjalankan usaha untuk mengendalikan berat badan.  Apakah penyakit dispepsia itu berbahaya?  Dispepsia, termasuk dispepsia fungsional, dapat menjadi kronis dengan gejala yang menyiksa. Jika tidak ditangani, dispepsia tentu dapat berbahaya dan mengganggu kehidupan pasien. Segera hubungi dokter apabila Anda merasakan gejala dispepsia, terlebih jika tidak merespons obat-obatan yang dijual bebas.  Catatan dari SehatQ  Dispepsia fungsional adalah kumpulan gejala pada saluran pencernaan bagian atas yang tidak diketahui penyebabnya. Dispepsia fungsional dapat ditangani dengan kombinasi obat-obatan dan perubahan gaya hidup.   Jika masih memiliki pertanyaan terkait dispepsia fungsional, Anda bisa menanyakan ke dokter di aplikasi kesehatan keluarga SehatQ. Aplikasi SehatQ bisa diunduh gratis di Appstore dan Playstore yang berikan informasi penyakit terpercaya."
# generate summary
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=5000, return_tensors='pt', truncation=True)
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
```

Output:

```
'Dispepsia fungsional adalah kumpulan gejala tanpa sebab pada saluran pencernaan bagian atas. Gejala tersebut dapat berupa rasa sakit, nyeri, dan tak nyaman pada perut bagian atas. Penderita dispepsia fungsional juga akan merasakan kenyang lebih cepat dan sensasi perut penuh berkepanjangan. Gejala-gejala tersebut bisa berlangsung selama sebulan atau lebih.
```

## Acknowledgement

Thanks to Immanuel Drexel for his article [Text Summarization, Extractive, T5, Bahasa Indonesia, Huggingface’s Transformers](https://medium.com/analytics-vidhya/text-summarization-t5-bahasa-indonesia-huggingfaces-transformers-ee9bfe368e2f)
