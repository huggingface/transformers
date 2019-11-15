from pyspark import SparkContext, SparkConf

#conf = SparkConf().setAppName("preprocessing")
#sc = SparkContext(conf=conf)

text_file = sc.textFile("/mnt/raid/gits/Graphical-Summarization/Preprocessing/pre/opinosis/*.txt,/mnt/raid/gits/Graphical-Summarization/Preprocessing/sum/opinosis/*.txt")
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile("/mnt/raid/vocab.txt")
