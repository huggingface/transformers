#!/usr/bin/env python3
import tensorflow_text  # noqa: F401

# TF1 version
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

tf.disable_eager_execution()

text_generator = hub.Module('https://tfhub.dev/google/bertseq2seq/roberta24_cnndm/1')
article = """(CNN)Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members singing a racist chant. SAE\'s national chapter suspended the students, but University of Oklahoma President David Boren took it a step further, saying the university\'s affiliation with the fraternity is permanently done. The news is shocking, but it\'s not the first time SAE has faced controversy. SAE was founded March 9, 1856, at the University of Alabama, five years before the American Civil War, according to the fraternity website. When the war began, the group had fewer than 400 members, of which "369 went to war for the Confederate States and seven for the Union Army," the website says. The fraternity now boasts more than 200,000 living alumni, along with about 15,000 undergraduates populating 219 chapters and 20 "colonies" seeking full membership at universities. SAE has had to work hard to change recently after a string of member deaths, many blamed on the hazing of new recruits, SAE national President Bradley Cohen wrote in a message on the fraternity\'s website. The fraternity\'s website lists more than 130 chapters cited or suspended for "health and safety incidents" since 2010. At least 30 of the incidents involved hazing, and dozens more involved alcohol. However, the list is missing numerous incidents from recent months. Among them, according to various media outlets: Yale University banned the SAEs from campus activities last month after members allegedly tried to interfere with a sexual misconduct investigation connected to an initiation rite. Stanford University in December suspended SAE housing privileges after finding sorority members attending a fraternity function were subjected to graphic sexual content. And Johns Hopkins University in November suspended the fraternity for underage drinking. "The media has labeled us as the \'nation\'s deadliest fraternity,\' " Cohen said. In 2011, for example, a student died while being coerced into excessive alcohol consumption, according to a lawsuit. SAE\'s previous insurer dumped the fraternity. "As a result, we are paying Lloyd\'s of London the highest insurance rates in the Greek-letter world," Cohen said. Universities have turned down SAE\'s attempts to open new chapters, and the fraternity had to close 12 in 18 months over hazing incidents."""

input_documents = [article]
output_summaries = text_generator(input_documents)

init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
with tf.Session() as sess:
    sess.run([init, table_init])
    print(sess.run(output_summaries))
