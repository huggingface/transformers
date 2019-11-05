========================

Offensive Language Identification Dataset (OLID)
v 1.0: March 15 2018
https://scholar.harvard.edu/malmasi/olid

========================

1) DESCRIPTION

This is the README file for OLID described in: https://arxiv.org/abs/1902.09666

OLID contains 14,100 annotate tweets. It has been used as the official dataset for OffensEval: Identifying and Categorizing Offensive Language in Social Media (SemEval 2019 - Task 6): https://competitions.codalab.org/competitions/20011

The files included are: 

- olid-training-v1.tsv contains 13,240 annotated tweets. 
- olid-annotation.txt contains a short summary of the annotation guidelines.
- testset-levela.tsv contains the test set instances of level a.
- testset-levelb.tsv contains the test set instances of level b.
- testset-levelc.tsv contains the test set instances of level c.
- labels-levela.csv contains the gold labels and IDs of the instances in test set layer a.
- labels-levelb.csv contains the gold labels and IDs of the instances test set layer b.
- labels-levelc.csv contains the gold labels and IDs of the instances test set layer c.

The dataset was annotated using crowdsourcing. The gold labels were assigned taking the agreement of three annotators into consideration. No correction has been carried out on the crowdsourcing annotations. 

Twitter user mentions were substituted by @USER and URLs have been substitute by URL.

OLID is annotated using a hierarchical annotation. Each instance contains up to 3 labels each corresponding to one of the following levels:

- Level (or sub-task) A: Offensive language identification; 

- Level (or sub-task) B: Automatic categorization of offense types;

- Level (or sub-task) C: Offense target identification.	

2) FORMAT

Instances are included in TSV format as follows:

ID	INSTANCE	SUBA	SUBB	SUBC 

Whenever a label is not given, a value NULL is inserted (e.g. INSTANCE	NOT	NULL	NULL)

The column names in the file are the following:

id	tweet	subtask_a	subtask_b	subtask_c

The labels used in the annotation are listed below.

3) TASKS AND LABELS

(A) Level A: Offensive language identification

- (NOT) Not Offensive - This post does not contain offense or profanity.
- (1) Offensive - This post contains offensive language or a targeted (veiled or direct) offense

In our annotation, we label a post as offensive (1) if it contains any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct.

(B) Level B: Automatic categorization of offense types

- (TIN) Targeted Insult and Threats - A post containing an insult or threat to an individual, a group, or others (see categories in sub-task C).
- (UNT) Untargeted - A post containing non-targeted profanity and swearing.

Posts containing general profanity are not targeted, but they contain non-acceptable language.

(C) Level C: Offense target identification

- (IND) Individual - The target of the offensive post is an individual: a famous person, a named individual or an unnamed person interacting in the conversation.
- (GRP) Group - The target of the offensive post is a group of people considered as a unity due to the same ethnicity, gender or sexual orientation, political affiliation, religious belief, or something else.
- (OTH) Other – The target of the offensive post does not belong to any of the previous two categories (e.g., an organization, a situation, an event, or an issue)

Label Combinations

Here are the possible label combinations in the OLID annotation.

-	NOT NULL NULL
-	1 UNT NULL
-	1 TIN (IND|GRP|OTH)

4) CITING THE DATASET

If you use this dataset we kindly ask you to include a reference to the paper below. 

@inproceedings{zampierietal2019, 
title={{Predicting the Type and Target of Offensive Posts in Social Media}}, 
author={Zampieri, Marcos and Malmasi, Shervin and Nakov, Preslav and Rosenthal, Sara and Farra, Noura and Kumar, Ritesh}, 
booktitle={Proceedings of NAACL}, 
year={2019}, 
} 

The paper will be presented at NAACL and the pre-print is available on arXiv.org: https://arxiv.org/abs/1902.09666

If you would like to refer to the OffensEval competition, here is the bib entry to the report:

@inproceedings{offenseval,
title={{SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval)}},
author={Zampieri, Marcos and Malmasi, Shervin and Nakov, Preslav and Rosenthal, Sara and Farra, Noura and Kumar, Ritesh},
booktitle={Proceedings of The 13th International Workshop on Semantic Evaluation (SemEval)},
year={2019}
}
