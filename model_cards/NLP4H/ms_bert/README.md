# MS-BERT

## Introduction

This repository provides codes and models of MS-BERT.
MS-BERT was pre-trained on notes from neurological examination for Multiple Sclerosis (MS) patients at St. Michael's Hospital in Toronto, Canada.

## Data

The dataset contained approximately 75,000 clinical notes, for about 5000 patients, totaling to over 35.7 million words.
These notes were collected from patients who visited St. Michael's Hospital MS Clinic between 2015 to 2019.
The notes contained a variety of information pertaining to a neurological exam.
For example, a note can contain information on the patient's condition, their progress over time and diagnosis.
The gender split within the dataset was observed to be 72% female and 28% male ([which reflects the natural discrepancy seen in MS][1]).
Further sections will describe how MS-BERT was pre trained through the use of these clinically relevant and rich neurological notes.

## Data pre-processing

The data was pre-processed to remove any identifying information. This includes information on: patient names, doctor names, hospital names, patient identification numbers, phone numbers, addresses, and time. In order to de-identify the information, we used a curated database that contained patient and doctor information. This curated database was paired with regular expressions to find and remove any identifying pieces of information. Each of these identifiers were replaced with a specific token. These tokens were chosen based on three criteria: (1) they belong to the current BERT vocab, (2), they have relatively the same semantic meaning as the word they are replacing, and (3), the token is not found in the original unprocessed dataset. The replacements that met the criteria above were as follows: 

Female first names -> Lucie

Male first names -> Ezekiel

Last/family names -> Salamanca.

Dates -> 2010s

Patient IDs -> 999

Phone numbers -> 1718

Addresses -> Silesia

Time -> 1610

Locations/Hospital/Clinic names -> Troy

## Pre-training

The starting point for our model is the already pre-trained and fine-tuned BLUE-BERT base. We further pre-train it using the masked language modelling task from the huggingface transformers [library](https://github.com/huggingface). 

The hyperparameters can be found in the config file in this repository or [here](https://s3.amazonaws.com/models.huggingface.co/bert/NLP4H/ms_bert/config.json)

## Acknowledgements

We would like to thank the researchers and staff at the Data Science and Advanced Analytics (DSAA) department, St. Michael’s Hospital, for providing consistent support and guidance throughout this project.
We would also like to thank Dr. Marzyeh Ghassemi, Taylor Killan, Nathan Ng and Haoran Zhang for providing us the opportunity to work on this exciting project.

## Disclaimer

MS-BERT shows the results of research conducted at the Data Science and Advanced Analytics (DSAA) department, St. Michael’s Hospital. The results produced by MS-BERT are not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not make decisions about their health solely on the basis of the results produced by MS-BERT. St. Michael’s Hospital does not independently verify the validity or utility of the results produced by MS-BERT. If you have questions about the results produced by MS-BERT please consult a healthcare professional. If you would like more information about the research conducted at DSAA please contact [Zhen Yang](mailto:zhen.yang@unityhealth.to). If you would like more information on neurological examination notes please contact [Dr. Tony Antoniou](mailto:tony.antoniou@unityhealth.to) or [Dr. Jiwon Oh](mailto:jiwon.oh@unityhealth.to) from the MS clinic at St. Michael's Hospital.

[1]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707353/
