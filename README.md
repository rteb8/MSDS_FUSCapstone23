# MSDS_FUSCapstone23
MSDS capstone project for the Focused Ultrasound Foundation (FUSF) by Rose Eluvathingal Muttikkal, Reanna Panagides, Skye Jung, and Abhishek Singh during the 2023-2024 academic year at the University of Virginia. Research presented at the 2024 IEEE SIEDS conference.

# Problem statement 
The current literature review process for the clinical team at Focused Ultrasound Foundation (FUSF) is time-consuming, inefficient, and prone to error

# Proposed solution
The primary need for the FUSF clinical team is to determine if an article they are reviewing during their monthly literature review is related to focused ultrasound. In its simplest form, this is a binary text classification problem. Given an Excel file with article abstract and metadata features, our model needs to accurately determine if it's relevance to focused ultrasound technology. Our proposed solution is to fine-tune various pre-trained BERT models on a curated collection of scientific articles related to and not related to focused ultrasound. We will then select the best-performing model to use for text classification.

The study was conducted in two distinct phases: 1) Exploration of Machine Learning (ML) methods for text classification, and 2) Integration of ML methods into the scientific literature review process. In the initial phase, we investigated various traditional and deep learning approaches to text classification to identify the most effective one. Subsequently, in the second phase, we adapted the existing literature review pipeline to include ML automation, aiming to enhance its efficiency.

# Dataset
The data comprises a curated selection of scientific abstracts gathered from PubMed, an NCBI-maintained search engine, supplemented by Excel files provided by the Focused Ultrasound Foundation (FUSF) from their monthly literature review process between February and August 2023, totaling 489 articles. Additionally, 90 labeled abstracts from previous FUSF work were included, along with 1960 FUS-related abstracts from a publicly accessible Zotero database on the FUSF website, excluding veterinary-related entries. To distinguish between articles on FUS technology and those on ultrasound for diagnostic purposes, PubMed articles with "ultrasound diagnosis" in their titles were added. The entire collection, comprising 1794 FUS-related abstracts and an equal number of non-FUS abstracts, was consolidated into a CSV file, removing duplicates and null values, resulting in a final dataset of 3588 abstracts.

# Data Preprocessing
The data preparation process for our analysis began with classifying scientific abstracts from the FUS therapy field, with each abstract marked by the Chief Medical Officer (CMO) of the Foundation based on relevance to the FUS domain, indicated by a binary label. Following data compilation, the pre-processing phase standardized the text for machine learning analysis. Abstracts were converted to lowercase strings, tokenized into individual words using a tokenizer compatible with the BERT model, and transformed into PyTorch tensors for computational alignment. Tokenized text underwent feature extraction using pre-trained BERT models, assigning vectors to tokens to capture meaning and structure, facilitated by BERT's transformer architecture and pre-training on large datasets. For logistic regression, Support Vector Machine (SVM), and Naive Bayes models, TF-IDF vectorization converted text into a matrix of TF-IDF features, considering word frequency and uniqueness across the dataset, optimizing performance for these traditional machine learning models.

# Model training
To address the primary aim of our study, we conducted a comparative analysis between traditional machine learning and deep learning methodologies for text classification. As for traditional machine learning methods, Logistic Regression, Naive Bayes, and Support Vector Machine (SVM) models were employed for binary classification. These models used features extracted from a TF-IDF tokenizer as parameters for binary classification, with no additional optimization performed.

In contrast, for the deep learning approach, we selected and trained several BERT (Bidirectional Encoder Representations from Transformers) models, including  TinyBERT, SciBERT, DistilBERT, and Bio-ClinicalBERT to gauge their performance. Each BERT model had been pre-trained on a distinct corpora of text. For instance, Bio-ClinicalBERT was pre-trained on biomedical and clinical text data, tailored for healthcare and biomedicine tasks, while DistilBERT and TinyBERT were scaled-down versions of the original BERT model designed for improved efficiency while maintaining performance. SciBERT, on the other hand, was pre-trained on scientific text, particularly enhancing performance within scientific domains.

In pursuit of our secondary aim, the chosen model was integrated into the literature review process. The initial steps of the pipeline, involving article retrieval and compilation, were automated using Python scripts to scrape articles from PubMed and preprocess the abstracts. Subsequently, our FusBERT model generated predictions regarding the relatedness of abstracts to FUS therapies, aiding researchers in efficiently screening articles. The workflow for integrating FusBERT underwent iterative refinement through collaboration with the clinical and data management team at the FUSF, with the final workflow presented in the subsequent Results section.

# Results
Overall, the deep learning models outperformed traditional machine learning methods used in this study. The deep learning models achieved higher accuracy, recall, and F1 scores. However, the traditional machine learning methods achieved higher precision scores. These metrics are based on model performance on a test dataset of 359 abstracts. The best performing fine-tuned FusBERT model exhibited the following performance metrics on our test data: accuracy 0.91, precision 0.85, recall 0.99, and F1 0.91.

# Solution workflow
Our best-performing FusBERT model was then integrated into the conventional literature review process. As mentioned previously, the literature review process consists of six distinct steps. Step 3, which involves screening for inclusion criteria, emerges as a prime candidate for leveraging machine learning techniques to enhance efficiency.

![alt text](https://github.com/rteb8/MSDS_FUSCapstone23/blob/main/handover/Worflow_Diagram.png)

The integration of ML into the relevant publication screening step of the entire process can be further delineated into five stages. Upon conducting searches in publication databases to identify relevant articles, researchers first export the search results into an Excel file. Subsequently, abstracts undergo preprocessing and tokenization to prepare them as inputs for our model. The FusBERT model receives these abstracts as inputs, generates predictions, and exports them back into the original Excel file. These predictions are then utilized to compile a final dataset comprising pertinent abstracts. This curated dataset is subsequently subject to quality assessment.

# Model limitations
The content of FUS-relevant abstracts may evolve over time, a phenomenon known as concept drift. Without a feedback loop for automatic adaptation to these shifting distributions, model performance could deteriorate as it struggles to cope with new data patterns. Periodic retraining may become necessary to uphold model accuracy and ensure it remains effective in its task.

# Future work
Looking ahead, the potential for expanding the use of BERT models in literature review processes is vast. One potential direction for future work involves developing BERT models that are capable of multi-class classification. This advancement would enable the models to categorize literature into multiple predefined categories, further refining the review process. This capability would significantly enhance the precision of literature reviews, making it easier for researchers to locate studies relevant with greater granularity. 

## Important files
| File path                                    | Description                                                                                                   | 
|:--------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
| dataset/zotero_data.csv                     | Final compiled dataset used for training, validation, and testing                                           |
| hyper-parameter optimization/Bio_hyperparams.ipynb | Fine-tuned Bio+ClinicalBERT model with grid search hyper-parameter optimization, chosen as final FusBERT model |
| model_files/all_BERT_models.ipynb          | All models used for binary classification                                                                     |
| handover/export_excel.ipynb                 | Python notebook to use FusBERT to make predictions on FUS-relevancy of articles given Excel input             | 
| handover/Progress_Presentation.pptx         | Presentation given to sponsors after fine-tuning BERT models on final compiled dataset                       | 
| handover/1pm_leveraging_nlp.pptx            | Final project presentation given at SIEDS conference                                                                | 
| handover/FUSFCapstone24_Updates.pdf         | Presentation with sponsor updates. Slides 11-17 discuss the final transition meeting on 5/8/24, covering model demo, limitations, future work, and next steps.|

## Paper 
R. Panagides, S. Jung, S. Fu, A. Singh and R. E. Muttikkal, "Enhancing Focused Ultrasound Literature Review Through Natural Language Processing-Driven Text Classification," 2024 Systems and Information Engineering Design Symposium (SIEDS), Charlottesville, VA, USA, 2024, pp. 409-414, doi: 10.1109/SIEDS61124.2024.10534719.

## FusBERT Hugging Face
https://huggingface.co/rpanagides/fusBERT/tree/main 


