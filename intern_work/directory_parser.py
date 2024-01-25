# Code takes in a provided directory containing files to be parsed and classified. Results returned in Excel sheet

# 1. Ask user for API Key (unless they already provided it in the code), and file directory
# 2. Finds .pdf files in directory, runs read_string_text_from_file to obtain text from each pdf
# 3. Runs that text through other_info and fus_model to classify the pdfs, and add that information to lists
# 4. Runs write_excel to write inforomation from the lists into excel

# Classifications are: fus/non-fus, ML type, treatment Cycle, medical indication, keywords

# Before running, ensure API Key is entered, change directory with pdf files and directory for excel sheet accordingly

import os
from datetime import date
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from joblib import load
import xlsxwriter
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv

# _________________________________________________________________

# Global variables
other_list = []
fus_list = []
ml_list = []


# Sets API Key
api_key = os.environ['OPENAI_API_KEY'] = 'ENTER API KEY HERE'

# Function to intake API key while running code (will not run if key is already assigned above)
def ask_api():
    api_key = (input("Please enter your API Key: "))
    os.environ['OPENAI_API_KEY'] = api_key


# Code to generate summary (no API)
def read_string_text_from_file(pdf_file):
    
    loader = PyPDFLoader(pdf_file)

    doc = loader.load_and_split()
        
    stringtxt = str(doc)


    # Had to convert into summary from string to list in order to remove '\n' from text
    mylist = []
    final_string_no_lines = '' # Final text that is returned

    for char in stringtxt:
        mylist.append(char)

    # Finds the indices where '\n' is present
    pop_index = []
    for word in range(0,len(mylist)):
        if mylist[word] =='\\' and mylist[word+1]=='n':
            pop_index.append(word)
            pop_index.append(word+1)

    # Replaces those indices with an empty space
    for word in pop_index:
        mylist[word] = ' '

    # Converts cleaned list back into string and returns
    for i in mylist:
        final_string_no_lines+=i

    return final_string_no_lines


# Uses API to classify pdf files
def other_info(pdf_file):

    loader = PyPDFLoader(pdf_file)

    doc = loader.load_and_split()
        
    stringtxt = str(doc)


    # Had to convert into summary from string to list in order to remove '\n' from text
    mylist = []
    final_string_no_lines = '' # Final text that is used for model

    for char in stringtxt:
        mylist.append(char)

    # Finds the indices where '\n' is present
    pop_index = []
    for word in range(0,len(mylist)):
        if mylist[word] =='\\' and mylist[word+1]=='n':
            pop_index.append(word)
            pop_index.append(word+1)

    # Replaces those indices with an empty space
    for word in pop_index:
        mylist[word] = ' '

    # Converts cleaned list back into string and returns
    for i in mylist:
        final_string_no_lines+=i

    load_dotenv()

    chat = ChatOpenAI(openai_api_key=api_key)

    ml_type_messages = [
        SystemMessage(content='''Classify the article into Supervised Machine Learning, Unsupervised Machine Learning, Both, or None, and include brackets around the answer.
                       On a new line, write a short blurb justifying why:'''),
        HumanMessage(content=final_string_no_lines[0:4000]) # Note will only use first 4000 characters as classifier, due to limit on API
    ]


    treatment_cycle_messages = [
        SystemMessage(content='''Classify the article in one of the following treatment cycles, including brackets around the answer: 
                       [Treatment Planning, Treatment Monitoring and Results Analysis, Patient Selection, Clinical Decision Support]. 
                       On a new line, write a short blurb justifying why:'''),
        HumanMessage(content=final_string_no_lines[0:4000])
    ]


    medical_indication_messages = [
        SystemMessage(content='''Classify the article in one of the following medical indications, including brackets around the answer: 
                       [Cardiovascular, Emerging Indications, Gynelogical, Neurological (blood-brain-barrier opening), Neurosurgery, Oncological, Urological (prostate), Veterinary, Other].
                       On a new line, write a short blurb justifying why.'''),
        HumanMessage(content=final_string_no_lines[0:4000])
    ]

    key_word_messages = [
        SystemMessage(content='''Pick some of the keywords, and ONLY KEY WORDS LISTED BELOW that you feel the article encompasses. Provide in a numbered list:
                       [Angular spectrum, Artificial intelligence, Artificial neural networks, Auto encoders, Bio-heat transfer, Cat Swarm Operation, Chaotic krill her algorithm (CKHA), CIVA HealthCare platform, Classification,
                        Coefficient based method, Computed tomography (CT), Computer architecture, Convolutional neural network (CNN), Decision trees, Deep CNN, Deep leaning, Diagnostic imaging,, Differential equation solver,
                        Encoder-decoder, Fourier transform, Functional mapping, Functional neurosurgery, FUS monitoring, Generative adversarial networks (GAN), Global convolutional networks, Harmonic motion imaging,
                        HIFU Artifact, Image filtering, Intelligent theranostics, Joint Mutual Information (JMI), K means clustering, Kapur entropy, K-nearest neighbor, Logistic regression, Magnetic resonance imaging (MRI),
                        Medical diagnostics, Metamodel, Multilayer Perception (MLP), Multistage neural network, Mutual Information Maximisation (MIM), Naive Bayes classifier, NDE, Neural network, Neuromodulation,
                        Numerical model, Partial dependence plots, Photon counting CT, Prediction, Preoperative prediction, Principal component analysis, Prognosis, Radiomics, Random forest, Rayleigh-Sommerfeld, Real-time lesion tracking,
                        Regression models (linear and logistic), Residual, Rule based decision tree method, Segmentation, Skull density ratio, Support vector classification (SVC) model, Support vector machines, SWOT, Temperature monitoring, Transfer learning,
                        Transformers, Ultrasonography, Ultrasound (US), U-net (CNN, Encoder, Decoder, Autoencoder), Unsupervised learning, VGG Net, Vision transformers (ViT), Wiener Filtering]. Remember to only use the keywords in the list above'''),
        HumanMessage(content=final_string_no_lines[0:4000])

    ]

    title = [
        SystemMessage(content='''What is the title of the article? Return nothing but the title'''),
        HumanMessage(content=final_string_no_lines[0:4000])
    ]
    

    return chat(ml_type_messages).content, chat(treatment_cycle_messages).content, chat(medical_indication_messages).content, chat(key_word_messages).content, chat(title).content



# FUS/Non-fus classification
def fus_model(pdf_file):

    # Loads FUS model with Joblib
    fus_model = load('fus_model.joblib')

    prediction = fus_model.predict_proba([read_string_text_from_file(pdf_file)])

    percentage_pos = (prediction[0][0])*100
    percentage_neg = (prediction[0][1])*100
    
    return 'Focused Ultrasound Related: ' + str((round(percentage_pos,1)))+'%'+'   '+'Non-Fus: '+ str((round(percentage_neg,1)))+'%'


# _____________________________________________________________________

# Where the result excel file will be located - CHANGE
results_dir = "C:\\Users\\fuzhe\\OneDrive\\Documents\\2023 Summer Intern NLP Project 81123\\fusf_pdf_2023\\result_excel_files\\"


# Function to write all the info to excel
def write_excel(fus,other_list):

    today = date.today()

    xlsx_file = results_dir + 'Directory_data_' + str(today) + '.xlsx'

    workbook = xlsxwriter.Workbook(xlsx_file)
    worksheet = workbook.add_worksheet('First Sheet')

    

    format_header = workbook.add_format({'font_color': 'blue', 'font_name': 'Arial', 'font_size': '10', 'valign': 'top'})
    wrap_format = workbook.add_format({'font_name': 'Arial', 'font_size': '10', 'valign': 'top', 'text_wrap': True})
    worksheet.set_row(0, None, format_header)


    worksheet.set_column(0, 0, 40)
    worksheet.set_column(1, 1, 30)
    worksheet.set_column(2, 2, 30)
    worksheet.set_column(3, 3, 50)
    worksheet.set_column(4, 4, 50)
    worksheet.set_column(5, 5, 30)

    worksheet.write(0,0,'Name')
    worksheet.write(0,1,'Fus / NonFus')
    worksheet.write(0,2,'ML Type')
    worksheet.write(0,3,'Treatment Cycle')
    worksheet.write(0,4,'Medical Indications')
    worksheet.write(0,5,'Keyword(s)')


    title_row_index=1
    for title in other_list:

        worksheet.set_row(title_row_index, 50)

        worksheet.write(title_row_index, 0, title[4], wrap_format)
        
        title_row_index+=1


    fus_row_index=1
    for fus in fus_list:

        worksheet.write(fus_row_index, 1 ,fus, wrap_format)

        fus_row_index+=1


    ml_row_index=1
    for ml in other_list:

        worksheet.write(ml_row_index, 2, ml[0], wrap_format)

        ml_row_index+=1


    cycle_row_index=1
    for cycle in other_list:

        worksheet.write(cycle_row_index, 3, cycle[1], wrap_format)

        cycle_row_index+=1


    indication_row_index=1
    for indication in other_list:

        worksheet.write(indication_row_index, 4, indication[2], wrap_format)

        indication_row_index+=1

    keywords_row_index=1
    for keyword in other_list:

        worksheet.write(keywords_row_index, 5, keyword[3], wrap_format)

        keywords_row_index+=1

    workbook.close()

# Function to intake directory, output all the pdf files in that directory
def get_pdf_paths(directory):
    pdf_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths

#___________________________________________________________________

# Prompts user to enter API key if one isn't already present
if 'OPENAI_API_KEY' not in os.environ:
    ask_api()
else:
    print('API Key Receieved...')


# Get pdf paths from the directory - CHANGE
pdf_paths = get_pdf_paths(r"C:\Users\fuzhe\OneDrive\Documents\2023 Summer Intern NLP Project 81123\PubMed_PDFs")

# Gives user chance to enter directory while running the program
user_input = input("Enter the directory path where your PDFs are located (if you specified path in code press the enter key): ")

if user_input != '':
    pdf_paths = get_pdf_paths(user_input)


# Displays the PDFs found in the provided directory
print("\nPDF files found in provided directory:")
for path in pdf_paths:
    print(path)


print('\nProcessing files and writing to excel sheet...')


for path in pdf_paths:
    other_list.append(other_info(path))
    fus_list.append(fus_model(path))


#Writes information to the Excel sheet
write_excel(fus_list,other_list)

print(f'\nSucessfully written to: {str(results_dir)}')