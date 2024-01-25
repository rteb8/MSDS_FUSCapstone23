# Creates an "application" to process individual PDF files. Either accessible locally by running the program,
# or at https://huggingface.co/spaces/Gators123/fusf_pdf_2023

# 1. Enter open API key at the top bar
# 2. Select a pdf file to classify
# 3. Additional questions about the pdf can be asked to the built-in chat bot

import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv
from PIL import Image
import fitz
from joblib import load
import gradio as gr
# _________________________________________________________________

# Global variables
COUNT, N = 0, 0
chat_history = []
chain = ''

# API Textboxes
enable_box = gr.Textbox.update(value=None, placeholder='Upload your OpenAI API key', interactive=True)
disable_box = gr.Textbox.update(value='OpenAI API key is Set', interactive=False)

# Function to set the API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box

# Function to enable the API key input box
def enable_api_box():
    return enable_box


# Function to add text to the chat history
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

# Function to process the PDF file and create a conversation chain
def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key')

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    
    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3), 
                                   retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                   return_source_documents=True)
    return chain



# Function to generate a response based on the chat history and query
def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain
    
    if not btn:
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:
        chain = process_file(btn)
        COUNT += 1
    
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    N = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''


# Function to render a specific page of a PDF file as an image
def render_file(pdf_file):
    global N
    doc = fitz.open(pdf_file.name)
    page = doc[N]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

#____________________________________________________________________

# Returns text from the pdf file
def read_string_text_from_file(pdf_file):

    if btn:
        loader = PyPDFLoader(pdf_file.name)

        doc = loader.load_and_split()
        
        stringtxt = str(doc)


        # Convert into summary from string to list in order to remove '\n' from text
        mylist = []
        final_string_no_lines = '' # This is the final summary that is printed
      
        # Adds all the characters to the list
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

# ___________________________________________________________

# Classifications using GPT API
def other_info(pdf_file):

    if btn:
        loader = PyPDFLoader(pdf_file.name)

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

    load_dotenv()

    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key')

    chat = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])


    ml_type_messages = [
        SystemMessage(content='''Classify the article into Supervised Machine Learning, Unsupervised Machine Learning, Both, or None. Surround the answer in brackets. On a new line, write a short blurb justifying why, no longer than 5 sentences:
                       Do not include brackets around your answer.'''),
        HumanMessage(content=final_string_no_lines[0:4000])
    ]


    treatment_cycle_messages = [
        SystemMessage(content='''Classify the article in one of the following treatment cycles. On a new line, write a short blurb justifying why:
                       [Treatment Planning, Treatment Monitoring and Results Analysis, Patient Selection, Clinical Decision Support]'''),
        HumanMessage(content=final_string_no_lines[0:4000])
    ]


    medical_indication_messages = [
        SystemMessage(content='''Classify the article in one of the following medical indications. On a new line, write a short blurb justifying why.:
                       [Cardiovascular, Emerging Indications, Gynelogical, Neurological (blood-brain-barrier opening), Neurosurgery, Oncological, Urological (prostate), Veterinary, Other]'''),
        HumanMessage(content=final_string_no_lines[0:4000])
    ]

    key_word_messages = [
        SystemMessage(content='''Pick some of the keywords, and ONLY KEY WORDS LISTED BELOW that the article encompasses. Provide in a numbered list:
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


    summary = [
        SystemMessage(content='''Write a summary of the article.'''),
        HumanMessage(content=final_string_no_lines[0:4000])
    ]


    # ML Type, Treatment Cycle, Medical Indication, Keywords, Summary
    return chat(ml_type_messages).content,chat(treatment_cycle_messages).content, chat(medical_indication_messages).content, chat(key_word_messages).content, chat(summary).content


# Fus/Non-fus Model
def fus_model(pdf_file):

    # Loads FUS model with Joblib
    fus_model = load('fus_model.joblib')

    prediction = fus_model.predict_proba([read_string_text_from_file(pdf_file)])

    percentage_pos = (prediction[0][0])*100
    percentage_neg = (prediction[0][1])*100
    
    # Returns probability that it is Fus and probability that it is Non-fus
    return 'Focused Ultrasound Related: ' + str((round(percentage_pos,1)))+'%'+'\n'+'Non-Fus: '+ str((round(percentage_neg,1)))+'%'


# Setting up Gradio application layout
with gr.Blocks() as demo:
    

    # Top Row, the place for submitting or changing the API Key
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(
                    placeholder='Paste OpenAI API key and press Enter',
                    show_label=False,
                    interactive=True
                )
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')

    
    # Second row, the place for displaying the FUS/Non-FUS and ML type probabilities
    with gr.Column():
        with gr.Row():
            fus = gr.Textbox(label='FUS/Non-FUS',interactive=False)
            ml = gr.Textbox(label='ML Type',interactive=False)

        
        with gr.Row():
            treatment_cycle = gr.Textbox(label='Treatment Cycle',interactive=False)
            medical_indication = gr.Textbox(label='Medical Indication',interactive=False)

        
        # Contains the summary generation box
        with gr.Row():
            keyword = gr.Textbox(label='Keywords',interactive=False)
            summary = gr.Textbox(label='Summary',interactive=False)


        # Contains Chatbot and PDF displayer
        with gr.Row():       
            chatbot = gr.Chatbot(value=[], elem_id='chatbot',height=780)
            
            show_img = gr.Image(label='Upload PDF', tool='select',height=780)

    
    # Text box for user to input questions into the chatbot
    with gr.Row():
        with gr.Column(scale=0.5):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press submit",
                container=False)

        # Button for uploading PDF
        with gr.Column(scale=0.5):
            btn = gr.UploadButton("📁 Upload a PDF", file_types=[".pdf"])


    # Example prompts that can be entered into the chatbot
    gr.Examples(

        #Add or customize example prompts here
        examples=[['What are five important keywords?'],['What were the conclusions/results of this study?'],['Who were the authors of this study?']],
        inputs = [txt]
        )


    with gr.Row():

        # Submit button
        with gr.Column(scale=0.5):
            submit_btn = gr.Button('Submit')
        

# __________________________________________________________________________________

    # Set up event handlers

    # Event handler for submitting the OpenAI API key
    api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])

    # Event handler for changing the API key
    change_api_key.click(fn=enable_api_box, outputs=[api_key])

    # Event handler for uploading a PDF
    def on_upload(btn):
        show_img.value = render_file(btn)
        
        fus.value = fus_model(btn) # Fus/Non Fus
        
        ml.value = other_info(btn)[0] # ML Type

        treatment_cycle.value = other_info(btn)[1] # Treatment Cycle

        medical_indication.value = other_info(btn)[2] # Medical Indication

        keyword.value = other_info(btn)[3] # Keywords

        summary.value = other_info(btn)[4] # Summary


        return show_img.value, fus.value, ml.value, treatment_cycle.value, medical_indication.value, keyword.value, summary.value

        
    btn.upload(on_upload, inputs=[btn], outputs=[show_img, fus, ml, treatment_cycle, medical_indication, keyword, summary])


    # Event handler for submitting text and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )

# Launches the Gradio application
demo.queue()
if __name__ == "__main__":
   
   demo.launch()