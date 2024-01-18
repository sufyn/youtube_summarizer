%%writefile mistral.py

import streamlit as st
import os

import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_4bit = AutoModelForCausalLM.from_pretrained( model_id, device_map="auto",quantization_config=quantization_config, )
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=500,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
llm = HuggingFacePipeline(pipeline=pipeline)


import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma


from youtube_transcript_api import YouTubeTranscriptApi

def download(url):
      youtube_video = url
      video_id = youtube_video.split("=")[1]

      YouTubeTranscriptApi.get_transcript(video_id)
      transcript = YouTubeTranscriptApi.get_transcript(video_id)
      with open(os.path.join('original_text' + '.txt'), 'w') as f:
          for line in transcript:
              f.write(line['text'] + '\n')
      st.write(transcript)

      return transcript



def summarize(transcript):
          from langchain.schema.document import Document
          documents = [Document(page_content=mna_news, metadata={"source": "local"})]
          #######################
          text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
          all_splits = text_splitter.split_documents(documents)
          model_name = "sentence-transformers/all-mpnet-base-v2"
          model_kwargs = {"device": "cuda"}
          embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
          #######################
          vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
          #######################
          retriever = vectordb.as_retriever()
          #######################
          qa = RetrievalQA.from_chain_type(
              llm=llm,
              chain_type="stuff",
              retriever=retriever,
              verbose=True
          )

          
          query = 'Summarize the text in 2-3 paragraphs'
          result = qa.run(query)
          print("\nResult: ", result)

          return result





# Set page title
st.set_page_config(page_title="YouTube Video Summarization", page_icon="📜", layout="wide")

# Set title
st.title("YouTube Video Summarization", anchor=False)
st.header("Summarize YouTube videos with AI", anchor=False)


# Input URL
st.divider()
url = st.text_input("Enter YouTube URL", value="")

# Download audio
st.divider()
if url:
    with st.status("Processing...", state="running", expanded=True) as status:
        st.write("Downloading audio file from YouTube...")
        transcript = download(url)
        st.write("Summarizing transcript...")
        summary = summarize(transcript)
        status.update(label="Finished", state="complete")
    # Show Summary
    st.subheader("Summary:", anchor=False)
    st.write(summary)
