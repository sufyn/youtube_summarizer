import streamlit as st
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

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



def load_documents():

    loader = DirectoryLoader('./', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    return embeddings

# Function to create vector store
def create_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

# Function to create LLMS model
def create_llms_model():
    llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", config={'max_new_tokens': 528, 'temperature': 0.1})
    return llm



def summarize():
    # loading of documents
    import io
    # documents = 'original_text.txt'
    documents = load_documents()

    # Split text into chunks
    text_chunks = split_text_into_chunks(documents)

    # Create embeddings
    embeddings = create_embeddings()

    # Create vector store
    vector_store = create_vector_store(text_chunks, embeddings)

    # Create LLMS model
    llm = create_llms_model()

    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    # Define chat function
    def conversation_chat(query):
          result = chain({"question": query})

          return result["answer"]


    user_input = "Summarize the text in 2-3 paragraphs"
    output = conversation_chat(user_input)
    print(output)
    return output





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
        summary = summarize()
        status.update(label="Finished", state="complete")
    # Show Summary
    st.subheader("Summary:", anchor=False)
    st.write(summary)
