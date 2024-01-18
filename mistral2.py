import streamlit as st
import os

from langchain import PromptTemplate, LLMChain

from langchain.llms import CTransformers
config = {'max_new_tokens': 100, 'temperature': 0}
llm = CTransformers(model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config=config)


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
        template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
        Answer the question below from context below :
        {context}
        {question} [/INST] </s>
        """
        
        question_p = """Summarize the text in 2-3 paragraphs"""
        context_p = transcript
        prompt = PromptTemplate(template=template, input_variables=["question","context"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({"question":question_p,"context":context_p})
        
        return response





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
