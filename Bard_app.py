import streamlit as st

from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
def download(url):
      youtube_video = url
      video_id = youtube_video.split("=")[1]

      YouTubeTranscriptApi.get_transcript(video_id)
      transcript = YouTubeTranscriptApi.get_transcript(video_id)

      summarizer = pipeline('summarization')
      result = ""
      for i in transcript:
          result += ' ' + i['text']
      st.write(result)

      num_iters = int(len(result)/1000)
      summarized_text = []
      for i in range(0, num_iters + 1):
        start = 0
        start = i * 1000
        end = (i + 1) * 1000
        print("input text \n" + result[start:end])
        out = summarizer(result[start:end])
        out = out[0]
        out = out['summary_text']
        print("Summarized text\n"+out)
        summarized_text.append(out)

      return str(summarized_text)


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
        summary = download(url)
        st.write("Summarizing transcript...")
        status.update(label="Finished", state="complete")
    # Show Summary
    st.subheader("Summary:", anchor=False)
    st.write(summary)
