import streamlit as st

from yt_dlp import YoutubeDL

def download_audio_from_url(url):
    videoinfo = YoutubeDL().extract_info(url=url, download=False)
    length = videoinfo['duration']
    filename = f"./audio/youtube/{videoinfo['id']}.mp3"
    options = {
        'format': 'bestaudio/best',
        'keepvideo': False,
        'outtmpl': filename,
    }
    with YoutubeDL(options) as ydl:
        ydl.download([videoinfo['webpage_url']])
    return filename, length


from transformers import pipeline
def transcribe_audio(filename):
    model = "facebook/wav2vec2-large-960h-lv60-self" #speech to text

    #speech to text
    pipe = pipeline(model = model)
    text = pipe(filename, chunk_length_s=10)

    #save text
    text_file = open("transcript.txt", "w")
    n = text_file.write(text["text"])
    text_file.close()

    #read article
    transcript = open("transcript.txt", "r").read()
    print(len(transcript.split()))
    transcript
    from hugchat import hugchat
    from hugchat.login import Login

    if transcript!="":

            huggingface_username = ''
            huggingface_pwd = ''
            # Log in to huggingface and grant authorization to huggingchat
            sign = Login(huggingface_username,huggingface_pwd)
            cookies = sign.login()

            # Save cookies to the local directory
            cookie_path_dir = "./cookies_snapshot"
            sign.saveCookiesToDir(cookie_path_dir)

            # Create a ChatBot
            chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

            # Extract the summary from the response
            query_result = chatbot.query("Summarize in 5-10 lines: "+ transcript)

            # Print the summary
            print("Summary:")
            print(query_result)
            return query_result


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
        audio_file, length = download_audio_from_url(url)
        st.write("Transcribing audio file...")
        transcript = transcribe_audio(audio_file)
        st.write("Summarizing transcript...")

        status.update(label="Finished", state="complete")

    # Play Audio
    st.divider()
    st.audio(audio_file, format='audio/mp3')

    # Show Summary
    st.subheader("Summary:", anchor=False)
    st.write(transcript)
