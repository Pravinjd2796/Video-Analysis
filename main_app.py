
# import os
# import streamlit as st
# import moviepy.editor as mp
# import whisper
# from transformers.pipelines import pipeline

# # Set up sentiment analysis pipeline
# sentiment_pipeline = pipeline("sentiment-analysis")

# # Function to extract audio from video
# def extract_audio(video_path: str, audio_path: str):
#     video = mp.VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_path)

# # Function to transcribe audio using Whisper
# def transcribe_audio(audio_path: str):
#     model = whisper.load_model("base")  # You can choose other models like "small", "medium", "large" depending on your need.
#     result = model.transcribe(audio_path)
#     return result

# # Function to format the transcript
# def format_transcript(transcription_result):
#     segments = transcription_result['segments']
#     transcript = []

#     for segment in segments:
#         start = segment['start']
#         end = segment['end']
#         text = segment['text']
#         transcript.append({
#             'start': start,
#             'end': end,
#             'text': text
#         })

#     return transcript

# # Function to analyze sentiment
# def analyze_sentiment(text: str):
#     sentiment = sentiment_pipeline(text)
#     return sentiment[0]['label'], sentiment[0]['score']

# # Function to analyze transcript sentiment
# def analyze_transcript_sentiment(transcript):
#     results = []
#     for entry in transcript:
#         text = entry['text']
#         sentiment_label, sentiment_score = analyze_sentiment(text)
#         results.append({
#             'start': entry['start'],
#             'end': entry['end'],
#             'text': text,
#             'sentiment': sentiment_label,
#             'score': sentiment_score
#         })
#     return results

# # Streamlit app
# st.title("Video Sentiment Analysis")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])

# if uploaded_file is not None:
#     # Create a directory to save uploaded files if it doesn't exist
#     upload_dir = "uploaded_videos"
#     os.makedirs(upload_dir, exist_ok=True)

#     video_path = os.path.join(upload_dir, uploaded_file.name)

#     # Save the uploaded video file
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.video(video_path)

#     # Extract audio from video
#     audio_path = os.path.join(upload_dir, "extracted_audio.wav")
#     extract_audio(video_path, audio_path)

#     # Transcribe audio
#     st.write("Transcribing audio...")
#     transcription_result = transcribe_audio(audio_path)
#     transcript = format_transcript(transcription_result)

#     # Display transcript
#     st.write("Transcript:")
#     for entry in transcript:
#         st.write(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['text']}")

#     # Analyze sentiment
#     st.write("Analyzing sentiment...")
#     sentiment_results = analyze_transcript_sentiment(transcript)

#     # Display sentiment analysis results
#     st.write("Sentiment Analysis Results:")
#     for result in sentiment_results:
#         st.write(f"[{result['start']:.2f} - {result['end']:.2f}] {result['text']} - Sentiment: {result['sentiment']} (Score: {result['score']:.2f})")

# import os
# import streamlit as st
# import moviepy.editor as mp
# import whisper
# from transformers import pipeline
# import plotly.express as px
# import pandas as pd

# # Set up sentiment analysis pipeline
# sentiment_pipeline = pipeline("sentiment-analysis")

# # Function to extract audio from video
# def extract_audio(video_path: str, audio_path: str):
#     video = mp.VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_path)

# # Function to transcribe audio using Whisper
# def transcribe_audio(audio_path: str):
#     model = whisper.load_model("base")  # You can choose other models like "small", "medium", "large" depending on your need.
#     result = model.transcribe(audio_path)
#     return result

# # Function to format the transcript
# def format_transcript(transcription_result):
#     segments = transcription_result['segments']
#     transcript = []

#     for segment in segments:
#         start = segment['start']
#         end = segment['end']
#         text = segment['text']
#         transcript.append({
#             'start': start,
#             'end': end,
#             'text': text
#         })

#     return transcript

# # Function to analyze sentiment
# def analyze_sentiment(text: str):
#     sentiment = sentiment_pipeline(text)
#     return sentiment[0]['label'], sentiment[0]['score']

# # Function to analyze transcript sentiment
# def analyze_transcript_sentiment(transcript):
#     results = []
#     for entry in transcript:
#         text = entry['text']
#         sentiment_label, sentiment_score = analyze_sentiment(text)
#         results.append({
#             'start': entry['start'],
#             'end': entry['end'],
#             'text': text,
#             'sentiment': sentiment_label,
#             'score': sentiment_score
#         })
#     return results

# # Streamlit app
# st.title("Video Sentiment Analysis")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])

# if uploaded_file is not None:
#     # Create a directory to save uploaded files if it doesn't exist
#     upload_dir = "uploaded_videos"
#     os.makedirs(upload_dir, exist_ok=True)

#     video_path = os.path.join(upload_dir, uploaded_file.name)

#     # Save the uploaded video file
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.video(video_path)

#     # Extract audio from video
#     audio_path = os.path.join(upload_dir, "extracted_audio.wav")
#     extract_audio(video_path, audio_path)

#     # Transcribe audio
#     st.write("Transcribing audio...")
#     transcription_result = transcribe_audio(audio_path)
#     transcript = format_transcript(transcription_result)

#     # Display transcript
#     st.write("Transcript:")
#     for entry in transcript:
#         st.write(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['text']}")

#     # Analyze sentiment
#     st.write("Analyzing sentiment...")
#     sentiment_results = analyze_transcript_sentiment(transcript)

#     # Prepare data for visualization
#     df = pd.DataFrame(sentiment_results)

#     # Display sentiment analysis results
#     st.write("Sentiment Analysis Results:")
#     for result in sentiment_results:
#         st.write(f"[{result['start']:.2f} - {result['end']:.2f}] {result['text']} - Sentiment: {result['sentiment']} (Score: {result['score']:.2f})")

#     # Visualization
#     st.write("Sentiment Analysis Visualization:")

#     # Convert sentiment labels to numeric values for plotting
#     df['sentiment_value'] = df['sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

#     # Create a line chart for sentiment scores over time
#     fig = px.line(df, x='start', y='score', color='sentiment', title='Sentiment Scores Over Time', labels={
#         'start': 'Time (seconds)',
#         'score': 'Sentiment Score'
#     })
#     st.plotly_chart(fig)

#     # Create a bar chart for sentiment distribution
#     sentiment_counts = df['sentiment'].value_counts().reset_index()
#     sentiment_counts.columns = ['sentiment', 'count']
#     fig_bar = px.bar(sentiment_counts, x='sentiment', y='count', title='Sentiment Distribution', labels={
#         'sentiment': 'Sentiment',
#         'count': 'Count'
#     })
#     st.plotly_chart(fig_bar)

# import os
# import streamlit as st
# import moviepy.editor as mp
# import whisper
# from transformers import pipeline
# import plotly.express as px
# import pandas as pd

# # Set up sentiment analysis pipeline
# sentiment_pipeline = pipeline("sentiment-analysis")

# # Function to extract audio from video
# def extract_audio(video_path: str, audio_path: str):
#     video = mp.VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_path)

# # Function to transcribe audio using Whisper
# def transcribe_audio(audio_path: str):
#     model = whisper.load_model("base")  # You can choose other models like "small", "medium", "large" depending on your need.
#     result = model.transcribe(audio_path)
#     return result

# # Function to format the transcript
# def format_transcript(transcription_result):
#     segments = transcription_result['segments']
#     transcript = []

#     for segment in segments:
#         start = segment['start']
#         end = segment['end']
#         text = segment['text']
#         transcript.append({
#             'start': start,
#             'end': end,
#             'text': text
#         })

#     return transcript

# # Function to analyze sentiment
# def analyze_sentiment(text: str):
#     sentiment = sentiment_pipeline(text)
#     return sentiment[0]['label'], sentiment[0]['score']

# # Function to analyze transcript sentiment
# def analyze_transcript_sentiment(transcript):
#     results = []
#     for entry in transcript:
#         text = entry['text']
#         sentiment_label, sentiment_score = analyze_sentiment(text)
#         results.append({
#             'start': entry['start'],
#             'end': entry['end'],
#             'text': text,
#             'sentiment': sentiment_label,
#             'score': sentiment_score
#         })
#     return results

# # Streamlit app
# st.title("Video Sentiment Analysis")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])

# if uploaded_file is not None:
#     # Create a directory to save uploaded files if it doesn't exist
#     upload_dir = "uploaded_videos"
#     os.makedirs(upload_dir, exist_ok=True)

#     video_path = os.path.join(upload_dir, uploaded_file.name)

#     # Save the uploaded video file
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.video(video_path)

#     # Extract audio from video
#     audio_path = os.path.join(upload_dir, "extracted_audio.wav")
#     extract_audio(video_path, audio_path)

#     # Transcribe audio
#     st.write("Transcribing audio...")
#     transcription_result = transcribe_audio(audio_path)
#     transcript = format_transcript(transcription_result)

#     # Display transcript
#     st.write("Transcript:")
#     for entry in transcript:
#         st.write(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['text']}")

#     # Analyze sentiment
#     st.write("Analyzing sentiment...")
#     sentiment_results = analyze_transcript_sentiment(transcript)

#     # Prepare data for visualization
#     df = pd.DataFrame(sentiment_results)

#     # Display sentiment analysis results
#     st.write("Sentiment Analysis Results:")
#     for result in sentiment_results:
#         st.write(f"[{result['start']:.2f} - {result['end']:.2f}] {result['text']} - Sentiment: {result['sentiment']} (Score: {result['score']:.2f})")

#     # Visualization
#     st.write("Sentiment Analysis Visualization:")

#     # Create a line chart for sentiment scores over time with tooltips
#     fig = px.line(df, x='start', y='score', color='sentiment', title='Sentiment Scores Over Time', labels={
#         'start': 'Time (seconds)',
#         'score': 'Sentiment Score'
#     }, hover_data={'text': True, 'start': True, 'end': True, 'score': True})
#     st.plotly_chart(fig)

#     # Create a bar chart for sentiment distribution
#     sentiment_counts = df['sentiment'].value_counts().reset_index()
#     sentiment_counts.columns = ['sentiment', 'count']
#     fig_bar = px.bar(sentiment_counts, x='sentiment', y='count', title='Sentiment Distribution', labels={
#         'sentiment': 'Sentiment',
#         'count': 'Count'
#     })
#     st.plotly_chart(fig_bar)
import os
import streamlit as st
import moviepy.editor as mp
import whisper
from transformers import pipeline
import plotly.express as px
import pandas as pd

# Set up sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to extract audio from video
def extract_audio(video_path: str, audio_path: str):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path: str):
    model = whisper.load_model("base")  # You can choose other models like "small", "medium", "large" depending on your need.
    result = model.transcribe(audio_path)
    return result

# Function to format the transcript
def format_transcript(transcription_result):
    segments = transcription_result['segments']
    transcript = []

    for segment in segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']
        transcript.append({
            'start': start,
            'end': end,
            'text': text
        })

    return transcript

# Function to analyze sentiment
def analyze_sentiment(text: str):
    sentiment = sentiment_pipeline(text)[0]
    label = sentiment['label']
    score = sentiment['score']
    
    # Define thresholds for neutral sentiment
    if label == "NEGATIVE" and score < 0.6:
        label = "NEUTRAL"
    elif label == "POSITIVE" and score < 0.6:
        label = "NEUTRAL"

    return label, score

# Function to analyze transcript sentiment
def analyze_transcript_sentiment(transcript):
    results = []
    for entry in transcript:
        text = entry['text']
        sentiment_label, sentiment_score = analyze_sentiment(text)
        results.append({
            'start': entry['start'],
            'end': entry['end'],
            'text': text,
            'sentiment': sentiment_label,
            'score': sentiment_score
        })
    return results

# Streamlit app
st.title("Video Sentiment Analysis")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])

if uploaded_file is not None:
    # Create a directory to save uploaded files if it doesn't exist
    upload_dir = "uploaded_videos"
    os.makedirs(upload_dir, exist_ok=True)

    video_path = os.path.join(upload_dir, uploaded_file.name)

    # Save the uploaded video file
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)

    # Extract audio from video
    audio_path = os.path.join(upload_dir, "extracted_audio.wav")
    extract_audio(video_path, audio_path)

    # Transcribe audio
    st.write("Transcribing audio...")
    transcription_result = transcribe_audio(audio_path)
    transcript = format_transcript(transcription_result)

    # Display transcript
    st.write("Transcript:")
    for entry in transcript:
        st.write(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['text']}")

    # Analyze sentiment
    st.write("Analyzing sentiment...")
    sentiment_results = analyze_transcript_sentiment(transcript)

    # Prepare data for visualization
    df = pd.DataFrame(sentiment_results)

    # Display sentiment analysis results
    st.write("Sentiment Analysis Results:")
    for result in sentiment_results:
        st.write(f"[{result['start']:.2f} - {result['end']:.2f}] {result['text']} - Sentiment: {result['sentiment']} (Score: {result['score']:.2f})")

    # Visualization
    st.write("Sentiment Analysis Visualization:")

    # Create a line chart for sentiment scores over time with tooltips
    fig = px.line(df, x='start', y='score', color='sentiment', title='Sentiment Scores Over Time', labels={
        'start': 'Time (seconds)',
        'score': 'Sentiment Score'
    }, hover_data={'text': True, 'start': True, 'end': True, 'score': True})
    st.plotly_chart(fig)

    # Create a bar chart for sentiment distribution
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_bar = px.bar(sentiment_counts, x='sentiment', y='count', title='Sentiment Distribution', labels={
        'sentiment': 'Sentiment',
        'count': 'Count'
    })
    st.plotly_chart(fig_bar)
