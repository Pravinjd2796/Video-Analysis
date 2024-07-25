# Video Sentiment Analysis

## Overview

This Streamlit application allows users to upload a video file, extract audio from it, transcribe the audio to text using Whisper, and perform sentiment analysis on the transcribed text. The results, including the transcript and sentiment analysis, are displayed on the web interface.

## Features

- **Upload a Video File**: Supports various formats including MP4, MKV, AVI, and MOV.
- **Audio Extraction**: Extracts audio from the uploaded video.
- **Speech-to-Text Transcription**: Uses Whisper to convert audio to text.
- **Sentiment Analysis**: Analyzes the sentiment of the transcribed text using the Transformers library.
- **Display Results**: Shows the transcript with sentiment analysis results on the web interface.

## Sentiment Classification

The sentiment analysis model classifies text into three categories: "positive", "negative", and "neutral". The classification is based on the following thresholds:

- **Positive Sentiment**: If the model's output label is "POSITIVE" and the score is greater than or equal to 0.6.
- **Negative Sentiment**: If the model's output label is "NEGATIVE" and the score is greater than or equal to 0.6.
- **Neutral Sentiment**: If the model's output score for either positive or negative sentiment is below 0.6.

These thresholds are adjustable and can be modified in the code based on the desired sensitivity of the sentiment classification.

## Visualization

The results of the sentiment analysis are visualized using Plotly:

- **Sentiment Scores Over Time**: A line chart that shows sentiment scores over the duration of the video. Hovering over the line will display the sentiment score, the corresponding text segment, and the start and end times.
- **Sentiment Distribution**: A bar chart that shows the count of each sentiment category ("positive", "negative", and "neutral").

## Requirements

To run this application, you need to have the following Python packages installed:

- `streamlit`
- `moviepy`
- `whisper` (from OpenAI)
- `transformers`
- `torch`
- `numpy`

You can install these packages using pip:

```bash
pip install streamlit moviepy git+https://github.com/openai/whisper.git transformers torch numpy
