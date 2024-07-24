# Video Sentiment Analysis

## Overview

This Streamlit application allows users to upload a video file, extract audio from it, transcribe the audio to text using Whisper, and perform sentiment analysis on the transcribed text. The results, including the transcript and sentiment analysis, are displayed on the web interface.

## Features

- **Upload a Video File**: Supports various formats including MP4, MKV, AVI, and MOV.
- **Audio Extraction**: Extracts audio from the uploaded video.
- **Speech-to-Text Transcription**: Uses Whisper to convert audio to text.
- **Sentiment Analysis**: Analyzes the sentiment of the transcribed text using the Transformers library.
- **Display Results**: Shows the transcript with sentiment analysis results on the web interface.

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
