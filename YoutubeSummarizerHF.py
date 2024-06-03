#Hugging Face (HF)
import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re

model_path = "C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Model/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"

text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

def get_video_id(url):
    """
    Extracts the video ID from the given YouTube URL.
    """
    # Regular expression to match the video ID
    video_id_match = re.match(r'.*(v=|\/)([a-zA-Z0-9_-]{11}).*', url)
    if video_id_match:
        return video_id_match.groups()[-1]
    else:
        raise ValueError("Invalid YouTube URL")

def get_transcript(video_url):
    """
    Fetches the transcript of a YouTube video given its URL.
    """
    try:
        # Extract video ID from the URL
        video_id = get_video_id(video_url)
        
        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine the transcript text
        transcript_text = " ".join([item['text'] for item in transcript])


        Summarized_text = summary(transcript_text)
        return Summarized_text
    except Exception as e:
        return str(e)

# Example usage
# video_url = input("Enter YouTube video URL: ")
# transcript = get_transcript(video_url)
# print("Transcript:\n", transcript)

gr.close_all()

demo = gr.Interface(fn=get_transcript, inputs=[gr.Textbox(label="Input text to summarize",lines = 6)],outputs=[gr.Textbox(label="Summarized Text",lines = 4)],title="Project 1: Text summary",
                    description="""This is a simple text summarization model.""")
demo.launch()