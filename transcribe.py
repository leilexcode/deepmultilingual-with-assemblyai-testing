import os
import assemblyai as aai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv('ASSEMBLYAI_API_KEY')
if not api_key:
    raise ValueError("Please set the ASSEMBLYAI_API_KEY environment variable")

# Configure AssemblyAI
aai.settings.api_key = api_key

def transcribe_audio(file_url):
    """
    Transcribe audio from a URL or local file path
    """
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_url)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"Error: {transcript.error}")
            return None
        else:
            return transcript.text

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    FILE_URL = "https://assembly.ai/wildfires.mp3"  # Replace with your audio file URL
    
    # You can also transcribe a local file by passing in a file path
    # FILE_URL = './path/to/file.mp3'
    
    result = transcribe_audio(FILE_URL)
    if result:
        print("\nTranscription:")
        print(result) 