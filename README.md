# Audio Transcription with AssemblyAI

This project uses AssemblyAI to transcribe audio files from URLs or local files, with comprehensive performance analysis and accuracy metrics.

## Setup

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your AssemblyAI API key:

```
ASSEMBLYAI_API_KEY=your_api_key_here
```

## Usage

### Basic Transcription

Run the basic transcription script:

```bash
python transcribe.py
```

### Comprehensive Testing and Analysis

To analyze transcription performance, accuracy, and confidence levels:

1. Place your audio files in the `test_audio` directory
2. Run the analysis script:

```bash
python test_transcription.py
```

The analysis script will:

- Process each audio file in the `test_audio` directory
- Generate detailed performance metrics including:
  - Processing time
  - Total audio duration
  - Word count
  - Average confidence score
  - Utterance count and words per utterance
  - Processing speed (words/second)
  - Confidence distribution (high/medium/low)
- Save detailed results to JSON files in the `transcription_results` directory

## Features

- Transcribe audio from URLs or local files
- Comprehensive performance analysis
- Word-level confidence scoring
- Utterance-level analysis
- Processing time and speed metrics
- Detailed JSON output for further analysis
- Support for multiple audio formats (mp3, wav, m4a, ogg)
- Error handling and status checking
- Environment variable support for API key

## Directory Structure

- `test_audio/` - Place your audio files here for testing
- `transcription_results/` - Contains detailed JSON results of each transcription
- `transcribe.py` - Basic transcription script
- `test_transcription.py` - Comprehensive testing and analysis script

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```
