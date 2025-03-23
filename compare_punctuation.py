import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
import assemblyai as aai
from dotenv import load_dotenv
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel

# Load environment variables
load_dotenv()

# Configure AssemblyAI
api_key = os.getenv('ASSEMBLYAI_API_KEY')
if not api_key:
    raise ValueError("Please set the ASSEMBLYAI_API_KEY environment variable")
aai.settings.api_key = api_key

class PunctuationComparison:
    def __init__(self):
        self.transcriber = aai.Transcriber()
        self.punctuation_model = PunctuationModel()
        self.results_dir = Path("punctuation_comparison_results")
        self.results_dir.mkdir(exist_ok=True)

    def transcribe_with_assemblyai(self, file_path: str, punctuate: bool = True) -> Dict:
        """
        Transcribe audio using AssemblyAI with or without punctuation
        """
        start_time = time.time()
        
        try:
            print(f"Starting AssemblyAI transcription of {file_path}...")
            
            config = aai.TranscriptionConfig(
                language_detection=True,
                punctuate=punctuate,
                format_text=True
            )
            
            transcript = self.transcriber.transcribe(file_path, config=config)
            processing_time = time.time() - start_time
            
            if transcript is None or transcript.status == aai.TranscriptStatus.error:
                return {
                    "status": "error",
                    "error": transcript.error if transcript else "Transcription returned None",
                    "processing_time": processing_time
                }
            
            return {
                "status": "success",
                "processing_time": processing_time,
                "text": transcript.text,
                "language": getattr(transcript, 'language_code', 'unknown'),
                "word_count": len(transcript.words) if hasattr(transcript, 'words') else 0
            }
            
        except Exception as e:
            print(f"AssemblyAI transcription error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def process_with_deepmultilingual(self, text: str) -> Dict:
        """
        Process text using DeepMultilingual Punctuation
        """
        start_time = time.time()
        
        try:
            print("Processing with DeepMultilingual Punctuation...")
            punctuated_text = self.punctuation_model.restore_punctuation(text)
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "processing_time": processing_time,
                "text": punctuated_text,
                "word_count": len(text.split())
            }
            
        except Exception as e:
            print(f"DeepMultilingual processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def save_results(self, results: Dict, filename: str):
        """
        Save comparison results to a JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{filename}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return output_file

    def compare_texts(self, original: str, assemblyai: str, deepmultilingual: str) -> Dict:
        """
        Compare the different versions of the text
        """
        return {
            "original_word_count": len(original.split()),
            "assemblyai_word_count": len(assemblyai.split()),
            "deepmultilingual_word_count": len(deepmultilingual.split()),
            "assemblyai_punctuation_marks": sum(1 for c in assemblyai if c in '.,!?;:'),
            "deepmultilingual_punctuation_marks": sum(1 for c in deepmultilingual if c in '.,!?;:')
        }

def main():
    analyzer = PunctuationComparison()
    
    # Test files directory
    test_files_dir = Path("test_audio")
    if not test_files_dir.exists():
        print(f"Please create a 'test_audio' directory and add your audio files there.")
        return
    
    # Process each audio file in the test directory
    for audio_file in test_files_dir.glob("*"):
        if audio_file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.ogg']:
            print(f"\nProcessing {audio_file.name}...")
            
            # Get unpunctuated text from AssemblyAI
            unpunctuated = analyzer.transcribe_with_assemblyai(str(audio_file), punctuate=False)
            if unpunctuated["status"] == "error":
                print(f"Error in AssemblyAI transcription: {unpunctuated['error']}")
                continue
            
            # Get punctuated text from AssemblyAI
            assemblyai_punctuated = analyzer.transcribe_with_assemblyai(str(audio_file), punctuate=True)
            if assemblyai_punctuated["status"] == "error":
                print(f"Error in AssemblyAI punctuation: {assemblyai_punctuated['error']}")
                continue
            
            # Process with DeepMultilingual
            deepmultilingual = analyzer.process_with_deepmultilingual(unpunctuated["text"])
            if deepmultilingual["status"] == "error":
                print(f"Error in DeepMultilingual processing: {deepmultilingual['error']}")
                continue
            
            # Compare results
            comparison = analyzer.compare_texts(
                unpunctuated["text"],
                assemblyai_punctuated["text"],
                deepmultilingual["text"]
            )
            
            # Prepare results
            results = {
                "file_name": audio_file.name,
                "language": unpunctuated["language"],
                "processing_times": {
                    "assemblyai_transcription": unpunctuated["processing_time"],
                    "assemblyai_punctuation": assemblyai_punctuated["processing_time"],
                    "deepmultilingual": deepmultilingual["processing_time"]
                },
                "comparison": comparison,
                "texts": {
                    "unpunctuated": unpunctuated["text"],
                    "assemblyai_punctuated": assemblyai_punctuated["text"],
                    "deepmultilingual_punctuated": deepmultilingual["text"]
                }
            }
            
            # Save results
            output_file = analyzer.save_results(results, audio_file.stem)
            print(f"\nResults saved to: {output_file}")
            
            # Display comparison
            print("\nComparison Results:")
            print(f"Language: {results['language']}")
            print(f"\nProcessing Times:")
            print(f"AssemblyAI Transcription: {results['processing_times']['assemblyai_transcription']:.2f} seconds")
            print(f"AssemblyAI Punctuation: {results['processing_times']['assemblyai_punctuation']:.2f} seconds")
            print(f"DeepMultilingual Processing: {results['processing_times']['deepmultilingual']:.2f} seconds")
            
            print(f"\nText Statistics:")
            print(f"Original Word Count: {comparison['original_word_count']}")
            print(f"AssemblyAI Punctuation Marks: {comparison['assemblyai_punctuation_marks']}")
            print(f"DeepMultilingual Punctuation Marks: {comparison['deepmultilingual_punctuation_marks']}")
            
            print("\nSample of texts (first 200 characters):")
            print("Unpunctuated:")
            print(results['texts']['unpunctuated'][:200] + "...")
            print("\nAssemblyAI Punctuated:")
            print(results['texts']['assemblyai_punctuated'][:200] + "...")
            print("\nDeepMultilingual Punctuated:")
            print(results['texts']['deepmultilingual_punctuated'][:200] + "...")

if __name__ == "__main__":
    main() 