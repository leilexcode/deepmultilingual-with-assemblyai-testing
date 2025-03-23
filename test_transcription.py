import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
import assemblyai as aai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure AssemblyAI
api_key = os.getenv('ASSEMBLYAI_API_KEY')
if not api_key:
    raise ValueError("Please set the ASSEMBLYAI_API_KEY environment variable")
aai.settings.api_key = api_key

class TranscriptionAnalyzer:
    def __init__(self):
        self.transcriber = aai.Transcriber()
        self.results_dir = Path("transcription_results")
        self.results_dir.mkdir(exist_ok=True)

    def transcribe_with_metrics(self, file_path: str, punctuate: bool = True) -> Dict:
        """
        Transcribe audio and collect performance metrics
        """
        start_time = time.time()
        
        try:
            print(f"Starting transcription of {file_path}...")
            
            # Configure transcription with language detection
            config = aai.TranscriptionConfig(
                language_detection=True,
                punctuate=punctuate,
                format_text=True
            )
            
            # Transcribe the audio
            transcript = self.transcriber.transcribe(file_path, config=config)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            if transcript is None:
                return {
                    "status": "error",
                    "error": "Transcription returned None",
                    "processing_time": processing_time
                }
            
            if transcript.status == aai.TranscriptStatus.error:
                return {
                    "status": "error",
                    "error": transcript.error,
                    "processing_time": processing_time
                }
            
            print(f"Transcription completed. Status: {transcript.status}")
            
            # Get detected language
            detected_language = getattr(transcript, 'language_code', 'unknown')
            print(f"Detected language: {detected_language}")
            
            # Collect word-level confidence scores
            word_confidences = []
            if hasattr(transcript, 'words') and transcript.words:
                for word in transcript.words:
                    word_confidences.append({
                        "word": word.text,
                        "confidence": word.confidence,
                        "start": word.start,
                        "end": word.end,
                        "duration": word.end - word.start
                    })
            else:
                print("Warning: No words found in transcript")
            
            # Calculate average confidence
            avg_confidence = sum(word["confidence"] for word in word_confidences) / len(word_confidences) if word_confidences else 0
            
            # Collect utterance-level information
            utterances = []
            if hasattr(transcript, 'utterances') and transcript.utterances:
                for utterance in transcript.utterances:
                    utterances.append({
                        "text": utterance.text,
                        "confidence": utterance.confidence,
                        "duration": utterance.duration,
                        "words": len(utterance.words),
                        "speaker": utterance.speaker if hasattr(utterance, 'speaker') else None
                    })
            else:
                print("Warning: No utterances found in transcript")
            
            return {
                "status": "success",
                "processing_time": processing_time,
                "total_duration": getattr(transcript, 'audio_duration', 0),
                "word_count": len(word_confidences),
                "average_confidence": avg_confidence,
                "utterances": utterances,
                "word_confidences": word_confidences,
                "full_text": getattr(transcript, 'text', ''),
                "language": detected_language,
                "punctuated": punctuate
            }
            
        except Exception as e:
            print(f"Exception during transcription: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def save_results(self, results: Dict, filename: str):
        """
        Save transcription results to a JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{filename}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return output_file

    def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze transcription results and generate statistics
        """
        if results["status"] == "error":
            return {"error": results["error"]}
        
        # Calculate statistics
        stats = {
            "processing_time": results["processing_time"],
            "total_duration": results["total_duration"],
            "word_count": results["word_count"],
            "average_confidence": results["average_confidence"],
            "utterance_count": len(results["utterances"]),
            "language": results.get("language", "unknown"),
            "punctuated": results.get("punctuated", False)
        }
        
        # Only calculate these if we have words
        if stats["word_count"] > 0:
            stats["words_per_utterance"] = stats["word_count"] / stats["utterance_count"] if stats["utterance_count"] > 0 else 0
            stats["processing_speed"] = stats["word_count"] / stats["processing_time"] if stats["processing_time"] > 0 else 0
            
            # Confidence distribution
            confidence_ranges = {
                "high": 0,    # > 0.9
                "medium": 0,  # 0.7-0.9
                "low": 0      # < 0.7
            }
            
            for word in results["word_confidences"]:
                conf = word["confidence"]
                if conf > 0.9:
                    confidence_ranges["high"] += 1
                elif conf > 0.7:
                    confidence_ranges["medium"] += 1
                else:
                    confidence_ranges["low"] += 1
            
            stats["confidence_distribution"] = confidence_ranges
        
        return stats

def main():
    analyzer = TranscriptionAnalyzer()
    
    # Test files directory
    test_files_dir = Path("test_audio")
    if not test_files_dir.exists():
        print(f"Please create a 'test_audio' directory and add your audio files there.")
        return
    
    # Process each audio file in the test directory
    for audio_file in test_files_dir.glob("*"):
        if audio_file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.ogg']:
            print(f"\nProcessing {audio_file.name}...")
            
            # Test both with and without punctuation
            for punctuate in [True, False]:
                print(f"\nTesting with punctuation={punctuate}")
                
                # Transcribe and collect metrics
                results = analyzer.transcribe_with_metrics(str(audio_file), punctuate=punctuate)
                
                # Save detailed results
                output_file = analyzer.save_results(results, f"{audio_file.stem}_{'punctuated' if punctuate else 'unpunctuated'}")
                print(f"Detailed results saved to: {output_file}")
                
                # Analyze and display statistics
                stats = analyzer.analyze_results(results)
                
                if "error" in stats:
                    print(f"Error processing {audio_file.name}: {stats['error']}")
                    continue
                
                print("\nTranscription Statistics:")
                print(f"Processing Time: {stats['processing_time']:.2f} seconds")
                print(f"Total Audio Duration: {stats['total_duration']:.2f} seconds")
                print(f"Detected Language: {stats['language']}")
                print(f"Word Count: {stats['word_count']}")
                print(f"Average Confidence: {stats['average_confidence']:.2%}")
                print(f"Utterance Count: {stats['utterance_count']}")
                
                if stats["word_count"] > 0:
                    print(f"Words per Utterance: {stats['words_per_utterance']:.2f}")
                    print(f"Processing Speed: {stats['processing_speed']:.2f} words/second")
                    
                    print("\nConfidence Distribution:")
                    for range_name, count in stats["confidence_distribution"].items():
                        percentage = (count / stats["word_count"]) * 100
                        print(f"{range_name.capitalize()} confidence words: {count} ({percentage:.1f}%)")
                    
                    print("\nSample of transcribed text:")
                    # Print first 200 characters of the text
                    sample_text = results["full_text"][:200] + "..." if len(results["full_text"]) > 200 else results["full_text"]
                    print(sample_text)
                else:
                    print("\nNo words were detected in the audio file.")

if __name__ == "__main__":
    main() 