from typing import List, Tuple, Union, Dict, Any
import re
import os

class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

        # change name according to our preset wav file
        new_dict = {}
        for name, path in self.voice_presets.items():
            
            if '_' in name:
                name = name.split('_')[0]
            
            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)
        # print(list(self.voice_presets.keys()))

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        
        # Default to first voice if no match found
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice

def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    lines = txt_content.strip().split('\n')
    scripts = []
    speaker_numbers = []
    
    # Pattern to match "Speaker X:" format where X is a number
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    
    current_speaker = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            # If we have accumulated text from previous speaker, save it
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            
            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            # Continue text for current speaker
            if current_text:
                current_text += " " + line
            else:
                current_text = line
    
    # Don't forget the last speaker
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)
    
    return scripts, speaker_numbers

def input_process(txt_content: str, voices: List[str]) -> Tuple[str, List[str]]:
    voice_mapper = VoiceMapper()
    scripts, speaker_numbers = parse_txt_script(txt_content)
    full_script = '\n'.join(scripts)

    speaker_name_mapping = {}
    speaker_names_list = voices
    for i, name in enumerate(speaker_names_list):
        speaker_name_mapping[str(i)] = name
    voice_samples = []

    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)

    for speaker_num in unique_speaker_numbers:
        speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        voice_path = voice_mapper.get_voice_path(speaker_name)
        voice_samples.append(voice_path)

    return full_script, voice_samples
