import os
import sys
import asyncio
import subprocess
from pathlib import Path
import edge_tts

# Clinical consultation dialogue (Doctor and Patient)
DIALOGUE = [
    ("Doctor", "Good morning, Mr. Anderson. Please take a seat and make yourself comfortable. What brings you into the cardiology clinic today?"),
    ("Patient", "Good morning, Doctor. Over the last two to three weeks, I have been having this uncomfortable tightness and pressure right in the center of my chest. It usually happens whenever I walk up a flight of stairs or carry groceries from my car. It forces me to stop and rest for four to five minutes before it fades away."),
    ("Doctor", "I see. That is an important symptom. Does this chest tightness radiate anywhere, such as into your left shoulder, your arm, your neck, or your jaw?"),
    ("Patient", "Yes, actually. Two days ago, it radiated up along the left side of my neck and down to my elbow. I also felt quite short of breath and broke into a mild cold sweat. In addition, my wife noticed that both of my ankles have become puffy and swollen by the evening."),
    ("Doctor", "Have you experienced any palpitations, lightheadedness, or dizziness when standing up?"),
    ("Patient", "I did feel a fluttering sensation in my chest a couple of nights ago while lying flat in bed, and I had to prop myself up on three pillows just to breathe comfortably through the night."),
    ("Doctor", "That sounds like orthopnea. Let us check your medical background first. Do you have a history of high blood pressure, diabetes, or elevated cholesterol? And do you have any known drug allergies?"),
    ("Patient", "I was told three years ago that my blood pressure and cholesterol were high, but I stopped taking my tablets six months ago because I felt fine. I have no known drug allergies."),
    ("Doctor", "Let us perform a thorough physical and cardiovascular examination now. Please lie back on the examination table... Your blood pressure is currently 154 over 94 millimeters of mercury in the right arm, which is significantly elevated. Your heart rate is 88 beats per minute and regular. Your oxygen saturation is 96 percent on room air, and your body mass index is 28.6. Upon auscultation, S1 and S2 heart sounds are normal with no cardiac murmurs, but I can hear mild bilateral bibasilar crackles at the bases of your lungs, along with two-plus pitting edema around both your ankles."),
    ("Patient", "That sounds concerning, Doctor. What exactly is happening to my heart?"),
    ("Doctor", "Based on your symptoms and physical exam, you have stable exertional angina pectoris secondary to suspected coronary artery disease, alongside stage two uncontrolled hypertension and mild fluid retention. We must act quickly to optimize your cardiovascular circulation and relieve your symptoms."),
    ("Patient", "What is the treatment plan, Doctor? Will I need to be hospitalized?"),
    ("Doctor", "We do not need immediate hospitalization today since your resting vitals and room air oxygen are stable, but we must begin targeted medical therapy immediately: First, for the chest discomfort and plaque stabilization, take Atorvastatin 40 milligrams once daily at bedtime. Second, to protect your heart vessels and prevent clot formation, take enteric-coated Ecosprin 75 milligrams once daily after lunch. Third, to control your blood pressure and reduce cardiac workload, take Metoprolol Succinate 50 milligrams once daily in the morning, combined with Telmisartan 40 milligrams once daily. Fourth, for the ankle swelling and lung congestion, take Furosemide 20 milligrams once daily in the morning for seven days. Fifth, I am prescribing Sublingual Nitroglycerin 0.5 milligram tablets. If you experience acute chest tightness, sit down immediately and place one tablet under your tongue. If the pain does not subside within five minutes, call for emergency medical services right away."),
    ("Patient", "Understood. Are there any diagnostic tests or lifestyle changes I should start right now?"),
    ("Doctor", "Yes. You must get an urgent 2D Echocardiogram, a Fasting Lipid Panel, Serum Creatinine, and an HbA1c test completed by tomorrow. You need to restrict your daily sodium intake to less than two grams per day and restrict your total daily fluid intake to 1.5 liters until the swelling resolves. Avoid heavy exertion or lifting weights. Please return to my clinic in exactly ten days with all your lab reports so we can schedule an outpatient coronary angiogram or stress test."),
    ("Patient", "Thank you very much, Doctor. I will start the medications today and follow all instructions strictly.")
]

VOICE_CONFIG = {
    "Doctor": "en-US-ChristopherNeural",   # Authority, authoritative male voice
    "Patient": "en-US-EricNeural"          # Clear, conversational male voice
}

async def generate_all_turns():
    temp_dir = Path("temp_audio_segments")
    temp_dir.mkdir(exist_ok=True)
    
    # 1. Create a 500ms silence MP3 using ffmpeg
    silence_file = temp_dir / "silence_500ms.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", "0.5", "-q:a", "2", str(silence_file)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    file_list = []
    
    print(f"Synthesizing {len(DIALOGUE)} dialogue turns with Neural AI voices...")
    for idx, (speaker, text) in enumerate(DIALOGUE, 1):
        voice = VOICE_CONFIG[speaker]
        out_file = temp_dir / f"turn_{idx:02d}_{speaker}.mp3"
        print(f"[{idx}/{len(DIALOGUE)}] {speaker}: {text[:60]}... -> {voice}")
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(out_file))
        
        file_list.append(out_file)
        file_list.append(silence_file)
    
    # Write concat manifest
    manifest_path = temp_dir / "concat_list.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for fpath in file_list:
            clean_path = str(fpath.resolve()).replace("\\", "/")
            f.write(f"file '{clean_path}'\n")
            
    mp3_output = Path("cardiology_consultation.mp3").resolve()
    wav_output = Path("cardiology_consultation.wav").resolve()
    
    print("\nMerging audio segments with FFmpeg into MP3 and 16kHz WAV...")
    
    # Generate MP3
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(manifest_path),
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(mp3_output)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # Generate 16kHz mono WAV (optimal for Whisper Speech-to-Text)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp3_output),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(wav_output)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # Probe duration
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_output)
    ], capture_output=True, text=True)
    
    duration_sec = float(result.stdout.strip()) if result.stdout.strip() else 0.0
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    
    print("=" * 60)
    print("AUDIO GENERATION COMPLETE!")
    print(f"Duration: {minutes} minutes {seconds} seconds ({duration_sec:.1f}s)")
    print(f"MP3 Path: {mp3_output} ({mp3_output.stat().st_size / (1024*1024):.2f} MB)")
    print(f"WAV Path: {wav_output} ({wav_output.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(generate_all_turns())
