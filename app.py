import os
import cv2
import whisper
import uuid
import numpy as np

from flask import Flask, render_template, request, send_file, jsonify
import yt_dlp

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import TextClip
import moviepy.video.fx.all as vfx
from moviepy.audio.io.AudioFileClip import AudioFileClip

# Celery (Background Processing)
from celery import Celery

# --- CONFIGURATION ---
os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\ffmpeg\bin\ffmpeg.exe"

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- CELERY CONFIG ----
app.config['CELERY_BROKER_URL'] = os.getenv("CELERY_BROKER_URL")
app.config['CELERY_RESULT_BACKEND'] = os.getenv("CELERY_RESULT_BACKEND")

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# ---- LOAD WHISPER MODEL ----

def get_whisper_model():
    print("Loading Whisper model at runtime (fast mode)...")
    return whisper.load_model("tiny", device="cpu")


# =========================
# HELPER FUNCTIONS
# =========================

def get_face_center(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return x + w // 2
    return None

def download_yt_video(url):
    unique_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{unique_id}.mp4")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': filepath,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return filepath

# =========================
# 1) SMART HIGHLIGHT DETECTION
# =========================

def find_highlight_moments(video_path, result, min_len=25, max_len=45):
    """
    Smart highlights using Whisper speech density instead of raw audio.
    More stable on Windows + Python 3.13.
    """

    highlights = []
    current_start = None

    for segment in result['segments']:
        start = segment['start']
        end = segment['end']

        # If continuous speech is happening, extend highlight
        if current_start is None:
            current_start = start
            last_end = end
        else:
            # If next speech is close (< 3 sec gap), merge
            if start - last_end < 3:
                last_end = end
            else:
                # finalize previous highlight
                clip_len = last_end - current_start
                if clip_len >= min_len:
                    highlights.append((
                        current_start,
                        min(current_start + max_len, last_end)
                    ))
                current_start = start
                last_end = end

    # Add last segment
    if current_start is not None:
        clip_len = last_end - current_start
        if clip_len >= min_len:
            highlights.append((
                current_start,
                min(current_start + max_len, last_end)
            ))

    # Keep top 5 best highlights
    return highlights[:5]

# =========================
# 2) MOVING FACE CROP (9:16)
# =========================

import moviepy.video.fx.all as vfx

def smart_crop_9_16(clip):
    w, h = clip.size
    target_w = int(h * (9 / 16))
    target_w -= target_w % 2  # make even

    # Detect face from first frame
    first_frame = clip.get_frame(0)
    face_x = get_face_center(first_frame) or (w // 2)

    # Compute crop boundaries
    x1 = int(max(0, min(w - target_w, face_x - target_w // 2)))
    x2 = x1 + target_w

    print(f"Cropping video: x1={x1}, x2={x2}, height={h}")

    # ✅ MOVIEPY-SAFE WAY (works on your version)
    return clip.fx(vfx.crop, x1=x1, y1=0, x2=x2, y2=h)


# =========================
# 3) MAIN CLIP GENERATION (AI VERSION)
# =========================

def generate_ai_clips(input_path):
    clip_filenames = []
    print("AI analyzing video for highlights...")

    # Load Whisper model
    model = get_whisper_model()

    # Transcribe full video
    result = model.transcribe(input_path, fp16=False)

    # Find smart highlight moments
    highlights = find_highlight_moments(input_path, result)

    with VideoFileClip(input_path) as video:

        for (start_time, end_time) in highlights[:5]:  # Max 5 clips
            clip_id = str(uuid.uuid4())
            output_filename = f"AI_Clip_{clip_id}.mp4"
            output_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, output_filename))

            print(f"Processing AI Highlight: {start_time}s to {end_time}s")

            sub = video.subclip(start_time, end_time)

            # Apply moving face crop
            final_sub = smart_crop_9_16(sub)

            # ========== 4) ANIMATED SUBTITLES ==========
            subtitle_clips = []
            h = final_sub.size[1]

            for s in result['segments']:
                if start_time <= s['start'] <= end_time:
                    relative_start = s['start'] - start_time

                    txt = (
                        TextClip(
                            text=s['text'].upper(),
                            font_size=42,
                            color='white',
                            stroke_color='black',
                            stroke_width=2,
                            font=r"C:\Windows\Fonts\arialbd.ttf",
                            method='caption',
                            size=(int(final_sub.w * 0.85), None)
                        )
                        .with_start(relative_start)
                        .with_duration(min(s['end'] - s['start'], 3))
                        .with_position(
                            lambda t: (
                                'center',
                                int(h * 0.65 + 10 * np.sin(t * 6))
                            )
                        )
                    )
                    subtitle_clips.append(txt)

            # Render final clip
            final_video = CompositeVideoClip([final_sub] + subtitle_clips)
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
                temp_audiofile=f"temp-{clip_id}.m4a",
                remove_temp=True
            )

            clip_filenames.append(output_filename)

    return clip_filenames

# =========================
# 5) CELERY BACKGROUND TASK
# =========================

@celery.task
def generate_ai_clips_task(video_path):
    return generate_ai_clips(video_path)

# =========================
# FLASK ROUTES
# =========================

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        file = request.files.get('video_file')

        video_path = ""

        if url:
            video_path = os.path.abspath(download_yt_video(url))
        elif file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            video_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
            file.save(video_path)

        if video_path:
            task = generate_ai_clips_task.delay(video_path)
            return render_template('processing.html', task_id=task.id)

    return render_template('index.html')

@app.route('/status/<task_id>')
def task_status(task_id):
    task = generate_ai_clips_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return jsonify({"status": "done", "clips": task.result})
    return jsonify({"status": task.state})

@app.route('/display/<filename>')
def display_video(filename):
    return send_file(os.path.abspath(os.path.join(UPLOAD_FOLDER, filename)))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
