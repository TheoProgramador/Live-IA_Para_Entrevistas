import ffmpeg

input_video = r"video.mkv"
output_audio = r"audio_extraido.wav"

ffmpeg.input(input_video).output(output_audio).run()
