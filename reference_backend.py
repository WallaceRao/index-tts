import requests
import base64
import os
import sys
sys.path.append('.')

from pydub import AudioSegment
from io import BytesIO



# API配置 - 使用环境变量配置IndexTTS2服务器地址
api_url = os.environ.get("REFERENCE_SERVER", "http://127.0.0.1:19200/generate_reference_audio")

#api_url = os.environ.get("REFERENCE_SERVER", "http://127.0.0.1:8006/generate_reference_audio")

def convert_wav_binary_to_mp3_binary(wav_binary):
    audio = AudioSegment.from_file(BytesIO(wav_binary), format="wav",parameters=["-loglevel", "quiet"])
    # 导出为 MP3 字节流
    mp3_buffer = BytesIO()
    audio.export(mp3_buffer, format="mp3", bitrate="128K")
    return mp3_buffer.getvalue()



def get_reference_audio(source_wav, target_wav=None):
    """
    使用IndexTTS2的generate_tts接口进行语音合成
    
    Args:
        source_wav: 参考音频文件路径
        source_text: 参考音频对应的文本（暂未使用）
        src_lang: 源语言（暂未使用）
        src_duration: 源音频时长（暂未使用）
        target_wav: 输出文件路径，如果为None则返回音频数据
        target_text: 要合成的文本
        target_lang: 目标语言（暂未使用）
        target_duration: 目标音频时长（暂未使用）
        
    Returns:
        bool or bytes: 如果指定了target_wav返回bool，否则返回音频数据
    """
    try:
        # 准备音频文件路径（取第一个如果是列表）
        audio_path = source_wav if isinstance(source_wav, str) else source_wav[0]
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"音频文件不存在: {audio_path}")
            return None
        
        # 调用generate_tts接口
        # normalize text
        # 准备multipart/form-data请求
        with open(audio_path, 'rb') as audio_file:
            files = {
                'prompt_audio': ('audio.wav', audio_file, 'audio/wav')
            }
            data = {
                'retry_times': '0',  # 可以根据需要调整重试次数
            }
            # only modify speed after synthesis
            #if target_duration > 0:
            #    data['desired_seconds'] = target_duration
            response = requests.post(api_url, files=files, data=data)
        
        if response.status_code == 200:
            # 直接返回音频数据
            audio_data = response.content
            if target_wav:
                if target_wav.endswith(".mp3"):
                    audio_data = convert_wav_binary_to_mp3_binary(audio_data)
                with open(target_wav, "wb") as f:
                    f.write(audio_data)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"REFERENCE SERVER URL合成成功，保存到: {target_wav}")
            return audio_data
        else:
            print(f"IndexTTS2 URL请求错误: {response.status_code}, {response.text}")
            return None
            
    except Exception as e:
        print(f"IndexTTS2 URL调用出错: {str(e)}")
        return None



if __name__ == "__main__":
    # 测试代码
    prompt_wav = "test_output.wav"
    target_wav = "test_output2.wav"
    audio_data = get_reference_audio(prompt_wav, target_wav)
    if audio_data:
        print(f"合成成功，保存到: {target_wav}")
    else:
        print(f"合成失败")