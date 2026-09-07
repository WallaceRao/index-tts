import requests
import base64
import os
import sys


# API配置 - 使用环境变量配置IndexTTS2服务器地址
api_url = os.environ.get("INDEX_TTS2_SERVER", "http://127.0.0.1:19200/generate_tts")


def indextts2_tts_url(source_wav, source_text,src_lang, src_duration,target_wav,target_text,target_lang,target_duration):
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
        tts_url_endpoint = api_url.replace("/index_tts", "/generate_tts")

        # 准备multipart/form-data请求
        with open(audio_path, 'rb') as audio_file:
            files = {
                'prompt_audio': ('audio.wav', audio_file, 'audio/wav')
            }
            data = {
                'text': target_text,
                'retry_times': '0',  # 可以根据需要调整重试次数
                'lang':target_lang
            }
            # only modify speed after synthesis
            #if target_duration > 0:
            #    data['desired_seconds'] = target_duration
            response = requests.post(tts_url_endpoint, files=files, data=data)
        
        if response.status_code == 200:
            # 直接返回音频数据
            audio_data = response.content
            if target_wav:
                with open(target_wav, "wb") as f:
                    f.write(audio_data)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"IndexTTS2 URL合成成功，保存到: {target_wav}")
            return audio_data
        else:
            print(f"IndexTTS2 URL请求错误: {response.status_code}, {response.text}")
            return None
            
    except Exception as e:
        print(f"IndexTTS2 URL调用出错: {str(e)}")
        return None



if __name__ == "__main__":
    # 测试代码
    prompt_wav = "/home/ubuntu/raoyonghui/indextts_2.5/index-tts/test_input.wav"
    target_seg_file = "indextts2_result.wav"
    req_id = ""
    reference_text = None
    target_text = "我要开始测试了！！、"
    reference_lang = 'en'
    target_lang = 'zh'
    
    # 测试零样本TTS
    success = indextts2_tts_url(
        prompt_wav, 
        reference_text, 
        reference_lang, 
        2.12, 
        target_seg_file, 
        target_text,
        target_lang,
        0.0,
    )
    
    if success:
        print(f"IndexTTS2输出音频: {target_seg_file}")
    else:
        print("IndexTTS2合成失败")
