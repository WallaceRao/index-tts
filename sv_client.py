import requests
import os

sv_url = os.environ.get("SV_SERVER_URL", "http://127.0.0.1:8072/sv_process")
mos_url = sv_url.replace("sv_process", "mos_process")


def get_similar_wav(source_wav):
    url = sv_url
    file_name = source_wav.split("/")[-1]
    with open(source_wav, 'rb') as audio_file:
        files = {
            'uploaded_file': (file_name, audio_file, 'audio/wav')
        }
        foo = {
            'source_wav_path': source_wav,
        }
        response = requests.post(url, data=foo, files=files)
        json_obj = response.json()
        return json_obj

    return None


def get_wav_mos(source_wav):
    url = mos_url
    file_name = source_wav.split("/")[-1]
    with open(source_wav, 'rb') as audio_file:
        files = {
            'uploaded_file': (file_name, audio_file, 'audio/wav')
        }
        foo = {
            'source_wav_path': source_wav,
        }
        response = requests.post(url, data=foo, files=files)
        json_obj = response.json()
        return json_obj['mos']

    return None


if __name__ == "__main__":
    for i in range(100):
        ret = get_similar_wav("/home/ubuntu/raoyonghui/chatterbox/cosyvoice3_result.wav")
        print(f"origin_seg_{i}.mp3: {ret}")
