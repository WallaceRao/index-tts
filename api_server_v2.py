import os
import sys
import asyncio
import io
import traceback
import base64
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse

from fastapi import APIRouter, HTTPException, Form
import uuid
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import time
import soundfile as sf
from typing import List, Optional
from logger import logger
import tempfile
import torch

try:
    from vllm.v1.engine.exceptions import EngineDeadError
except Exception:  # noqa: BLE001
    class EngineDeadError(Exception):
        """Fallback when vLLM EngineDeadError is unavailable."""

try:
    import resampy
    RESAMPY_AVAILABLE = True
except ImportError:
    RESAMPY_AVAILABLE = False
    print("Warning: resampy not available, some audio resampling features may not work")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "indextts"))

from indextts.infer_vllm_omni_v2_5 import IndexTTS2
from sv_client import get_similar_wav
from wetext import Normalizer
import librosa

import numpy as np
from scipy.signal import find_peaks


LIBROSA_AVAILABLE = True
zh_normalizer = Normalizer(lang="zh", operator="tn", remove_erhua=True)

mos_model = None
tts = None
args = None


def _map_lang(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    return str(lang).strip().lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, mos_model
    logger.info(
        f"Initializing IndexTTS2.5 (vLLM-Omni): model_dir={args.model_dir}, "
        f"is_fp16={args.is_fp16}, gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"temperature={args.temperature}"
    )
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        deploy_config=getattr(args, "deploy_config", None),
    )
    yield
    try:
        tts.shutdown()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def count_syllables(audio_path, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)

    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    energy = np.array([
        np.sum(np.abs(y[i:i + frame_length]) ** 2)
        for i in range(0, len(y) - frame_length, hop_length)
    ])

    peaks, _ = find_peaks(
        energy,
        height=0.1 * np.max(energy),
        distance=int(sr * 0.1 / hop_length),
    )

    syllable_count = len(peaks)
    logger.info(f"检测到的音节数: {syllable_count}")
    return syllable_count


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


async def synthesis_thread(prompt_wav, text, output_wav, desired_seconds=None, calc_mos=True, lang="en"):
    global tts
    try:
        logger.info(f"TTS generation attempt begin for text:{text}")
        await tts.infer(
            spk_audio_prompt=prompt_wav,
            text=text,
            output_path=output_wav,
            lang=_map_lang(lang),
            desired_seconds=desired_seconds,
            verbose=True,
            # API layer already applies zh Normalizer when lang=='zh'
            text_normalization=(_map_lang(lang) != "zh"),
        )

        if not os.path.exists(output_wav):
            logger.warning(f"TTS inference attempt failed to generate output file:{output_wav}.")
            return None

        gen_audio, sample_rate = sf.read(output_wav)

        mos_value = 0
        if calc_mos:
            predict_sample_mos(gen_audio, sample_rate)
        logger.info(f"TTS generation attempt finish for text:{text}, mos:{mos_value}")
        return mos_value
    except torch.AcceleratorError as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with AcceleratorError error: {e}, text:{text}, now restart service")
        os._exit(1)
    except RuntimeError as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with runtime error: {e}, text:{text}, now restart service")
        os._exit(1)
    except EngineDeadError as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with EngineDeadError error: {e}, text:{text}, now restart service")
        os._exit(1)
    except Exception as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with error: {e}, text:{text}")
        return None


async def synthesis_tts_list_thread(prompt_wavs, texts, output_wavs, desired_seconds=None, calc_mos=True, langs=None):
    global tts
    try:
        logger.info(f"TTS generation attempt begin for text:{texts}")
        if desired_seconds is None:
            desired_seconds = [None] * len(texts)
        if langs is None:
            langs = ["en"] * len(texts)

        await tts.infer_list(
            spk_audio_prompts=prompt_wavs,
            texts=texts,
            output_paths=output_wavs,
            langs=[_map_lang(x) for x in langs],
            desired_seconds=desired_seconds,
            verbose=True,
        )

        for output_wav in output_wavs:
            if not os.path.exists(output_wav):
                logger.warning(f"TTS inference attempt failed to generate output file:{output_wav}.")
                return None

        mos_value_list = []
        for idx, output_wav in enumerate(output_wavs):
            gen_audio, sample_rate = sf.read(output_wav)

            mos_value = 0
            if calc_mos:
                predict_sample_mos(gen_audio, sample_rate)
            logger.info(f"TTS generation attempt finish for text:{texts[idx]}, mos:{mos_value}")
            mos_value_list.append(mos_value)
        return mos_value_list
    except torch.AcceleratorError as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with AcceleratorError error: {e}, text:{texts}, now restart service")
        os._exit(1)
    except RuntimeError as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with runtime error: {e}, text:{texts}, now restart service")
        os._exit(1)
    except EngineDeadError as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with EngineDeadError error: {e}, text:{texts}, now restart service")
        os._exit(1)
    except Exception as e:
        logger.exception(e)
        logger.error(f"TTS generation attempt failed with error: {e}, text:{texts}")
        return None


def get_wav_duration(wav_path):
    try:
        # torchaudio>=2.11 may not expose torchaudio.info; use soundfile.
        info = sf.info(wav_path)
        if info.samplerate <= 0 or info.frames <= 0:
            return None
        return info.frames / float(info.samplerate)
    except Exception as e:
        logger.exception(e)
        return None


def get_text_bytes_len(text):
    return len(text.encode('utf-8'))


@app.post("/generate_tts", responses={
    200: {"content": {"audio/wav": {}}},
    400: {"description": "Bad Request"},
    500: {"description": "Internal Server Error"}
})
async def generate_tts(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    lang: str = Form('en'),
    desired_seconds: float = Form(None),
    retry_times: int = Form(0)
):
    """
    使用上传的音频作为音色参考，根据输入文本生成语音。
    - **text**: 要转换为语音的文本。
    - **prompt_audio**: 作为音色参考的音频文件 (WAV, MP3, etc.)。
    - **retry_times**: 重试次数。如果大于0，将多次生成并选择MOS分数最高的音频。
    """
    prompt_audio_temp_path = None
    to_be_clean_wav = None
    output_temp_path = None
    best_audio_path = None
    if lang == 'zh':
        text = zh_normalizer.normalize(text)
    try:
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        if not prompt_audio:
            raise HTTPException(status_code=400, detail="Prompt audio file is required.")

        logger.info(f"Processing TTS request: text='{text}', retry_times={retry_times}")

        prompt_audio_temp_path = await save_audio_to_temp(prompt_audio)
        to_be_clean_wav = prompt_audio_temp_path
        prompt_seconds = get_wav_duration(prompt_audio_temp_path)
        logger.info(f"prompt_seconds: {prompt_seconds}, bytes len:{get_text_bytes_len(text)}")
        if prompt_seconds:
            if prompt_seconds < 1.5 and count_syllables(prompt_audio_temp_path) < 3:
                sv_ret = get_similar_wav(prompt_audio_temp_path)
                if sv_ret and 'sim_wav' in sv_ret:
                    logger.info(f"pick a similar prompt wav {sv_ret['sim_wav']}, origin:{prompt_audio_temp_path}")
                    prompt_audio_temp_path = sv_ret['sim_wav']
                else:
                    logger.error(f"get similar wav failed, prompt_audio:{prompt_audio_temp_path}, sv_ret:{sv_ret}")
        max_mos = -1
        best_audio_bytes = None
        tasks = []
        output_wavs = []
        calc_mos = False
        if retry_times > 0:
            calc_mos = True
        for i in range(retry_times + 1):
            output_temp_path = os.path.join(tempfile.gettempdir(), f"tts_output_{uuid.uuid4().hex}.wav")
            tasks.append(synthesis_thread(
                prompt_audio_temp_path, text, output_temp_path, desired_seconds, calc_mos, lang=lang
            ))
            output_wavs.append(output_temp_path)
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            cur_mos = result
            if cur_mos is not None and cur_mos > max_mos:
                max_mos = cur_mos
                best_audio_path = output_wavs[i]
        logger.info(f"pick best audio path:{best_audio_path} for text{text}, most:{max_mos}")
        if best_audio_path is not None:
            with open(best_audio_path, 'rb') as f:
                best_audio_bytes = f.read()

        if best_audio_bytes is None:
            raise HTTPException(status_code=500, detail="TTS inference failed to generate any valid audio.")

        return Response(content=best_audio_bytes, media_type="audio/wav")

    except HTTPException as e:
        raise e
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"Error in generate_tts: {tb_str}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

    finally:
        await cleanup_resources([to_be_clean_wav, output_temp_path, best_audio_path])
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            GPU_MEM_THRESHOLD_MB = 16000
            if allocated > GPU_MEM_THRESHOLD_MB or reserved > GPU_MEM_THRESHOLD_MB:
                logger.info(f"Freeing GPU memory, reserved:{reserved}, allocated:{allocated}")
                torch.cuda.empty_cache()
                reserved = torch.cuda.memory_reserved() / 1024 ** 2
                allocated = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(f"After freeing GPU memory, reserved:{reserved}, allocated:{allocated}")


class TTSListRequest(BaseModel):
    texts: List[str]
    prompt_audios: List[str]
    langs: List[str] = None
    desired_seconds: List[Optional[float]] = None
    retry_times: int = 0


@app.post("/generate_tts_list", responses={
    200: {"content": {"audios bytes list json"}},
    400: {"description": "Bad Request"},
    500: {"description": "Internal Server Error"}
})
async def generate_tts_list(request: TTSListRequest):
    """
    使用上传的音频作为音色参考，根据输入文本生成语音。
    - **text**: 要转换为语音的文本。
    - **prompt_audio**: 作为音色参考的音频文件 (WAV, MP3, etc.)。
    - **retry_times**: 重试次数。如果大于0，将多次生成并选择MOS分数最高的音频。
    """
    texts = request.texts
    prompt_audios = request.prompt_audios
    langs = request.langs if request.langs is not None else ['en'] * len(texts)
    desired_seconds = request.desired_seconds if request.desired_seconds is not None else [None] * len(texts)
    retry_times = request.retry_times

    prompt_audio_temp_paths = []
    output_temp_paths_list = []
    best_audio_paths = []

    try:
        if not texts:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        if not prompt_audios:
            raise HTTPException(status_code=400, detail="Prompt audio file is required.")
        if not (len(texts) == len(prompt_audios) == len(langs) == len(desired_seconds)):
            raise HTTPException(status_code=400, detail="Input lists must have the same length.")

        logger.info(f"Processing TTS request: text='{texts}', retry_times={retry_times}")

        for idx, lang in enumerate(langs):
            texts[idx] = zh_normalizer.normalize(texts[idx]) if lang == 'zh' else texts[idx]

        for pa in prompt_audios:
            audio_bytes = base64.b64decode(pa)
            temp_path = os.path.join(tempfile.gettempdir(), f"prompt_{uuid.uuid4().hex}.wav")
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            prompt_audio_temp_paths.append(temp_path)

        max_mos_list = [-1 for _ in range(len(texts))]
        tasks = []
        output_temp_paths_list = []
        best_audio_paths = [None for _ in range(len(texts))]
        calc_mos = False
        if retry_times > 0:
            calc_mos = True
        for i in range(retry_times + 1):
            output_temp_paths = [
                os.path.join(tempfile.gettempdir(), f"tts_output_{uuid.uuid4().hex}.wav")
                for _ in range(len(texts))
            ]
            tasks.append(synthesis_tts_list_thread(
                prompt_audio_temp_paths, texts, output_temp_paths, desired_seconds, calc_mos, langs=langs
            ))
            output_temp_paths_list.append(output_temp_paths)
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            if result is None:
                continue
            mos_value_list = result
            for idx, cur_mos in enumerate(mos_value_list):
                if cur_mos is not None and cur_mos > max_mos_list[idx]:
                    max_mos_list[idx] = cur_mos
                    best_audio_paths[idx] = output_temp_paths_list[i][idx]
        logger.info(f"pick best audio path:{best_audio_paths} for text{texts}, most:{max_mos_list}")
        best_audio_bytes_list = []
        for best_audio_path in best_audio_paths:
            if best_audio_path is not None:
                with open(best_audio_path, 'rb') as f:
                    best_audio_bytes = f.read()
            else:
                raise HTTPException(status_code=500, detail="TTS inference failed to generate any valid audio.")
            best_audio_bytes_list.append(best_audio_bytes)

        return JSONResponse(content={"audios": [base64.b64encode(b).decode() for b in best_audio_bytes_list]})

    except HTTPException as e:
        raise e
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"Error in generate_tts: {tb_str}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

    finally:
        output_temp_paths = []
        for paths in output_temp_paths_list:
            output_temp_paths.extend(paths)
        await cleanup_resources(prompt_audio_temp_paths + output_temp_paths + best_audio_paths)


async def detect_audio_format(audio_binary: bytes) -> str:
    """检测音频格式"""
    for signature, format_name in AUDIO_SIGNATURES.items():
        if audio_binary.startswith(signature):
            return format_name
    return "unknown"


async def file_to_base64(file_path):
    """读取文件并转换为Base64编码"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


async def cleanup_resources(file_paths):
    """清理临时文件资源"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"Cleaned up temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up file {path}: {str(e)}")


def predict_sample_mos(samples, sample_rate, window=None, score_rate=None, return_mean=False, round_digits=None):
    if score_rate is None:
        score_rate = 16000

    if RESAMPY_AVAILABLE and sample_rate != score_rate:
        audio_test = resampy.resample(samples, sample_rate, score_rate, axis=0)
    else:
        audio_test = samples
        if sample_rate != score_rate:
            logger.warning(f"Cannot resample from {sample_rate} to {score_rate}: resampy not available")

    if len(audio_test.shape) == 2:
        if audio_test.shape[1] > 1:
            audio_test = audio_test[:, 0]
        else:
            audio_test = audio_test[0]

    data = {}
    data['audio'] = [audio_test]
    data['rate'] = score_rate if RESAMPY_AVAILABLE else sample_rate
    result_score = 1  # mos_model.scoring(data, window, data['rate'], round_digits)
    return 1


async def save_audio_to_temp(uploaded_file: UploadFile) -> str:
    """将上传的音频文件保存到临时文件，如果不是WAV格式则进行转换。"""
    try:
        audio_binary = await uploaded_file.read()

        audio_format = await detect_audio_format(audio_binary)
        logger.info(f"Detected audio format: {audio_format}")

        temp_path = os.path.join(tempfile.gettempdir(), f"prompt_{uuid.uuid4().hex}.wav")

        if audio_format == "wav":
            with open(temp_path, 'wb') as temp_audio:
                temp_audio.write(audio_binary)
        else:
            if not LIBROSA_AVAILABLE:
                raise ValueError(f"Cannot convert {audio_format} format: librosa is not available. Please install it.")

            try:
                audio_stream = io.BytesIO(audio_binary)
                y, sr = librosa.load(audio_stream, sr=None)
                sf.write(temp_path, y, sr)
                logger.info(f"Audio converted from {audio_format} to WAV format.")
            except Exception as e:
                logger.error(f"Audio conversion failed: {str(e)}")
                raise ValueError(f"Unsupported audio format '{audio_format}' or conversion failed.")

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise IOError(f"Failed to write audio data to {temp_path}")

        logger.info(f"Audio saved to temporary file: {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"Error saving audio file: {str(e)}")
        raise


AUDIO_SIGNATURES = {
    b'RIFF': 'wav',
    b'OggS': 'ogg',
    b'fLaC': 'flac',
    b'\xff\xf1': 'aac',
    b'\xff\xf9': 'mp3',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19200)
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Kept for CLI compatibility")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.3, help="Stage0 sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument(
        "--deploy_config",
        type=str,
        default=None,
        help="Optional override for vLLM-Omni indextts2_5.yaml",
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    logger.info(f"Server will launch with port: {args.port} (backend=vLLM-Omni)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)
