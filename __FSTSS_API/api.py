import subprocess
import sys
import importlib
import os
import json
import queue
from huggingface_hub import snapshot_download
import numpy as np
import wave
import io
import gc
from datetime import datetime
import html
import threading
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import librosa
import torch
import torchaudio
from loguru import logger
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import Form
from pydantic import BaseModel
import uvicorn
from typing import Optional
torchaudio.set_audio_backend("soundfile")

# Инициализация FastAPI
app = FastAPI(title="Fish Speech API")

class TTSParams(BaseModel):
    max_new_tokens: int = 1024
    chunk_length: int = 200
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    seed: Optional[int] = None
    output_format: str = "wav"
    
class ReferenceAudio(BaseModel):
    audio: bytes
    text: str

class TTSRequest(BaseModel):
    text: str
    references: list[ReferenceAudio]
    params: TTSParams

def install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("gradio_i18n")

os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

root_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(root_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
temp_dir = os.path.join(root_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

import shutil
for filename in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Error delete in {file_path}. {e}')

os.environ["GRADIO_TEMP_DIR"] = temp_dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
from fish_speech.utils import autocast_exclude_mps, set_seed
from tools.api import decode_vq_tokens, encode_reference
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
)
from tools.vqgan.inference import load_model as load_decoder_model

def normalize_audio_rms(audio, target_db=-20.0):
    current_rms = np.sqrt(np.mean(audio ** 2))
    current_db = 20 * np.log10(current_rms) if current_rms > 0 else -80.0
    gain = 10 ** ((target_db - current_db) / 20)
    return np.clip(audio * gain, -1.0, 1.0)

@torch.inference_mode()
def inference_api(req: TTSRequest):
    try:
        logger.info(f"Starting inference for text: {req.text}")
        
        # Проверяем reference
        logger.info(f"Processing {len(req.references)} references")
        refs = req.references
        if not refs:
            logger.error("No references provided")
            raise HTTPException(status_code=400, detail="No references provided")
        
        # Кодируем reference аудио
        prompt_tokens = []
        for ref in refs:
            logger.info(f"Encoding reference audio: {ref.text[:50]}...")
            tokens = encode_reference(
                decoder_model=decoder_model,
                reference_audio=ref.audio,
                enable_reference_audio=True,
            )
            if tokens is None or len(tokens) == 0:
                logger.error("Failed to encode reference audio")
                raise HTTPException(status_code=500, detail="Reference encoding failed")
            prompt_tokens.append(tokens)
            logger.info(f"Encoded tokens length: {len(tokens)}")
        
        # Настройка генерации
        logger.info(f"Preparing generation request: {req.params.dict()}")

        request = dict(
            device=decoder_model.device,
            max_new_tokens=req.params.max_new_tokens,
            text=req.text,
            top_p=req.params.top_p,
            repetition_penalty=req.params.repetition_penalty,
            temperature=req.params.temperature,
            compile=args.compile,
            iterative_prompt=req.params.chunk_length > 0,
            chunk_length=req.params.chunk_length,
            max_length=4096,
            prompt_tokens=prompt_tokens,
            prompt_text=[ref.text for ref in refs],
        )
        
        if "!" in req.text:
            request["repetition_penalty"] = max(request["repetition_penalty"] - 0.15, 1.0)
            request["chunk_length"] += 50
        
        # Отправляем запрос в очередь
        logger.info("Submitting request to llama_queue")
        response_queue = queue.Queue()
        llama_queue.put(GenerateRequest(request=request, response_queue=response_queue))
        
        # Обрабатываем ответы
        segments = []
        while True:
            result: WrappedGenerateResponse = response_queue.get()
            if result.status == "error":
                logger.error(f"Generation error: {result.error}")
                raise HTTPException(status_code=500, detail="Generation error")
            
            result: GenerateResponse = result.response
            if result.action == "next":
                logger.info("Generation completed")
                break
            
            logger.info(f"Received segment with {len(result.codes)} codes")
            with autocast_exclude_mps(device_type=decoder_model.device.type, dtype=args.precision):
                fake_audios = decode_vq_tokens(decoder_model=decoder_model, codes=result.codes)
            
            fake_audios = fake_audios.float().cpu().numpy()
            segments.append(fake_audios)
            # if fake_audios is None or fake_audios.numel() == 0:
                # logger.error("Empty audio generated")
                # continue

        
        # Собираем итоговое аудио
        if len(segments) == 0:
            logger.error("Audio not generated")
            raise HTTPException(status_code=500, detail="No audio generated")
        
        audio = np.concatenate(segments, axis=0)
        if audio.shape[0] > 0:
            # Добавляем 50 мс тишины в конец для стабилизации
            silence = np.zeros(int(0.05 * decoder_model.spec_transform.sample_rate))
            audio = np.concatenate([audio, silence])
        logger.info(f"Final audio length: {len(audio)} samples")
        
        # Сохраняем файл
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        if audio_tensor.abs().max() < 1e-6:
            logger.warning("Generated audio is silent")
        
        output_path = os.path.join(outputs_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{req.params.output_format}")
        torchaudio.save(output_path, audio_tensor, decoder_model.spec_transform.sample_rate)
        logger.success(f"Audio saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        output_path = inference_api(request)
        return FileResponse(
            output_path,
            media_type=f"audio/{request.params.output_format}",
            filename=os.path.basename(output_path)
        )
    except Exception as e:
        return {"error": str(e)}

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3"}

@app.post("/tts/simple")
async def simple_tts(
    text: str = Form(...),
    references: str = Form(...),  # "спикер1,спикер2"
    output_format: str = Form(default="wav"),
    max_new_tokens: int = Form(default=1024),
    chunk_length: int = Form(default=200),
    top_p: float = Form(default=0.7),
    repetition_penalty: float = Form(default=1.2),
    temperature: float = Form(default=0.7),
    seed: int = Form(default=None)
):
    try:
        # Парсим текст на сегменты с указанием спикеров
        segments = []
        current_speaker = None
        buffer = ""
        for part in text.split("["):
            if "]" in part:
                speaker, content = part.split("]", 1)
                if current_speaker:
                    segments.append((current_speaker, buffer.strip()))
                current_speaker = speaker
                buffer = content
            else:
                buffer += part
        if current_speaker:
            segments.append((current_speaker, buffer.strip()))
        
        # Загружаем все референсы
        examples_dir = os.path.join(root_dir, "examples")
        ref_list = references.split(",")
        ref_data = {}
        for ref in ref_list:
            audio_path = None
            for ext in ['wav', 'mp3']:
                candidate = os.path.join(examples_dir, f"{ref}.{ext}")
                if os.path.exists(candidate):
                    audio_path = candidate
                    break
            text_path = os.path.join(examples_dir, f"{ref}.txt")
            ref_data[ref] = {
                "audio": open(audio_path, 'rb').read(),
                "text": open(text_path, 'r', encoding='utf-8').read().strip()
            }
        
        # Формируем запрос для каждого сегмента
        all_audio = []
        for speaker, text_part in segments:
            if speaker not in ref_data:
                raise HTTPException(status_code=400, detail=f"Reference for {speaker} not found")
            
            req = TTSRequest(
                text=text_part,
                references=[ReferenceAudio(**ref_data[speaker])],
                params=TTSParams(
                    max_new_tokens=max_new_tokens,
                    chunk_length=chunk_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    seed=seed,
                    output_format=output_format
                )
            )
            audio_path = inference_api(req)
            audio, sr = torchaudio.load(audio_path)
            all_audio.append(audio)
        
        # Объединяем аудио
        final_audio = torch.cat(all_audio, dim=1)
        output_path = os.path.join(outputs_dir, f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}")
        torchaudio.save(output_path, final_audio, sr)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return FileResponse(output_path, media_type=f"audio/{output_format}")
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": decoder_model is not None}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    snapshot_download(repo_id="fishaudio/fish-speech-1.5", local_dir="./checkpoints/fish-speech-1.5")
    logger.info("Checkpoints downloaded")
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16
    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )
    logger.info("Loading decoder model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )