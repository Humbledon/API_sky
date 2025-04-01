from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import random
import logging
from typing import Optional
from diffusers.utils import export_to_video
import base64
from io import BytesIO
from PIL import Image
from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer
import socket

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SkyReels Video Generation API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes
    allow_headers=["*"],  # Autorise tous les headers
)

# Fonction pour obtenir l'adresse IP locale
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

class VideoGenerationRequest(BaseModel):
    model_id: str = "Skywork/SkyReels-V1-Hunyuan-T2V"
    task_type: str = "t2v"
    guidance_scale: float = 6.0
    height: int = 544
    width: int = 960
    num_frames: int = 97
    prompt: str
    embedded_guidance_scale: float = 1.0
    negative_prompt: str = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    num_inference_steps: int = 30
    seed: int = -1
    fps: int = 24
    quant: bool = False
    offload: bool = False
    high_cpu_memory: bool = False
    parameters_level: bool = False
    compiler_transformer: bool = False
    sequence_batch: bool = False
    image_base64: Optional[str] = None

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    try:
        logger.info(f"Received video generation request with task_type: {request.task_type}")
        out_dir = "results/api_outputs"
        os.makedirs(out_dir, exist_ok=True)

        # Gérer l'image si c'est une requête i2v
        image = None
        if request.task_type == "i2v" and request.image_base64:
            try:
                logger.info("Processing i2v request with image")
                # Décoder l'image base64
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(BytesIO(image_data))
                logger.info(f"Image successfully decoded, size: {image.size}")
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        if request.seed == -1:
            random.seed(time.time())
            request.seed = int(random.randrange(4294967294))
            logger.info(f"Generated random seed: {request.seed}")

        logger.info("Initializing SkyReelsVideoInfer")
        predictor = SkyReelsVideoInfer(
            task_type=TaskType.I2V if request.task_type == "i2v" else TaskType.T2V,
            model_id=request.model_id,
            quant_model=request.quant,
            world_size=1,
            is_offload=request.offload,
            offload_config=OffloadConfig(
                high_cpu_memory=request.high_cpu_memory,
                parameters_level=request.parameters_level,
                compiler_transformer=request.compiler_transformer,
            ),
            enable_cfg_parallel=request.guidance_scale > 1.0 and not request.sequence_batch,
        )

        # S'assurer que le prompt et le negative_prompt sont des chaînes de caractères
        prompt = str(request.prompt)
        negative_prompt = str(request.negative_prompt)

        kwargs = {
            "prompt": prompt,
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "num_inference_steps": request.num_inference_steps,
            "seed": request.seed,
            "guidance_scale": request.guidance_scale,
            "embedded_guidance_scale": request.embedded_guidance_scale,
            "negative_prompt": negative_prompt,
            "cfg_for": request.sequence_batch,
        }

        if request.task_type == "i2v" and image:
            kwargs["image"] = image

        logger.info("Starting video generation")
        output = predictor.inference(kwargs)
        video_out_file = f"{prompt[:100].replace('/','')}_{request.seed}.mp4"
        video_path = f"{out_dir}/{video_out_file}"
        export_to_video(output, video_path, fps=request.fps)
        logger.info(f"Video generated successfully: {video_path}")

        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_out_file,
            background=None
        )

    except Exception as e:
        logger.error(f"Error during video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def print_api_info():
    local_ip = get_local_ip()
    print("\n" + "="*50)
    print("🚀 SkyReels Video Generation API")
    print("="*50)
    print(f"\n📡 API URL: http://{local_ip}:8000")
    print(f"📝 Documentation: http://{local_ip}:8000/docs")
    print("\n🔧 Endpoint disponible:")
    print(f"   POST http://{local_ip}:8000/generate-video")
    print("\n📦 Exemple de requête (T2V):")
    print("""{
    "model_id": "Skywork/SkyReels-V1-Hunyuan-T2V",
    "task_type": "t2v",
    "guidance_scale": 6.0,
    "height": 544,
    "width": 960,
    "num_frames": 97,
    "prompt": "FPS-24, A cat wearing sunglasses and working as a lifeguard at a pool",
    "embedded_guidance_scale": 1.0
}""")
    print("\n📦 Exemple de requête (I2V):")
    print("""{
    "model_id": "Skywork/SkyReels-V1-Hunyuan-T2V",
    "task_type": "i2v",
    "guidance_scale": 6.0,
    "height": 544,
    "width": 960,
    "num_frames": 97,
    "prompt": "FPS-24, Transform this image into a cinematic scene",
    "embedded_guidance_scale": 1.0,
    "image_base64": "base64_encoded_image_string"
}""")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    import uvicorn
    print_api_info()
    uvicorn.run(app, host="0.0.0.0", port=8080, access_log=True) 