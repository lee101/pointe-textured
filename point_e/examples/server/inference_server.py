import os
import gradio as gr

import sellerinfo

# fastapi endpoints for text to 3d and for image to 3d

config = {}
from loguru import logger
import stripe

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    title="Generate Object API",
    description="Generate objects from a photo",
    version="1",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/text_to_3d")
def get_text_to_3d(text: str):
    return


is_app_engine = os.environ.get("IS_APP_ENGINE", False)
# if not is_app_engine:
#     def initial_load():
#         model = MODEL_CACHE.add_or_get("text_model", load_pipelines_and_model)
#     daemon = Thread(target=initial_load, args=(), name="Background")
#     # # # download in background thread so that the server can start faster.
#     daemon.start()


# def ensure_pipelines_loaded():
    # if questions.text_generator_inference.loading:
    #     return daemon.join()


# takes too long to start if so.
# if not debug:
#     load_pipelines_and_model()

debug = os.environ.get("SERVER_SOFTWARE", "").startswith("Development")

if debug:
    stripe_keys = {
        "secret_key": sellerinfo.STRIPE_TEST_SECRET,
        "publishable_key": sellerinfo.STRIPE_TEST_KEY,
    }
    GCLOUD_STATIC_BUCKET_URL = ""
else:
    stripe_keys = {
        "secret_key": sellerinfo.STRIPE_LIVE_SECRET,
        "publishable_key": sellerinfo.STRIPE_LIVE_KEY,
    }

stripe.api_key = stripe_keys["secret_key"]

# photo to 3d route
@app.post("/photo_to_3d")
def get_photo_to_3d(photo: str):
    return


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Input Image'),
    outputs=[
        gr.Image(label='Depth'),
        gr.Model3D(label='3D Model', clear_color=[0.0, 0.0, 0.0, 0.0]),
        gr.File(label='Download 3D Model')
    ],
    examples=examples,
    allow_flagging='never',
    cache_examples=False,
    title=title,
    description=description
)
iface.launch()
