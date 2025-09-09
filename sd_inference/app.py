import cv2  # pip install opencv-python
import torch
import gradio as gr
import numpy as np
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

# ================== 内嵌风格中的正面提示词定义 ==================
prompt_dict = {
    "None": "{prompt}",
    "Enhance": "breathtaking {prompt} . award-winning, professional, highly detailed",
    "Anime": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed",
    "Photographic": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    "Digital Art": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    "Comic Book": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "Fantasy Art": "ethereal fantasy concept art of {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    "Analog Film": "analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
    "Neon Punk": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    "Isometric": "isometric style {prompt} . vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
    "Low Poly": "low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
    "Origami": "origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
    "Line Art": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
    "Craft Clay": "play-doh style {prompt} . sculpture, clay art, centered composition, Claymation",
    "Cinematic": "cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    "3D Model": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    "Pixel Art": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    "Texture": "texture {prompt} top down close-up"
}

# ================== 内嵌风格中的负面提示词定义 ==================
negative_prompt_dict = {
    "None": "{negative_prompt}",
    "Enhance": "{negative_prompt} ugly, deformed, noisy, blurry, distorted, grainy",
    "Anime": "{negative_prompt} photo, deformed, black and white, realism, disfigured, low contrast",
    "Photographic": "{negative_prompt} drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    "Digital Art": "{negative_prompt} photo, photorealistic, realism, ugly",
    "Comic Book": "{negative_prompt} photograph, deformed, glitch, noisy, realistic, stock photo",
    "Fantasy Art": "{negative_prompt} photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    "Analog Film": "{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    "Neon Punk": "{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    "Isometric": "{negative_prompt} deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic",
    "Low Poly": "{negative_prompt} noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Origami": "{negative_prompt} noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Line Art": "{negative_prompt} anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
    "Craft Clay": "{negative_prompt} sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Cinematic": "{negative_prompt} anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    "3D Model": "{negative_prompt} ugly, deformed, noisy, low poly, blurry, painting",
    "Pixel Art": "{negative_prompt} sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    "Texture": "{negative_prompt} ugly, deformed, noisy, blurry"
}

# ================== 工具函数 ==================
torch.cuda.empty_cache()

def clear_fn(value):
    return "", "", "None", 768, 768, 10, 50, None

# ================== 初始化模型管道 ==================
pipe = pipeline(
    task=Tasks.text_to_image_synthesis,
    model='AI-ModelScope/stable-diffusion-xl-base-1.0',
    use_safetensors=True,
    model_revision='v1.0.0'
)

# ================== 推理函数 ==================
def display_pipeline(prompt: str,
                     negative_prompt: str,
                     style: str = 'None',
                     height: int = 768,
                     width: int = 768,
                     scale: float = 10,
                     steps: int = 50,
                     seed: int = 0):
    if not prompt:
        raise gr.Error('The validation prompt is missing.')

    # 格式化提示词
    prompt = prompt_dict[style].format(prompt=prompt)
    negative_prompt = negative_prompt_dict[style].format(negative_prompt=negative_prompt)

    # 设置随机数生成器
    generator = torch.Generator(device='cuda').manual_seed(seed)

    # 调用模型生成图片
    output = pipe({
        'text': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': steps,
        'guidance_scale': scale,
        'height': height,
        'width': width,
        'generator': generator
    })

    result = output['output_imgs'][0]

    # 保存 & 转换为RGB格式返回
    image_path = './lora_result.png'
    cv2.imwrite(image_path, result)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image

# ================== Gradio 界面 ==================
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label='提示词', lines=3)
            negative_prompt = gr.Textbox(label='负向提示词', lines=3)
            style = gr.Dropdown(
                choices=[
                ('无', 'None'),
                ('增强', 'Enhance'),
                ('二次元', 'Anime'),
                ('摄影', 'Photographic'),
                ('数字艺术', 'Digital Art'),
                ('漫画风', 'Comic Book'),
                ('奇幻艺术', 'Fantasy Art'),
                ('胶片风', 'Analog Film'),
                ('霓虹朋克', 'Neon Punk'),
                ('像素画', 'Pixel Art'),
                ('等距视角', 'Isometric'),
                ('低多边形', 'Low Poly'),
                ('折纸', 'Origami'),
                ('线条画', 'Line Art'),
                ('黏土工艺', 'Craft Clay'),
                ('电影感', 'Cinematic'),
                ('三维模型', '3D Model'),
                ('纹理', 'Texture')
                ],
                value='None',
                label='风格'
                )

            with gr.Row():
                height = gr.Slider(512, 1024, 768, step=128, label='高度')
                width = gr.Slider(512, 1024, 768, step=128, label='宽度')

            with gr.Row():
                scale = gr.Slider(1, 15, 10, step=.25, label='引导系数')
                steps = gr.Slider(25, maximum=100, value=50, step=5, label='迭代步数')

            seed = gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='随机数种子')

            with gr.Row():
                clear = gr.Button("清除")
                submit = gr.Button("提交")

        with gr.Column(scale=3):
            output_image = gr.Image()

    submit.click(fn=display_pipeline,
                 inputs=[prompt, negative_prompt, style, height, width, scale, steps, seed],
                 outputs=output_image)

    clear.click(fn=clear_fn, inputs=clear,
                outputs=[prompt, negative_prompt, style, height, width, scale, steps, output_image])

# 启动 Gradio 服务
demo.queue(status_update_rate=1).launch(share=False)
