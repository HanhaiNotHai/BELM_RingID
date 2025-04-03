import itertools
import random
import time

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from tqdm import tqdm

from BELM.samplers import BELM, test_sd15
from BELM.samplers.test_sd15 import load_im_into_format_from_path, pil_to_latents
from BELM.samplers.utils import PipelineLike
from BELM.scripts.reconstruction import get_jpg_paths
from RingID.inverse_stable_diffusion import InversableStableDiffusionPipeline
from RingID.utils import (
    fft,
    generate_Fourier_watermark_latents,
    get_distance,
    ifft,
    make_Fourier_ringid_pattern,
    ring_mask,
    transform_img,
)


@torch.inference_mode()
def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    directory = 'dataset/DIV2K/DIV2K_train_HR'
    test_num = 100

    ######################
    # RingID
    ######################
    RingID_model_id = 'stabilityai/stable-diffusion-2-1-base'
    channel_min = 1
    ring_value_range = 64
    num_inmost_keys = 2
    assigned_keys = -1
    fix_gt = 1
    time_shift = 1
    test_num_inference_steps = 50

    RADIUS = 14
    RADIUS_CUTOFF = 3

    HETER_WATERMARK_CHANNEL = [0]
    RING_WATERMARK_CHANNEL = [3]
    WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)

    scheduler = DPMSolverMultistepScheduler.from_pretrained(RingID_model_id, subfolder='scheduler')
    pipe: InversableStableDiffusionPipeline = InversableStableDiffusionPipeline.from_pretrained(
        RingID_model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        # variant='fp16',
    )
    pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    tester_prompt = ''  # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    if channel_min:
        assert len(HETER_WATERMARK_CHANNEL) > 0

    eval_methods = [
        {
            'Distance': 'L1',
            'Metrics': '|a-b|        ',
            'func': get_distance,
            'kwargs': {'p': 1, 'mode': 'complex', 'channel_min': channel_min},
        }
    ]

    base_latents = pipe.get_random_latents()
    base_latents = base_latents.to(torch.float64)
    original_latents_shape = base_latents.shape
    sing_channel_ring_watermark_mask = torch.tensor(
        ring_mask(size=original_latents_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF)
    )

    # get heterogeneous watermark mask
    if len(HETER_WATERMARK_CHANNEL) > 0:
        single_channel_heter_watermark_mask = torch.tensor(
            ring_mask(
                size=original_latents_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF
            )  # TODO: change to whole mask
        )
        heter_watermark_region_mask = (
            single_channel_heter_watermark_mask.unsqueeze(0)
            .repeat(len(HETER_WATERMARK_CHANNEL), 1, 1)
            .to(device)
        )

    watermark_region_mask = []
    for channel_idx in WATERMARK_CHANNEL:
        if channel_idx in RING_WATERMARK_CHANNEL:
            watermark_region_mask.append(sing_channel_ring_watermark_mask)
        else:
            watermark_region_mask.append(single_channel_heter_watermark_mask)
    watermark_region_mask = torch.stack(watermark_region_mask).to(device)  # [C, 64, 64]

    # ###### Make RingID pattern
    single_channel_num_slots = RADIUS - RADIUS_CUTOFF
    key_value_list = [
        [
            list(combo)
            for combo in itertools.product(
                np.linspace(-ring_value_range, ring_value_range, num_inmost_keys).tolist(),
                repeat=len(RING_WATERMARK_CHANNEL),
            )
        ]
        for _ in range(single_channel_num_slots)
    ]
    key_value_combinations = list(itertools.product(*key_value_list))

    # random select from all possible value combinations, then generate patterns for selected ones.
    if assigned_keys > 0:
        assert assigned_keys <= len(key_value_combinations)
        key_value_combinations = random.sample(key_value_combinations, k=assigned_keys)
    Fourier_watermark_pattern_list = [
        make_Fourier_ringid_pattern(
            device,
            list(combo),
            base_latents,
            radius=RADIUS,
            radius_cutoff=RADIUS_CUTOFF,
            ring_watermark_channel=RING_WATERMARK_CHANNEL,
            heter_watermark_channel=HETER_WATERMARK_CHANNEL,
            heter_watermark_region_mask=(
                heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL) > 0 else None
            ),
        )
        for _, combo in enumerate(key_value_combinations)
    ]

    ring_capacity = len(Fourier_watermark_pattern_list)

    if fix_gt:
        Fourier_watermark_pattern_list = [
            fft(ifft(Fourier_watermark_pattern).real)
            for Fourier_watermark_pattern in Fourier_watermark_pattern_list
        ]

    if time_shift:
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            # Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * time_shift_factor)
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(
                torch.fft.fftshift(
                    ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2)
                )
            )

    ######################
    # BELM
    ######################
    guidance_scale = 1.0
    num_inference_steps = 100
    BELM_model_id = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

    # sd = StableDiffusionPipeline.from_pretrained(
    #     BELM_model_id,
    #     torch_dtype=torch.float32,
    #     # variant='fp16',
    # )
    sd = pipe
    sche = DDIMScheduler(
        beta_end=0.012,
        beta_start=0.00085,
        beta_schedule='scaled_linear',
        clip_sample=False,
        timestep_spacing='linspace',
        set_alpha_to_one=False,
    )
    sd_pipe = PipelineLike(
        device=device,
        vae=sd.vae,
        text_encoder=sd.text_encoder,
        tokenizer=sd.tokenizer,
        unet=sd.unet,
        scheduler=sche,
    )
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)

    sd_params = {
        'prompt': '',
        'negative_prompt': '',
        'seed': 1,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps,
        'width': 512,
        'height': 512,
    }

    ######################
    # img2img
    ######################
    img2img = StableDiffusionImg2ImgPipeline(
        pipe.vae,
        pipe.text_encoder,
        pipe.tokenizer,
        pipe.unet,
        pipe.scheduler,
        pipe.safety_checker,
        pipe.feature_extractor,
        pipe.image_encoder,
        pipe.config.requires_safety_checker,
    )
    img2img.to(device)
    img2img.set_progress_bar_config(leave=False)

    jpg_paths = get_jpg_paths(directory)
    key_indices_to_evaluate = np.random.choice(ring_capacity, size=test_num, replace=True).tolist()

    for strength_i in range(11):
        strength = strength_i / 10
        results_list = []
        for im, key_index in tqdm(list(zip(jpg_paths, key_indices_to_evaluate))):
            # disk to pil_image
            im = load_im_into_format_from_path(im)

            ######################
            # BELM
            ######################
            # pil_image to latent
            latent = pil_to_latents(pil_image=im, sd_pipe=sd_pipe)
            # latent to intermediate
            intermediate, second_intermediate = BELM.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params, latent=latent
            )

            ######################
            # RingID
            ######################
            Fourier_watermark_latents = generate_Fourier_watermark_latents(
                device=device,
                radius=RADIUS,
                radius_cutoff=RADIUS_CUTOFF,
                original_latents=intermediate,
                watermark_pattern=Fourier_watermark_pattern_list[key_index],
                watermark_channel=WATERMARK_CHANNEL,
                watermark_region_mask=watermark_region_mask,
            )

            ######################
            # BELM
            ######################
            # intermediate to latent
            recon_latent = BELM.intermediate_to_latent(
                sd_pipe=sd_pipe,
                sd_params=sd_params,
                intermediate=Fourier_watermark_latents,
                # intermediate_second=second_intermediate,
            )
            # latent to pil_image
            pil = test_sd15.to_pil(latents=recon_latent, sd_pipe=sd_pipe)

            ######################
            # RingID
            ######################
            distorted_image = pil
            if strength > 0:
                distorted_image = img2img('', pil, strength).images[0]
            Fourier_watermark_image_distorted = (
                torch.stack([transform_img(distorted_image)]).to(text_embeddings.dtype).to(device)
            )
            Fourier_watermark_image_latents = pipe.get_image_latents(
                Fourier_watermark_image_distorted, sample=False
            )  # [N, c, h, w]

            Fourier_watermark_reconstructed_latents = pipe.forward_diffusion(
                latents=Fourier_watermark_image_latents,
                text_embeddings=torch.cat(
                    [text_embeddings] * len(Fourier_watermark_image_latents)
                ),
                guidance_scale=1,
                num_inference_steps=test_num_inference_steps,
            )

            Fourier_watermark_reconstructed_latents_fft = fft(
                Fourier_watermark_reconstructed_latents
            )  # [N，c, h, w]

            eval_method = eval_methods[0]
            distances_list = []
            for Fourier_watermark_pattern in Fourier_watermark_pattern_list:  # traverse all gts
                distance_per_gt = eval_method['func'](
                    Fourier_watermark_pattern,
                    Fourier_watermark_reconstructed_latents_fft,
                    watermark_region_mask,
                    channel=WATERMARK_CHANNEL,
                    **eval_method['kwargs'],
                )
                distances_list.append(distance_per_gt)
            acc = np.argmin(np.array(distances_list)) == key_index
            results_list.append(acc)

        acc = sum(results_list) / len(results_list)
        result = time.strftime(f'%Y/%m/%d %H:%M:%S ') + f'{test_num=} {strength=} {acc=}\n'
        with open('result', 'a') as f:
            f.write(result)


if __name__ == '__main__':
    main()
