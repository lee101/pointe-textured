import argparse
import itertools

from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.examples import point_to_mesh
from point_e.examples.align_clouds import align_two_point_clouds
from point_e.examples.utils import convert_float_to_bfloat16
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base1B'  # 'base40M' # use base300M or base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
# base_model=base_model.bfloat16()
base_model = convert_float_to_bfloat16(base_model)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
# upsampler_model=upsampler_model.bfloat16()
upsampler_model = convert_float_to_bfloat16(upsampler_model)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))


def image_to_mesh(image, grid_size=3, save_file_name='mesh.ply', text=""):
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )
    # Load an image to condition on.
    img = Image.open(image)
    img_name_no_extension = image.split('.')[0]
    images = [img]
    index = 0
    # while Path(f'{img_name_no_extension}{index}.png').exists():
    #     images.append(Image.open(f'{img_name_no_extension}{index}.png'))
    #     index += 1
    # # resize all images to 256
    # images = [img.resize((256, 256)) for img in images]
    # Produce a sample from the model.
    samples = None
    # auto bfloat16
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.inference_mode():
            # for x in itertools.islice(tqdm(sampler.sample_batch_progressive(batch_size=8, model_kwargs=dict(images=images))), 60): # randomly fails after 65 samples with OOM
            model_kwargs = dict(images=images)
            # if text:
            #     model_kwargs['text'] = text
            i = 0
            for x in itertools.islice(
                    tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=model_kwargs)),
                    190):  # randomly fails after 65 samples with OOM
                # if i > 18:
                #     # prev_pointcloud = sampler.output_to_point_clouds(samples)[0]
                #     current_sample = sampler.output_to_point_clouds(x)[0]
                #     combined = align_two_point_clouds(samples, current_sample)
                #     samples = combined
                # else:
                #     samples = sampler.output_to_point_clouds(x)[0]
                samples = sampler.output_to_point_clouds(x)[0]
                # 65 and 130 are the key areas
                # fig = plot_point_cloud(samples, grid_size=grid_size,
                #                        fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))
                # fig.savefig(f'pics/{img_name_no_extension}_point_cloud{i}.png')
                    # todo combine to pc after the loop again instead

                i+=1

    # pc = sampler.output_to_point_clouds(samples)[0]
    pc = samples
    fig = plot_point_cloud(pc, grid_size=grid_size, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))
    fig.savefig(f'pics/{img_name_no_extension}_point_cloud.png')
    point_to_mesh.convert_point_cloud_to_mesh(pc, 128, save_file_name)


if __name__ == '__main__':
    # get name arg

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='example_data/plastic_toy_soldier_01_aim_rifle.jpg')
    parser.add_argument('--text', type=str, default='Toy Train')
    parser.add_argument('--grid_size', type=int, default=3)
    parser.add_argument('--save_file_name', type=str, default='mesh4.ply')
    args = parser.parse_args()
    image_to_mesh(args.image, args.grid_size, args.save_file_name, args.text)
