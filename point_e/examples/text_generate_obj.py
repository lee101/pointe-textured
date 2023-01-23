import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.examples import point_to_mesh
from point_e.examples.utils import convert_float_to_bfloat16
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
# base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model = convert_float_to_bfloat16(base_model)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
# base_diffusion = convert_float_to_bfloat16(base_diffusion)
# base_diffusion.eval()

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model = convert_float_to_bfloat16(upsampler_model)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))
# %%
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''),  # Do not condition the upsampler at all
)


# %%
def text_to_mesh(prompt, grid_size=3, save_file_name='mesh.ply'):
    # Set a prompt to condition on.
    img_name_no_extension = save_file_name.split('.')[0]

    # Produce a sample from the model.
    samples = None
    # auto bfloat16
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.inference_mode():
            for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
                samples = x
    # %%
    pc = sampler.output_to_point_clouds(samples)[0]
    fig = plot_point_cloud(pc, grid_size=grid_size, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))
    fig.savefig(f'{img_name_no_extension}_point_cloud.png')
    point_to_mesh.convert_point_cloud_to_mesh(pc, 128, save_file_name)


if __name__ == '__main__':
    text_to_mesh('a toy soldier army sniper', grid_size=3, save_file_name='mesh-sniper.ply')
    # convert_point_cloud_to_mesh('example_data/pc_corgi.npz', grid_size=32, save_file_name='corgi_mesh.ply')
