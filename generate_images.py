import dnnlib
import dnnlib.tflib as tflib
import config
import pickle
import numpy as np
from PIL import Image

def generate_images(network_pkl, seeds, truncation_psi):
    tflib.init_tf()
    with open(network_pkl, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    for seed in seeds:
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in Gs.vars if 'noise' in var.name})
        images = Gs.run(z, None, truncation_psi=truncation_psi, randomize_noise=False, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        img = Image.fromarray(images[0], 'RGB')
        img.save(f'seed{seed:04d}.png')

seeds = [66, 230, 389, 1518]
generate_images('ffhq.pkl', seeds, 0.5)
