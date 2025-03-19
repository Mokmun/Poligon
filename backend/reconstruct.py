import chainer
import chainer.serializers
import neural_renderer
import numpy as np
import skimage.io
import cupy as cp
from models import Model

def reconstruct_3d(input_image_path, output_obj_path):
    # Load input image
    image_in = skimage.io.imread(input_image_path).astype('float32') / 255
    if image_in.ndim != 3 or image_in.shape[-1] != 4:
        raise Exception("Input must be an RGBA image.")

    images_in = image_in.transpose((2, 0, 1))[None, :, :, :]
    images_in = chainer.cuda.to_gpu(images_in)

    # Load Chainer model
    model = Model()
    model.to_gpu()
    chainer.serializers.load_npz("model.npz", model)

    # Run reconstruction
    vertices, faces = model.reconstruct(images_in)

    # Save as OBJ file
    neural_renderer.save_obj(output_obj_path, vertices.data.get()[0], faces.get()[0])

    print(f"✅ 3D model saved as {output_obj_path}")
