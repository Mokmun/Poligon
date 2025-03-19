import os
import numpy as np
import torch
from skimage import measure
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import KDTree

# import lightconvpoint.utils.transforms as lcp_T
# import torch_geometric.transforms as T
import logging
from generate import export_mesh_and_refine_vertices_region_growing_v2, count_parameters


def mesh_recon(input_points):

    # Configuration
    config = {
        "device": "cuda",
        "save_dir": "models",
        "network_n_labels": 2,  # Adjust based on your model
        "network_latent_size": 32,
        "network_backbone": "FKAConv",
        "network_decoder": {"name": "InterpAttentionKHeadsNet", "k": 64},
        "gen_resolution_global": 64,  # Adjust resolution as needed
        "gen_refine_iter": 5,
        "manifold_points": 642,
        "random_noise": 0.005,
        "normals": "false",
    }
    # Load the input point cloud
    print(f"Loaded point cloud from {input_points}: {input_points.shape} points")

    # Load the pretrained network
    device = torch.device(config["device"])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint_path = os.path.join(config["save_dir"], "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize the network
    from networks import Network  # Ensure 'networks' is accessible
    net = Network(3, config["network_latent_size"], config["network_n_labels"], config["network_backbone"], config["network_decoder"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()
    print(f"Network loaded with {count_parameters(net)} parameters.")

    input_points = torch.tensor(input_points, dtype=torch.float, device=device).unsqueeze(0)
    input_points = input_points.transpose(1, 2)
    # Feed into the network
    data = {"pos": input_points.to(device),
            "x": torch.zeros_like(input_points).to(device)}

    with torch.no_grad():
        # Prepare latent vector
        latent = net.get_latent(data, with_correction=False)

        # Generate 3D mesh
        mesh = export_mesh_and_refine_vertices_region_growing_v2(
            network=net,
            latent=latent,
            resolution=config["gen_resolution_global"],
            padding=1,
            mc_value=0,
            device=device,
            input_points=data["pos"][0].cpu().numpy().transpose(1, 0),
            refine_iter=config["gen_refine_iter"],
            out_value=1,
        )

        if mesh is not None:
            # Save the generated mesh
            # o3d.io.write_triangle_mesh(output_mesh_path, mesh)
            return mesh
        else:
            print(f"Mesh generation failed.")



