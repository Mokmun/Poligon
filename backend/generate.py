
import numpy as np
from tqdm import tqdm
import math

from skimage import measure
import open3d as o3d

# torch imports
import torch
import torch.nn.functional as F

def export_mesh_and_refine_vertices_region_growing_v2(
    network,latent,
    resolution,
    padding=0,
    mc_value=0,
    device=None,
    num_pts=50000, 
    refine_iter=10, 
    simplification_target=None,
    input_points=None,
    refine_threshold=None,
    out_value=np.nan,
    step = None,
    dilation_size=2,
    whole_negative_component=False,
    return_volume=False
    ):

    bmin=input_points.min()
    bmax=input_points.max()

    if step is None:
        step = (bmax-bmin) / (resolution -1)
        resolutionX = resolution
        resolutionY = resolution
        resolutionZ = resolution
    else:
        bmin = input_points.min(axis=0)
        bmax = input_points.max(axis=0)
        resolutionX = math.ceil((bmax[0]-bmin[0])/step)
        resolutionY = math.ceil((bmax[1]-bmin[1])/step)
        resolutionZ = math.ceil((bmax[2]-bmin[2])/step)

    bmin_pad = bmin - padding * step
    bmax_pad = bmax + padding * step

    pts_ids = (input_points - bmin)/step + padding
    pts_ids = pts_ids.astype(np.int)

    # create the volume
    volume = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), np.nan, dtype=np.float64)
    mask_to_see = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), True, dtype=bool)
    while(pts_ids.shape[0] > 0):

        # creat the mask
        mask = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)
        mask[pts_ids[:,0], pts_ids[:,1], pts_ids[:,2]] = True

        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i,0])
            yc = int(pts_ids[i,1])
            zc = int(pts_ids[i,2])
            mask[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True

        # get the valid points
        valid_points_coord = np.argwhere(mask).astype(np.float32)
        valid_points = valid_points_coord * step + bmin_pad

        # get the prediction for each valid points
        z = []
        near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
        # Move latent to the correct device
        for key in latent:
            if isinstance(latent[key], torch.Tensor):
                latent[key] = latent[key].to(device)

        # Move network to the correct device
        network.to(device)
        
        for pnts in tqdm(torch.split(near_surface_samples_torch,num_pts,dim=0), ncols=100, disable=True):

            latent["pos_non_manifold"] = pnts.unsqueeze(0)
            occ_hat = network.from_latent(latent).to(device)

            # get class and max non class
            class_dim = 1
            occ_hat = torch.stack([occ_hat[:, class_dim] , occ_hat[:,[i for i in range(occ_hat.shape[1]) if i!=class_dim]].max(dim=1)[0]], dim=1)
            occ_hat = F.softmax(occ_hat, dim=1)
            occ_hat[:, 0] = occ_hat[:, 0] * (-1)
            if class_dim == 0:
                occ_hat = occ_hat * (-1)


            # occ_hat = -occ_hat.sum(dim=1)
            occ_hat = occ_hat.sum(dim=1)
            outputs = occ_hat.squeeze(0)
            z.append(outputs.detach().cpu().numpy())

        z = np.concatenate(z,axis=0)
        z  = z.astype(np.float64)

        # update the volume
        volume[mask] = z

        # create the masks
        mask_pos = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)
        mask_neg = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)

        
        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i,0])
            yc = int(pts_ids[i,1])
            zc = int(pts_ids[i,2])
            mask_to_see[xc,yc,zc] = False
            if volume[xc,yc,zc] <= 0:
                mask_neg[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True
            if volume[xc,yc,zc] >= 0:
                mask_pos[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True

        # get the new points
        
        new_mask = (mask_neg & (volume>=0) & mask_to_see) | (mask_pos & (volume<=0) & mask_to_see)
        pts_ids = np.argwhere(new_mask).astype(np.int)

    volume[0:padding, :, :] = out_value
    volume[-padding:, :, :] = out_value
    volume[:, 0:padding, :] = out_value
    volume[:, -padding:, :] = out_value
    volume[:, :, 0:padding] = out_value
    volume[:, :, -padding:] = out_value

    # volume[np.isnan(volume)] = out_value
    maxi = volume[~np.isnan(volume)].max()
    mini = volume[~np.isnan(volume)].min()

    if not (maxi > mc_value and mini < mc_value):
        return None

    if return_volume:
        return volume

    # compute the marching cubes
    verts, faces, _, _ = measure.marching_cubes(
            volume=volume.copy(),
            level=mc_value,
            )

    # removing the nan values in the vertices
    values = verts.sum(axis=1)
    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
    mesh.remove_vertices_by_mask(np.isnan(values))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)


    if refine_iter > 0:

        dirs = verts - np.floor(verts)
        dirs = (dirs>0).astype(dirs.dtype)

        mask = np.logical_and(dirs.sum(axis=1)>0, dirs.sum(axis=1)<2)
        v = verts[mask]
        dirs = dirs[mask]

        # initialize the two values (the two vertices for mc grid)
        v1 = np.floor(v)
        v2 = v1 + dirs

        # get the predicted values for both set of points
        v1 = v1.astype(int)
        v2 = v2.astype(int)
        preds1 = volume[v1[:,0], v1[:,1], v1[:,2]]
        preds2 = volume[v2[:,0], v2[:,1], v2[:,2]]

        # get the coordinates in the real coordinate system
        v1 = v1.astype(np.float32)*step + bmin_pad
        v2 = v2.astype(np.float32)*step + bmin_pad

        # tmp mask
        mask_tmp = np.logical_and(
                        np.logical_not(np.isnan(preds1)),
                        np.logical_not(np.isnan(preds2))
                        )
        v = v[mask_tmp]
        dirs = dirs[mask_tmp]
        v1 = v1[mask_tmp]
        v2 = v2[mask_tmp]
        mask[mask] = mask_tmp

        # initialize the vertices
        verts = verts * step + bmin_pad
        v = v * step + bmin_pad

        # iterate for the refinement step
        for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):

            print(f"iter {iter_id}")

            preds = []
            pnts_all = torch.tensor(v, dtype=torch.float, device=device)
            for pnts in tqdm(torch.split(pnts_all,num_pts,dim=0), ncols=100, disable=True):

                
                latent["pos_non_manifold"] = pnts.unsqueeze(0)
                occ_hat = network.from_latent(latent)

                # get class and max non class
                class_dim = 1
                occ_hat = torch.stack([occ_hat[:, class_dim] , occ_hat[:,[i for i in range(occ_hat.shape[1]) if i!=class_dim]].max(dim=1)[0]], dim=1)
                occ_hat = F.softmax(occ_hat, dim=1)
                occ_hat[:, 0] = occ_hat[:, 0] * (-1)
                if class_dim == 0:
                    occ_hat = occ_hat * (-1)


                # occ_hat = -occ_hat.sum(dim=1)
                occ_hat = occ_hat.sum(dim=1)
                outputs = occ_hat.squeeze(0)


                # outputs = network.predict_from_latent(latent, pnts.unsqueeze(0), with_sigmoid=True)
                # outputs = outputs.squeeze(0)
                preds.append(outputs.detach().cpu().numpy())
            preds = np.concatenate(preds,axis=0)

            mask1 = (preds*preds1)>0
            v1[mask1] = v[mask1]
            preds1[mask1] = preds[mask1]

            mask2 = (preds*preds2)>0
            v2[mask2] = v[mask2]
            preds2[mask2] = preds[mask2]

            v = (v2 + v1)/2

            verts[mask] = v

            # keep only the points that needs to be refined
            if refine_threshold is not None:
                mask_vertices = (np.linalg.norm(v2 - v1, axis=1) > refine_threshold)
                # print("V", mask_vertices.sum() , "/", v.shape[0])
                v = v[mask_vertices]
                preds1 = preds1[mask_vertices]
                preds2 = preds2[mask_vertices]
                v1 = v1[mask_vertices]
                v2 = v2[mask_vertices]
                mask[mask] = mask_vertices

                if v.shape[0] == 0:
                    break
                print("V", v.shape[0])

    else:
        verts = verts * step + bmin_pad

    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
    print(f"o3d_verts: {np.asarray(o3d_verts).shape}")
    print(f"o3d_faces: {np.asarray(o3d_faces).shape}")
    if simplification_target is not None and simplification_target > 0:
        mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)
    print(f"mesh: {mesh.dimension}")
    return mesh


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
