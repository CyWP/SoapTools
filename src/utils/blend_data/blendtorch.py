import bpy
import numpy as np
import torch

from bpy.types import Object, Image
from itertools import chain
from typing import Tuple, List, Union, Optional, Any

from ..img import ImageTensor


class BlendTorch:

    @staticmethod
    def tensor2mesh(V: torch.Tensor, F: torch.Tensor, name="TensorMesh"):
        """
        Convert PyTorch tensors V (N,3) and F (M,3) into a Blender mesh
        using foreach_set for maximum speed.
        """
        # ---- ensure CPU + NumPy ----
        if V.is_cuda:
            V = V.cpu()
        if F.is_cuda:
            F = F.cpu()

        V_np = V.detach().contiguous().numpy()
        F_np = F.detach().contiguous().numpy()

        # Ensure correct dtypes
        V_np = V_np.astype(np.float32)
        F_np = F_np.astype(np.int32)

        # ---- Create empty mesh ----
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(len(V_np))
        mesh.loops.add(len(F_np) * 3)
        mesh.polygons.add(len(F_np))

        # ---- Assign vertices ----
        mesh.vertices.foreach_set("co", V_np.ravel())

        # ---- Assign loop vertex indices ----
        mesh.loops.foreach_set("vertex_index", F_np.ravel())

        # ---- Assign polygon sizes and loop start indices ----
        loop_starts = np.arange(0, len(F_np) * 3, 3, dtype=np.int32)
        sizes = np.full(len(F_np), 3, dtype=np.int32)

        mesh.polygons.foreach_set("loop_start", loop_starts)
        mesh.polygons.foreach_set("loop_total", sizes)

        # ---- Finalize ----
        mesh.update()
        return mesh

    @staticmethod
    def tensor2mesh_update(obj: Object, V: torch.Tensor) -> Object:
        """
        Update the vertex positions of an existing Blender mesh object
        using a PyTorch tensor V of shape (N, 3).
        """

        # --- Validate ---
        if obj.type != "MESH":
            raise TypeError("obj must be a Blender MESH")

        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("V must have shape (N, 3)")

        mesh = obj.data
        n_verts = len(mesh.vertices)
        if V.shape[0] != n_verts:
            raise ValueError(
                f"Vertex count mismatch: mesh has {n_verts}, tensor has {V.shape[0]}"
            )

        # --- Move to CPU and prepare numpy array ---
        if V.is_cuda:
            V = V.cpu()

        # ensure contiguous float32
        V_np = V.detach().contiguous().to(torch.float32).numpy()

        # --- Fast update ---
        mesh.vertices.foreach_set("co", V_np.ravel())

        mesh.update()
        return mesh

    @staticmethod
    def mesh2tensor(
        mesh_obj: Object, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract vertex positions and triangulated faces from a Blender mesh.
        """
        if mesh_obj.type != "MESH":
            raise TypeError("Input must be a mesh object")

        mesh = mesh_obj.data
        mesh.calc_loop_triangles()

        vert_count = len(mesh.vertices)
        verts_np = np.empty(vert_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts_np)
        verts = torch.from_numpy(verts_np.reshape(-1, 3)).to(device=device)

        tri_count = len(mesh.loop_triangles)
        faces_np = np.empty(tri_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", faces_np)
        faces = torch.from_numpy(faces_np.reshape(-1, 3)).to(device, dtype=torch.long)

        return verts, faces

    @staticmethod
    def vg2tensor(
        mesh_obj: Object, group_name: str, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        if mesh_obj.type != "MESH":
            raise ValueError("Object must be a mesh.")

        vg = mesh_obj.vertex_groups.get(group_name)
        if vg is None:
            raise ValueError(f"Vertex group '{group_name}' not found.")

        group_idx = vg.index
        nV = len(mesh_obj.data.vertices)

        # Flatten all vertex groups info into numpy arrays
        all_vertex_indices = torch.tensor(
            list(
                chain.from_iterable(
                    [[v.index] * len(v.groups) for v in mesh_obj.data.vertices]
                )
            ),
            device=device,
            dtype=torch.long,
        )
        all_group_indices = torch.tensor(
            list(
                chain.from_iterable(
                    [[g.group for g in v.groups] for v in mesh_obj.data.vertices]
                )
            ),
            device=device,
            dtype=torch.long,
        )
        all_weights = torch.tensor(
            list(
                chain.from_iterable(
                    [[g.weight for g in v.groups] for v in mesh_obj.data.vertices]
                )
            ),
            device=device,
            dtype=torch.float32,
        )

        # Mask only the entries for the target group
        mask = all_group_indices == group_idx
        selected_indices = all_vertex_indices[mask]
        selected_weights = all_weights[mask]
        nz_mask = selected_weights > 0.0
        W, idx = selected_weights[nz_mask], selected_indices[nz_mask]
        vmap = torch.zeros((nV,), device=device, dtype=torch.float32)
        vmap[idx] = W
        return vmap, idx

    @staticmethod
    def tensor2vg(mesh_obj, name, weights, indices=None):
        if mesh_obj.type != "MESH":
            raise TypeError("Object must be a mesh")

        # Normalize if outside [0,1]
        w_min, w_max = weights.min(), weights.max()
        if w_min < 0 or w_max > 1:
            weights = (weights - w_min) / (w_max - w_min)

        # Select non-zero indices if not provided
        if indices is None:
            indices = weights.nonzero(as_tuple=True)[0]

        idx_list = indices.unsqueeze(1).detach().cpu().tolist()
        w_list = weights[indices].detach().cpu().tolist()

        vg = mesh_obj.vertex_groups.new(name=name)

        for i, w in zip(idx_list, w_list):
            vg.add(i, w, "ADD")

        return vg

    @staticmethod
    def vn2tensor(mesh_obj: bpy.types.Object, device: torch.device) -> torch.Tensor:
        mesh = mesh_obj.data

        mesh.calc_loop_triangles()
        n_loops = len(mesh.loops)
        n_verts = len(mesh.vertices)
        loop_normals = np.empty(n_loops * 3, dtype=np.float32)
        mesh.loops.foreach_get("normal", loop_normals)
        loop_normals = loop_normals.reshape(-1, 3)
        loop_vert = np.empty(n_loops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vert)
        vert_normals = np.zeros((n_verts, 3), dtype=np.float32)
        np.add.at(vert_normals, loop_vert, loop_normals)
        norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vert_normals /= norms
        return torch.from_numpy(vert_normals).to(device)

    @staticmethod
    def e2tensor(mesh_obj: Object, device: torch.device) -> torch.Tensor:
        if mesh_obj.type != "MESH":
            raise TypeError("Input must be a mesh object")

        mesh = mesh_obj.data
        mesh.calc_loop_triangles()  # ensure triangulated faces

        # Collect all triangle edges
        tri_count = len(mesh.loop_triangles)
        tris = np.empty(tri_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)
        edges = np.concatenate(
            [
                tris[:, [0, 1]],
                tris[:, [1, 2]],
                tris[:, [2, 0]],
            ],
            axis=0,
        )

        # Sort each edge and remove duplicates
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        return torch.from_numpy(edges).to(device)

    @staticmethod
    def uv2tensor(mesh_obj: Object, uv_map: str, device: torch.device):
        mesh = mesh_obj.data
        uv_layer = mesh.uv_layers.get(uv_map)

        L = len(mesh.loops)

        # vertex indices
        loop_vert = np.empty(L, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vert)

        # UV coords
        uv_flat = np.empty(L * 2, dtype=np.float32)
        uv_layer.data.foreach_get("uv", uv_flat)

        uv_idx = torch.from_numpy(loop_vert).to(device, dtype=torch.long)
        uv_co = torch.from_numpy(uv_flat.reshape(-1, 2)).to(device, dtype=torch.float32)

        return uv_idx, uv_co

    @staticmethod
    def img2tensor(img: bpy.types.Image, device: torch.device):
        w, h = img.size
        arr = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(arr)
        arr = arr.reshape((h, w, 4))
        # arr = np.flip(arr, axis=0).copy()
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return ImageTensor.from_tensor(tensor, device=device)

    @staticmethod
    def tensor2img(tensor: ImageTensor, name: Union[str, List[str]]) -> List[Image]:
        B, C, H, W = tensor.shape
        if isinstance(name, str):
            name = [name]
        if len(name) != B:
            raise ValueError(
                f"The number of names should match the number of images.\nNames: {len(name)},\nImages: {B}"
            )
        imgs = []
        for batch in range(B):
            img_data = tensor[batch].permute(1, 2, 0).cpu().numpy().ravel()
            img = bpy.data.images.new(
                name=name[batch], width=W, height=H, alpha=C == 4, float_buffer=True
            )
            img.pixels.foreach_set(img_data)
            img.update()
            imgs.append(img)
        return imgs
