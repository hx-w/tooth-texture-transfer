# -*- coding: utf-8 -*-

import os
import hashlib
from typing import Tuple
import trimesh
from trimesh import Trimesh
import gradio as gr

import displacement_map as dm


g_base_mesh = trimesh.load('static/constrant.obj')

def hash_cache(mesh: Trimesh) -> Tuple[str, str, bool]:
    os.makedirs('.cache', exist_ok=True)

    # read mesh file and compute hash
    hash_inst = hashlib.md5()
    with open(mesh.metadata['file_path'], 'rb') as f:
        while chunk := f.read(8192):
            hash_inst.update(chunk)
    
    tag = hash_inst.hexdigest()
    cache_dir = os.path.join('.cache', tag)
    if os.path.exists(cache_dir):
        return cache_dir, tag, True

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir, tag, False

def registration(source: Trimesh, target: Trimesh) -> Trimesh:
    source.apply_obb()
    tmat, _ = trimesh.registration.mesh_other(source, target.vertices)
    source.apply_transform(tmat)
    return source

def make_dispmap(msh: str, resol: int, cache_dir: str, tag: str) -> Tuple[dm.DispMap, str]: # dispmap, figpath
    disp_map = dm.DispMap.from_mesh(msh, resol)
    
    mat_filename = os.path.join(cache_dir, tag + '.mat')
    png_filename = os.path.join(cache_dir, tag + '.png')
    
    disp_map.save_mat(mat_filename)
    fig = disp_map.plot(show=False)
    fig.savefig(png_filename, pad_inches=False, bbox_inches='tight')

    return disp_map, png_filename

#!! Important
def procceed(tex_mesh_path: str, smt_mesh_path: str, tex_resol: int, progress=None):
    tex_mesh = trimesh.load(tex_mesh_path)
    smt_mesh = trimesh.load(smt_mesh_path)

    cache_dir, tag, cached = hash_cache(tex_mesh)
    
    ## step - 1 registration
    # progress(0, desc='正在对齐网格')
    # tex_mesh = registration(tex_mesh, g_base_mesh)
    # smt_mesh = registration(smt_mesh, g_base_mesh)
    
    ## step - 2 compute disp map
    # progress(0, desc='正在计算移位贴图')
    try:
        disp_map, tex_filepath = make_dispmap(tex_mesh, tex_resol, cache_dir, tag)
    except Exception as e:
        raise gr.Error(f'移位贴图计算失败：{e}')
    
    ## step - 3 apply disp map
    # progress(0, desc='正在生成网格细节')
    trans_mesh = dm.DispMap.apply_dm(smt_mesh, disp_map)
    
    trans_meshpath = os.path.join(cache_dir, tag + '_trans.obj')
    trans_mesh.export(trans_meshpath)
    
    return tex_filepath, trans_meshpath
