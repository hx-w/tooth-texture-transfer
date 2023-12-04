# -*- coding: utf-8 -*-

import os
import math
from typing import Tuple, List

import trimesh
from trimesh import Trimesh
import numpy as np
from scipy.io import savemat, loadmat
import numpy_indexed as npi
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import parameterizer as prmter


class VMap:
    def __init__(self, data: np.array=None):
        self._vmatrix = data

    @classmethod
    def load_from_mat(cls, filename: str) -> object:
        if not os.path.exists(filename) or not filename.endswith('.mat'):
            raise RuntimeError(f'filename error: {filename}')
 
        tag = cls.__str__()
        mats = loadmat(filename)
        
        if tag not in mats.keys():
            raise RuntimeError(f'npz content not found: {mats.keys()}')

        return cls(mats[tag])        

    def save_mat(self, filename: str):
        if self._vmatrix is None:
            raise RuntimeError('matrix is None')
        
        savemat(filename, {self.__str__(): self._vmatrix})

    def plot(self, show: bool=True):
        if self._vmatrix is None:
            raise RuntimeError('matrix is None')
        
        heatmat = self._vmatrix
        if len(self._vmatrix.shape) == 3:
            heatmat = np.linalg.norm(self._vmatrix, axis=2)
        
        # plt.figure(figsize = (10, 10))
        heatplot = sns.heatmap(heatmat, cmap='coolwarm', xticklabels=False, yticklabels=False,  cbar=False)
        if show:
            plt.show()
        
        plt.close()
        return heatplot.get_figure()

    @classmethod
    def __str__(cls):
        return 'Virtual Map'
    

class DispMap(VMap):
    @classmethod
    def __str__(cls):
        return 'Displacement Map'

    @staticmethod
    def split_mesh_with_XOY(msh: Trimesh) -> Tuple[Trimesh, Trimesh]: # pos, neg
        pos_vinds = np.where(msh.vertices[:, 2] >= 0)[0]
        pos_finds = list(filter(lambda fi: np.isin(msh.faces[fi], pos_vinds).all(), range(msh.faces.shape[0])))
        neg_finds = npi.difference(range(msh.faces.shape[0]), pos_finds)
        pos_msh, neg_msh = msh.submesh([pos_finds, neg_finds], append=False)
        return pos_msh, neg_msh

    @staticmethod
    def merge_meshes(mshes: List[Trimesh]) -> Trimesh:
        vertices_list = [msh.vertices for msh in mshes]
        faces_list = [msh.faces for msh in mshes]
        
        faces_offset = np.cumsum([v.shape[0] for v in vertices_list])
        faces_offset = np.insert(faces_offset, 0, 0)[:-1]

        vertices = np.vstack(vertices_list)
        faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])

        return trimesh.Trimesh(vertices, faces)

    @staticmethod
    def __smooth_part(msh: Trimesh, hard: bool=True, pieces: int=12) -> Trimesh:
        bound = msh.bounds
        neg_vinds = np.where(
            (msh.vertices[:, 2] < 0) | \
            (msh.vertices[:, 0] < bound[0][0] + (bound[1][0] - bound[0][0]) / pieces) | \
            (msh.vertices[:, 0] > bound[1][0] - (bound[1][0] - bound[0][0]) / pieces) | \
            (msh.vertices[:, 1] < bound[0][1] + (bound[1][1] - bound[0][1]) / pieces) | \
            (msh.vertices[:, 1] > bound[1][1] - (bound[1][1] - bound[0][1]) / pieces)
        )[0]
        lap = trimesh.smoothing.laplacian_calculation(msh, equal_weight=False, pinned_vertices=neg_vinds)

        if hard:
            trimesh.smoothing.filter_mut_dif_laplacian(msh, lamb=0.7, iterations=100, laplacian_operator=lap)
        else:
            trimesh.smoothing.filter_taubin(msh)

        return msh

    '''
    compute displacement map from a mesh
    '''
    @classmethod
    def from_mesh(cls, msh: Trimesh, resol: int) -> VMap:
        sm_msh = msh.copy()
        
        # smooth only a part
        sm_msh = cls.__smooth_part(sm_msh)
        
        pos_msh, _ = cls.split_mesh_with_XOY(msh)
        pos_sm_msh, _ = cls.split_mesh_with_XOY(sm_msh)
        
        str_msh, _ = prmter.FixBoundaryPrmter.recon(pos_msh, resol)
        str_sm_msh, _ = prmter.FixBoundaryPrmter.recon(pos_sm_msh, resol)
        
        disp_matrix = str_msh.vertices - str_sm_msh.vertices
        disp_matrix = disp_matrix.reshape(resol, resol, 3)
        
        return cls(disp_matrix)

    '''
    assmue `msh` has already been registrated
    '''
    @classmethod
    def apply_dm(cls, msh: Trimesh, dm: VMap) -> Trimesh:
        resol = dm._vmatrix.shape[0]
        
        pos_msh, neg_msh = cls.split_mesh_with_XOY(msh)
        _, param_msh = prmter.FixBoundaryPrmter.recon(pos_msh, resol)
        
        # deform `pos_msh` according to `param_msh` and `dm`
        for vi in tqdm(range(pos_msh.vertices.shape[0]), desc='双线性插值优化中'):
            shift_loc = (param_msh.vertices[vi][:2] + np.array([1., 1.])) / 2.
            real_loc = [min(99., max(0., 100. * loc)) for loc in shift_loc]
            
            # bilinear interplot
            ## A(floor row, floor col), B(floor row, ceil col)
            ## C(ceil row, floor col), D(ceil row, ceil col)
            grids = [
                [math.floor(real_loc[0]), math.floor(real_loc[1])],
                [math.floor(real_loc[0]), math.ceil(real_loc[1])],
                [math.ceil(real_loc[0]), math.floor(real_loc[1])],
                [math.ceil(real_loc[0]), math.ceil(real_loc[1])]
            ]
            
            ## weights = rectangle area
            ws = [np.abs(grid[0] - real_loc[0]) * np.abs(grid[1] - real_loc[1])  for grid in grids]

            ## sum 4 triangle corner with ws
            disp = np.sum([ws[ind] * dm._vmatrix[grids[ind][0], grids[ind][0], :] for ind in range(4)], axis=0)
            
            pos_msh.vertices[vi] += disp
        
        # combind pos and neg
        result = cls.merge_meshes([pos_msh, neg_msh])
        result.remove_degenerate_faces()
        result.remove_duplicate_faces()
        
        trimesh.smoothing.filter_taubin(result)
        
        trimesh.repair.fill_holes(result)
        result.vertex_normals
        
        return result
