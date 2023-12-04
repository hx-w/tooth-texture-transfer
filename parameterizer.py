# -*- coding: utf-8 -*-

from typing import Tuple
from collections import defaultdict

from trimesh import Trimesh
import numpy as np
import numpy_indexed as npi
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class Prmter:
    @classmethod
    def recon(cls, msh: Trimesh, resol: int=100) -> Tuple[Trimesh, Trimesh]:
        if msh.is_watertight:
            raise ValueError('Mesh shound be OPEN manifold')
        
        return cls._recon_impl(msh, resol)
        
    @classmethod
    def _recon_impl(cls, msh: Trimesh, resol: int) -> Tuple[Trimesh, Trimesh]:
        raise NotImplementedError('Virtual class method')

    @staticmethod
    def get_boundary(mesh, close_paths=True):
        edge_set = set()
        boundary_edges = set()

        # Iterate over all edges, as tuples in the form (i, j) (sorted with i < j to remove ambiguities).
        # For each edge, three cases are possible:
        # 1. The edge has never been visited before. In this case, we can add it to the edge set and as a boundary
        #    candidate as well.
        # 2. The edge had already been visited once. We want to keep it into the set of all edges but remove it from the
        #    boundary set.
        # 3. The edge had already been visited at least twice. This is generally an indication that there is an issue with
        #    the mesh. More precisely, it is not a manifold, and boundaries are not closed-loops.
        for e in map(tuple, mesh.edges_sorted):
            if e not in edge_set:
                edge_set.add(e)
                boundary_edges.add(e)
            elif e in boundary_edges:
                boundary_edges.remove(e)
            else:
                raise RuntimeError(f"The mesh is not a manifold: edge {e} appears more than twice.")

        neighbours = defaultdict(lambda: [])
        for v1, v2 in boundary_edges:
            neighbours[v1].append(v2)
            neighbours[v2].append(v1)

        boundary_paths = []

        while len(boundary_edges) > 0:
            # Given the set of remaining boundary edges, get one of them and use it to start the current boundary path.
            # In the sequel, v_previous and v_current represent the edge that we are currently processing.
            v_previous, v_current = next(iter(boundary_edges))
            boundary_vertices = [v_previous]

            # Keep iterating until we close the current boundary curve (the "next" vertex is the same as the first one).
            while v_current != boundary_vertices[0]:
                # We grow the path by adding the vertex "v_current".
                boundary_vertices.append(v_current)

                # We now check which is the next vertex to visit.
                v1, v2 = neighbours[v_current]
                if v1 != v_previous:
                    v_current, v_previous = v1, v_current
                elif v2 != v_previous:
                    v_current, v_previous = v2, v_current
                else:
                    # This line should be un-reachable. I am keeping it only to detect bugs in case I made a mistake when
                    # designing the algorithm.
                    raise RuntimeError(f"Next vertices to visit ({v1=}, {v2=}) are both equal to {v_previous=}.")

            if close_paths:
                boundary_vertices.append(boundary_vertices[0])

            # "Convert" the vertices from indices to actual Cartesian coordinates.
            boundary_paths.append(boundary_vertices)

            # Remove all boundary edges that were added to the last path.
            boundary_edges = set(e for e in boundary_edges if e[0] not in boundary_vertices)

        return boundary_paths

    @classmethod
    def _get_laplacian_weights(cls, msh: Trimesh, inn_verts: np.array, bnd_verts: np.array) -> csc_matrix:
        def weights_for_edge(edge: list) -> float:
            adj_list_s = msh.vertex_neighbors[edge[0]]
            adj_list_b = msh.vertex_neighbors[edge[1]]
            adj_vts = npi.intersection(adj_list_s, adj_list_b)
            # assert len(adj_vts) == 2, 'not a manifold'
            # compute cotangent weight of edge
            ang1 = cls.mesh_vert_angle(msh, adj_vts[0], *edge)
            ang2 = cls.mesh_vert_angle(msh, adj_vts[1], *edge)
            _w = (cls.math_cot(ang1) + cls.math_cot(ang2)) / 2
            return -_w

        # sparse matrix index
        sp_row = np.array([], dtype=int)
        sp_col = np.array([], dtype=int)
        sp_data = np.array([], dtype=float)
        mtx_diag = np.zeros(len(msh.vertices))
        # generate
        _weights = list(map(weights_for_edge, msh.face_adjacency_edges))
        # update diag
        for idx, edge in enumerate(msh.face_adjacency_edges):
            mtx_diag[edge[0]] += -_weights[idx]
            mtx_diag[edge[1]] += -_weights[idx]
        
        # transpose indices
        _indices = msh.face_adjacency_edges.T
        sp_row = np.hstack([sp_row, _indices[0], _indices[1]])
        sp_col = np.hstack([sp_col, _indices[1], _indices[0]])
        sp_data = np.hstack([sp_data, _weights, _weights])

        # handle diag sparse index
        # all vertices in msh with order {INNER, BOUND}
        sp_diag_index = np.append(inn_verts, bnd_verts)
        sp_row = np.hstack([sp_row, sp_diag_index])
        sp_col = np.hstack([sp_col, sp_diag_index])
        sp_diag_data = [mtx_diag[v] for v in sp_diag_index]
        sp_data = np.hstack([sp_data, sp_diag_data])

        sp_weights = csc_matrix((sp_data, (sp_row, sp_col)), dtype=float)
        return sp_weights

    @classmethod
    def _solve_equation(cls, sp_weights: csc_matrix, f_B: np.array, inn_verts: np.array, bnd_verts: np.array) -> np.array:
        _mid = sp_weights[inn_verts, ...]
        sp_weights_II = _mid[..., inn_verts]
        sp_weights_IB = _mid[..., bnd_verts]

        assert sp_weights_IB.shape[1] == len(f_B), f'L_IB({sp_weights.shape[1]}) * f_B({len(f_B)}) illegal'

        f_I = spsolve(sp_weights_II, -sp_weights_IB * f_B)
        return f_I

    '''
    since the low efficiency of indicing Trimesh.edges_unique_length
    need to return edge length in mesh
    '''
    @staticmethod
    def mesh_vert_dist(msh: Trimesh, vidx1: int, vidx2: int) -> float:
        return np.linalg.norm(msh.vertices[vidx1] - msh.vertices[vidx2])

    '''
    return angle of two vectors conformed by three vertices
    '''
    @staticmethod
    def mesh_vert_angle(msh: Trimesh, mid: int, start: int, end: int) -> float:
        vec1 = msh.vertices[start] - msh.vertices[mid]
        vec2 = msh.vertices[end] - msh.vertices[mid]
        return np.arccos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


    @staticmethod
    def math_cot(angle: float) -> float:
        return np.cos(angle) / np.sin(angle)

    '''
    return cross product of vec: vec1 - vec2 and vec3 - vec2
    '''
    @staticmethod
    def mesh_vec_cross(vec1: tuple, vec2: tuple, vec3: tuple) -> float:
        vt1 = (vec1[0] - vec2[0], vec1[1] - vec2[1])
        vt2 = (vec3[0] - vec2[0], vec3[1] - vec2[1])
        return vt1[0] * vt2[1] - vt2[0] * vt1[1]

    '''
    return triangle area with vertices: vec1, vec2, vec3
    '''
    @staticmethod
    def mesh_trias_area(vec1: np.array, vec2: np.array, vec3: np.array) -> float:
        _a = np.linalg.norm(vec2 - vec1)
        _b = np.linalg.norm(vec3 - vec1)
        _c = np.linalg.norm(vec3 - vec2)
        _s = (_a + _b + _c) / 2
        return (_s * (_s - _a) * (_s - _b) * (_s - _c)) ** 0.5

    @staticmethod
    def rotate_90(pnts: list, times: int) -> list:
        # pnts is ndarray shape (n, 2)
        # rotate theta angle around (0, 0)
        rot_mat = np.array([[0, -1], [1, 0]])
        for _ in range(times):
            pnts = np.dot(pnts, rot_mat)
        return pnts
    
    '''
    build param mesh by inverse mapping
    assume Z=0 in param mesh
    '''
    @classmethod
    def build_param_mesh(cls, msh: Trimesh, inn_verts: list, bnd_verts: list, f_I: list, f_B: list) -> Trimesh:
        len_inn, len_bnd = len(inn_verts), len(bnd_verts)
        param_bnd_verts = [v + len_inn for v in range(len_bnd)]
        inv_mapping = dict(zip(bnd_verts, param_bnd_verts))
        param_inn_verts = [v for v in range(len_inn)]
        inv_mapping.update(zip(inn_verts, param_inn_verts))
        param_tot = np.append(f_I, f_B, axis=0)
        # param_tot add new column Z=0
        param_tot = np.hstack([param_tot, np.zeros((len(param_tot), 1))])

        param_mesh = Trimesh(
            vertices=[param_tot[inv_mapping[i]] for i in range(len_inn + len_bnd)],
            faces=msh.faces.copy()
        )
        param_mesh.remove_degenerate_faces()
        param_mesh.remove_duplicate_faces()
        param_mesh.remove_infinite_values()
        return param_mesh

    @classmethod
    def build_str_mesh_custom(cls, uns_mesh: Trimesh, param_mesh: Trimesh, sample_pnts: list, scale: float=2.) -> Trimesh:
        square_nums = len(sample_pnts)
        assert square_nums > 4, 'sample points illegal'
        sample_nums = int(square_nums ** 0.5)
        # flatten numpy elements to list will accelerate in cycle
        flt_faces = param_mesh.faces.tolist()
        flt_area_faces = param_mesh.area_faces.tolist()
        str_mesh = Trimesh()

        sample_trias = param_mesh.nearest.on_surface(sample_pnts)
        sample_trias = sample_trias[2].tolist()
        spot_trias = list(map(lambda tri: flt_faces[tri], sample_trias))
        vijk_areas = [
            [
                cls.mesh_trias_area(
                    sample_pnts[idx],
                    param_mesh.vertices[spot_trias[idx][1]],
                    param_mesh.vertices[spot_trias[idx][2]]
                ),
                cls.mesh_trias_area(
                    param_mesh.vertices[spot_trias[idx][0]],
                    sample_pnts[idx],
                    param_mesh.vertices[spot_trias[idx][2]]
                ),
                cls.mesh_trias_area(
                    param_mesh.vertices[spot_trias[idx][0]],
                    param_mesh.vertices[spot_trias[idx][1]],
                    sample_pnts[idx],
                )
            ]
            for idx in range(square_nums)
        ]

        str_pnts = [
            (
                vijk_areas[idx][0] * uns_mesh.vertices[spot_trias[idx][0]] +
                vijk_areas[idx][1] * uns_mesh.vertices[spot_trias[idx][1]] +
                vijk_areas[idx][2] * uns_mesh.vertices[spot_trias[idx][2]]
            ) / flt_area_faces[sample_trias[idx]]
            for idx in range(square_nums)
        ]

        half_trias1 = [
            [ir * sample_nums + ic, ir * sample_nums + ic - sample_nums, ir * sample_nums + ic - 1]
            for ir in range(1, sample_nums) for ic in range(1, sample_nums)
        ]
        half_trias2 = [
            [ir * sample_nums + ic - 1, ir * sample_nums + ic - sample_nums, ir * sample_nums + ic - sample_nums - 1]
            for ir in range(1, sample_nums) for ic in range(1, sample_nums)
        ]

        str_mesh.vertices = str_pnts
        str_mesh.faces = np.vstack([half_trias1, half_trias2])

        str_mesh.remove_infinite_values()
        str_mesh.remove_degenerate_faces()
        str_mesh.remove_unreferenced_vertices()
        str_mesh.fill_holes()
        str_mesh.fix_normals()

        return str_mesh

    '''
    build str mesh by sample vertices on param mesh
    '''
    @classmethod
    def build_str_mesh(cls, uns_mesh: Trimesh, param_mesh: Trimesh, sample_nums: int=50, scale: float=2.) -> Trimesh:
        assert sample_nums > 2, 'sample_nums too small'
        # flatten numpy elements to list will accelerate in cycle
        sample_pnts = []
        for ic in range(sample_nums):
            for ir in range(sample_nums):
                sample_pnts.append([-scale * ir / (sample_nums - 1) + scale / 2, -scale * ic / (sample_nums - 1) + scale / 2,  0.])

        return cls.build_str_mesh_custom(uns_mesh, param_mesh, sample_pnts, scale)

class FixBoundaryPrmter(Prmter):
    @classmethod
    def _recon_impl(cls, msh: Trimesh, resol: int) -> Tuple[Trimesh, Trimesh]:
        if msh.centroid[0] > 1e-1:
            raise RuntimeError(f'Mesh centriod is not at (0, 0, 0), which is {msh.centroid}')
        
        
        ## step - 1, get mesh boundary and inner vertices
        bnd_verts = cls.get_boundary(msh, close_paths=False)
        if len(bnd_verts) > 1:
            raise RuntimeError(f'Mesh has more than 1 boundary: {len(bnd_verts)}')

        bnd_verts = np.array(bnd_verts[0])
        inn_verts = npi.difference(msh.face_adjacency_edges.flatten(), bnd_verts)
        
        ## step - 2, get corner pivots
        pivots = cls._find_pivots(msh, bnd_verts)

        ## step - 3, mapping boundary
        f_B, bnd_verts = cls._map_boundary(msh, bnd_verts, pivots)
        
        ## step - 4, get laplacian weights
        sp_weights = cls._get_laplacian_weights(msh, inn_verts, bnd_verts)
        
        ## step - 5, solve equation
        f_I = cls._solve_equation(sp_weights, f_B, inn_verts, bnd_verts)
        
        ## step - 6, reconstruct
        param_msh = cls.build_param_mesh(msh, inn_verts, bnd_verts, f_I, f_B)
        str_msh = cls.build_str_mesh(msh, param_msh, resol)
        
        return str_msh, param_msh
    
    @staticmethod
    def _find_pivots(msh: Trimesh, vert_inds: np.array) -> list:
        # | ind | x_pos | y_pos |
        ind_pos = np.concatenate([vert_inds.reshape(-1, 1), msh.vertices[vert_inds, :2]], axis=1)
        y_mean = np.mean(ind_pos[:, 2])

        # max_x, max_y, min_x, min_y
        bound = [np.max(ind_pos[:, 1]), np.max(ind_pos[:, 2]), np.min(ind_pos[:, 1]), np.min(ind_pos[:, 2])]
        _k = [(bound[1] - y_mean) / bound[2], (bound[1] - y_mean) / bound[0], (bound[3] - y_mean) / bound[0], (bound[3] - y_mean) / bound[2]]
 
        # pivots order: (min_x, max_y), (max_x, max_y), (min_x, min_y), (max_x, min_y)
        pivots = [-1, -1, -1, -1]
        
        def __dist(k: float, x: float, tar: float) -> float:
            return np.abs(k * x + y_mean - tar)
        
        for row in range(ind_pos.shape[0]):
            k_ind = int(ind_pos[row, 2] < 0) * 2 + np.abs(int(ind_pos[row, 1] >= 0) - int(ind_pos[row, 2] < 0))
            
            dist = __dist(_k[k_ind], ind_pos[row, 1], ind_pos[row, 2])
            if pivots[k_ind] < 0 or dist < __dist(_k[k_ind], ind_pos[pivots[k_ind], 1], ind_pos[pivots[k_ind], 2]):
                pivots[k_ind] = row

        return [int(ind_pos[v, 0]) for v in pivots]

    @classmethod
    def _map_boundary(cls, msh: Trimesh, bnd_verts: np.array, pivots: list, scale: float=2.) -> list:
        # find pivots[0] in bnd_verts, and shift it to the first with rotate
        _bak = bnd_verts.copy()
        bnd_verts = np.roll(bnd_verts, -np.where(bnd_verts == pivots[0])[0][0])
        
        # split bnd_verts into 4 parts
        splitted = np.split(bnd_verts, [
            np.where(bnd_verts == pivots[1])[0][0],
            np.where(bnd_verts == pivots[2])[0][0],
            np.where(bnd_verts == pivots[3])[0][0],
        ])

        if 0 in [len(sub_bnd) for sub_bnd in splitted]:
            # reverse order of _bak
            return cls._map_boundary(msh, _bak[::-1], pivots, scale)

        # empty ndarray with shape (n, 2)
        f_B = np.empty((0, 2))
        for i in range(4):
            sub_f_B = cls._mapping_boundary_line(msh, splitted[i], splitted[(i + 1) % 4][0], scale)
            f_B = np.append(f_B, cls.rotate_90(sub_f_B, i), axis=0)

        return f_B, bnd_verts

    @classmethod
    def _mapping_boundary_line(cls, msh: Trimesh, sub_bnds: list, last_ind: int, scale: float) -> list:
        '''
        将边缘分为四部分，映射其中一部分，都映射到(-scale/2, scale/2) 和 (scale/2, scale/2)之间
        之后再旋转即可拼凑出四部分映射
        '''
        sub_bnds = np.append(sub_bnds, last_ind)
        bnd_size = len(sub_bnds)
        lengths = [cls.mesh_vert_dist(msh, *sub_bnds[i:i+2]) for i in range(bnd_size - 1)]
        lengths = np.array(lengths) / sum(lengths)
        _acc_lens = np.cumsum(lengths)
        _acc_lens -= _acc_lens[0]

        # mapping lengths to (-scale/2, scale/2) and (scale/2, scale/2)
        _x = _acc_lens * scale - scale / 2
        _y = np.ones(bnd_size) * scale / 2

        return np.array(list(zip(_x, _y)))
