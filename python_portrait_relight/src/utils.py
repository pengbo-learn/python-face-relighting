# -*- coding: utf-8 -*-
import numpy as np

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2) + 1e-6
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def frontalize(vertices, npy_path='data/uv-data/canonical_vertices.npy'):
    '''copy from PRNet, edit npy path. rotate 3d face points to frontal pose.'''
    canonical_vertices = np.load(npy_path)
    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices, rcond=None)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)
    return front_vertices


