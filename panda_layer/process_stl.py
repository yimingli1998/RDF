
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import trimesh
import glob
import os


for i in range(9):
    mesh_path = os.path.dirname(os.path.realpath(__file__)) + f"/meshes/visual/link{i}_vis.stl"
    mesh = trimesh.load(mesh_path)
    mesh.show()
    vertices = mesh.vertices
