# ============================================================================
#  lc_arrows.py -- Show LC output as arrows, like volume probes
#
#  Copyright 2023 Scott Forbes
#
# This file is part of BField.
# BField is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# BField is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# You should have received a copy of the GNU General Public License along
# with BField. If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

# Load and run this in Blender as a text file.

import bpy
from mathutils import Vector, Matrix
from bfield import IVector, StandardCollection, z0, arrow_base, arrow_scale
import numpy as np
import re
import math as ma

bfield_dir = "/Users/scott/github/bfield"

# test parameters
if 1:
    test_name = "small"
    N = IVector(8, 4, 4)  # grid dimensions, in cells
    dx = 0.001  # cell spacing, m
    nsteps = 3  # number of time steps to run
    zo0 = 1  # Z start cell
    nzo = 3  # Z number of cells
elif 0:
    test_name = "zcoax"
    N = IVector(12, 12, 12)  # grid dimensions, in cells
    dx = 0.001  # cell spacing, m
    nsteps = 5  # number of time steps to run
    zo0 = 4  # Z start cell
    nzo = 5  # Z number of cells


arrowPat = r"LC_[EH]([0-9][0-9][0-9])([0-9][0-9][0-9])([0-9][0-9][0-9])"
arrow_min = 10 ** (-2 * arrow_base)  # so scale doesn't go negative


def read_lc_data(field, step):
    """read LC probe-out files which are in k,j,i order, no PML"""
    tdir = f"{bfield_dir}/lc_output/{test_name}/"
    data = np.zeros((3, N.k, N.j, N.i))

    for axis in range(3):
        for k in range(zo0, zo0 + nzo):
            fname = f"{tdir}{field}z{k}{'xyz'[axis]}{step:03d}.out"
            raw = np.genfromtxt(fname, delimiter=(12, 12, 12, 12, 12, 12))
            trimmed = raw.reshape(-1)[: N.i * N.j]
            data[axis, k, :] = trimmed.reshape(N.j, N.i)

    return data


def create_lc_arrows(field):
    """Create a field of E or H arrows from LC output files"""

    print(f"Creating LC arrows for {field}")
    scn = bpy.context.scene
    objs = bpy.data.objects
    bmats = bpy.data.materials

    coll = StandardCollection(f"LC_{field}").get()
    coll.hide_viewport = False

    # color is green or purple
    mname = f"LC_{field}"
    mat = bpy.data.materials.get(mname)
    if mat is None:
        mat = bpy.data.materials.new(name=mname)
        color = ((0, 1.0, 0, 1.0), (1.0, 0, 1.0, 1.0))[field == 'H']
        mat.diffuse_color = color

    # find existing BField-probe arrow objects
    bf_arrows = []
    for ob in objs:
        name = ob.name
        if len(name) == 10 and name[0] == field and name[1:].isdecimal():
            bf_arrows.append(ob)
    if len(bf_arrows) == 0:
        raise RuntimeError(f"{field} arrow objects not found")
    probe = bf_arrows[0].parent

    # find the Tmp collection, used to delete all arrows
    tmpc = bpy.data.collections.get('Tmp')
    if not tmpc:
        raise RuntimeError(f"failed to create Tmp collection for {ob.name}")

    # duplicate the BField arrows as LC arrows, but change color
    mesh = bpy.data.meshes.get('Arrow')
    if mesh is None:
        raise RuntimeError("missing common 'Arrow' mesh")

    arrows = []
    for bfa in bf_arrows:
        name = f"LC_{bfa.name}"
        arrow = bpy.data.objects.new(name, mesh)
        arrow.location = bfa.location
        ##arrow.parent = ob
        coll.objects.link(arrow)
        tmpc.objects.link(arrow)
        if len(arrow.material_slots) == 0:
            arrow.data.materials.append(mat)
        arrow.material_slots[0].link = 'OBJECT'
        arrow.material_slots[0].material = mat
        arrows.append(arrow)

        # these used by FieldObjectPanel.draw()
        arrow.p_sfactor = probe.p_sfactor
        arrow.p_log = probe.p_log
        arrow.p_mag_scale = probe.p_mag_scale

    # animate size and direction of LC arrows
    for step in range(nsteps):
        scn.frame_set(step)
        data = read_lc_data(field, step)

        for arrow in arrows:
            m = re.match(arrowPat, arrow.name)
            i, j, k = [int(x) for x in m.groups()]
            arrow.rotation_euler.zero()
            x, y, z = data[:, k, j, i]
            r2 = x * x + y * y + z * z
            if field == 'H':
                r2 *= z0 * z0
            name = arrow.name
            ##if name.startswith('LC_E001006006'):
            ##    print(f"{step}: {name} [{i},{j},{k}] "
            ##          f"{x} {y} {z}, {r2=}")
            r2 *= arrow.p_mag_scale**2
            if r2 > arrow_min:
                r = ma.sqrt(r2)
                if arrow.p_log:
                    r = arrow_scale * (ma.log10(r) + arrow_base)
                else:
                    if r > 30:
                        r = 30
                r *= arrow.p_sfactor
                arrow.scale = (r, r, r)
                M = Matrix(((x, 0, 0), (y, 0, 0), (z, 0, 0)))
                # rely on rotate() to normalize matrix
                arrow.rotation_euler.rotate(M)
            else:
                arrow.scale = (0, 0, 0)
            arrow.keyframe_insert(data_path='rotation_euler')
            arrow.keyframe_insert(data_path='scale')


create_lc_arrows('E')
create_lc_arrows('H')
