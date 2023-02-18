# ============================================================================
#  bfield_export.py -- Export Blender BField model file for source control
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

import os
import bpy
from mathutils import Vector, Euler

# Blender object attributes added by BField, with default values
ob_bfield_attrs = (
    ('axis', 'X'),
    ('block_type', ''),
    ('cap_units', 'pf'),
    ('capacitance', 1.0),
    ('dx', 1.0),
    ('fmat_name', ''),
    ('ms_rate', 500),
    ('p_avg', False),
    ('p_axis_sign', 1),
    ('p_axis', 'XYZ'),
    ('p_disp_color', (0.75, 0.0, 0.8, 1.0)),
    ('p_disp_is_mesh', False),
    ('p_disp_is_plot', True),
    ('p_disp_pos', 0.5),
    ('p_disp_scale', 256.0),
    ('p_field', 'Electric'),
    ('p_image_alpha', 1.0),
    ('p_legend_loc', 'best'),
    ('p_log', True),
    ('p_mag_scale', 1.0),
    ('p_pixel_rep', 1),
    ('p_plot_scale', 1.0),
    ('p_sfactor', 1),
    ('p_shape', 'Plane'),
    ('p_sum', False),
    ('p_value', 0.0),
    ('p_value3', (0.0, 0.0, 0.0)),
    ('p_verbose', 0),
    ('pml_border', 4),
    ('res_units', 'ohms'),
    ('resistance', 1.0),
    ('s_axis', ''),
    ('s_duration_units', 'sec'),
    ('s_duration', 0.0),
    ('s_excitation', ''),
    ('s_function', ''),
    ('s_hard', False),
    ('s_resistance', 50.0),
    ('s_scale', 1.0),
    ('s_tfall_units', 'ps'),
    ('s_tfall', 10.0),
    ('s_trise_units', 'ps'),
    ('s_trise', 10.0),
    ('s_tstart_units', 'ps'),
    ('s_tstart', 0.0),
    ('snap', False),
    ('snapped_name', ''),
    ('stop_ps', 0.0),
    ('us_poll', 50),
    ('verbose', 0),
)


def repr_vec(vec):
    """Round values in a color or coordinate to 5 places"""
    return tuple([round(x, 5) for x in vec])


def repr_1(x):
    """Round a float value to 5 places"""
    return round(x, 5)


def export_header(out):
    """Write out all material definitions"""
    s = ""
    for mat in bpy.data.materials:
        s += f"""mat '{mat.name}'
"""
        color = mat.diffuse_color
        alpha = color[3]
        if mat.use_nodes:
            node = mat.node_tree.nodes["Principled BSDF"]
            color = node.inputs[0].default_value
            alpha = node.inputs['Alpha'].default_value
        s += f""" color = {repr_vec(color)}
 alpha = {repr_1(alpha)}
 fake = {int(mat.use_fake_user)}
"""
        for attr in ('epr', 'mur', 'sige'):
            if hasattr(mat, attr):
                s += f" {attr} = {getattr(mat, attr):1.5g}\n"
        s += "\n"
    out.write(s)


def export_ob(ob, out):
    """Write out a BField object definition, others ignored"""
    block_type = getattr(ob, 'block_type')
    if block_type is None or block_type == '':
        # ignore non-BField objects
        return

    if ob.rotation_euler != Euler((0.0, 0.0, 0.0), 'XYZ'):
        raise ValueError(f"Object {ob.name} needs Apply Rotation!")
    if ob.scale != Vector((1.0, 1.0, 1.0)):
        raise ValueError(f"Object {ob.name} needs Apply Scale!")

    s = ""
    data = ob.data
    if data and type(data) == bpy.types.Mesh:
        vert_lines = "\n".join(
            [f"  {repr_vec(tuple(v.co))}," for v in data.vertices]
        )

        face_lines = "\n".join(
            [f"  {tuple(p.vertices)}," for p in data.polygons.values()]
        )

        s += f"""mesh '{ob.name}'
 verts = [
{vert_lines}
 ]
 faces = [
{face_lines}
 ]
 loc = {repr_vec(ob.location)}
"""
    ## data.uv_layers["UVMap"].data.items()

    if ob.hide_get():
        s += " hide 1\n"

    m = getattr(ob, 'active_material')
    if m:
        s += f" mat = '{m.name}'\n"

    for attr, default in ob_bfield_attrs:
        ##print(attr)
        if hasattr(ob, attr):
            value = getattr(ob, attr)
            if attr in ('p_disp_color', 'p_value3'):
                value = repr_vec(value)
            if type(value) == type(1.0):
                value = repr_1(value)
            if value != default:
                s += f" {attr} = {repr(value)}\n"

    s += "\n"
    out.write(s)


def export():
    """Export all to a .bf file"""
    cwd, blend_name = os.path.split(bpy.data.filepath)
    name = os.path.splitext(blend_name)[0]
    out_name = f"{name}.bf"
    out_dir = os.path.join(cwd, 'export')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = f"{out_dir}/{out_name}"
    out = open(out_path, "w")
    out.write(
        f"""BField model {name}
version 1

"""
    )
    export_header(out)
    for ob in bpy.data.objects:
        export_ob(ob, out)
    out.close()
    print(f"Wrote {out_path}.")


export()
