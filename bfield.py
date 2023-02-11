# ============================================================================
#  bfield.py -- Blender AddOn for FDTD Electromagnetic Field Solver
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

bl_info = {
    "name": "FDTD Field Sim",
    "author": "Scott Forbes",
    "version": (1, 0),
    "blender": (3, 4, 0),
    "location": "View3D > Add > Mesh > FDTD",
    "description": "Adds FDTD simulation and field grids to scene",
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh",
}

import bpy
import bpy.props as bp
import bgl
import blf
import numpy as np
import math as ma
from mathutils import Vector, Matrix, Euler
from bpy_extras import view3d_utils
from bpy_extras.node_shader_utils import PrincipledBSDFWrapper
import sys
import os
import time
import re
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

cwd = bpy.path.abspath("//")
sys.path.append(cwd)
print(f"{cwd=}")
if 'siunits' in sys.modules:
    ##print("reloading module siunits")
    del sys.modules['siunits']
import siunits as si

is_linux = os.uname()[0] == 'Linux'

timeout = 2500  # sec, server communication

c0 = 2.998e8  # m/s speed of light
z0 = 376.73  # ohms free space impedance
e0 = 8.854e-12  # F/m free space permittivity
mm = 0.001  # m/mm

time_units = {
    'sec': 1.0,
    'ms': 1e-3,
    'us': 1e-6,
    'ns': 1e-9,
    'ps': 1e-12,
    'fs': 1e-15,
}
res_units = {'ohms': 1.0, 'K': 1e3, 'M': 1e6}
cap_units = {'uf': 1e-6, 'nf': 1e-9, 'pf': 1e-12}


class StandardCollection:
    """Name of layer collection

    Must be exactly one of each visible. Will be created if needed.
    """

    def __init__(self, base_name):
        self.base_name = base_name
        self._name = None

    def name(self):
        scene = bpy.context.scene
        coll_names = [
            cn
            for cn in scene.collection.children.keys()
            if cn.startswith(self.base_name)
        ]
        if self._name is None or not self._name in coll_names:
            if len(coll_names) == 1:
                self._name = coll_names[0]
            elif len(coll_names) == 0:
                c = bpy.data.collections.new(self.base_name)
                scene.collection.children.link(c)
                self._name = c.name
            else:
                raise RuntimeError(
                    f"Expected a single {self.base_name} collection in scene."
                    f" Found {coll_names}"
                )
        ##print(f"StandardCollection {self.base_name}: {self._name}")
        return self._name

    def get(self):
        return bpy.data.collections[self.name()]


coll_main = StandardCollection("Main")
coll_snap = StandardCollection("Snap")
coll_plane = StandardCollection("Plane")
coll_E = StandardCollection("E")
coll_H = StandardCollection("H")

# sim states
STOPPED = 0
INITIALIZING = 1
RUNNING = 3
PAUSED = 4

sims = {}  # dictionary of FDTD simulations, indexed by scene
field_operator = None


def tu(ob, name):
    return getattr(ob, name) * time_units[getattr(ob, name + 'Units')]


def get_mouse_3d(event):
    """3D mouse position display

    From users lemon, batFINGER at stackexchange.
    """

    # get the mouse position thanks to the event
    mouse_pos = [event.mouse_region_x, event.mouse_region_y]

    # contextual active object, 2D and 3D regions
    object = bpy.context.object
    region = bpy.context.region
    space_data = bpy.context.space_data
    loc = None
    if space_data:
        region3D = space_data.region_3d

        # the direction indicated by the mouse position from the current view
        view_vector = view3d_utils.region_2d_to_vector_3d(
            region, region3D, mouse_pos
        )
        # the 3D location in this direction
        loc = view3d_utils.region_2d_to_location_3d(
            region, region3D, mouse_pos, view_vector
        )
        # the 3D location converted in object local coordinates
        ##loc = object.matrix_world.inverted() * loc
    else:
        print(f"get_mouse_3d: {space_data=} {object=} {region=} {mouse_pos=}")
    return loc


class IVector:
    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

    def __repr__(self):
        return f"({self.i}, {self.j}, {self.k})"


class IGrid:
    def __init__(self, V, dx):
        hdx = dx / 2
        self.i = ma.floor((V.x + hdx) / dx)
        self.j = ma.floor((V.y + hdx) / dx)
        self.k = ma.floor((V.z + hdx) / dx)

    def __repr__(self):
        return f"({self.i}, {self.j}, {self.k})"


def fv(V):
    return f'({", ".join(["% 7.3f"]*len(V))})' % tuple(V)


def gv(V):
    return f'({", ".join(["% 9.3g"]*len(V))})' % tuple(V)


def bounds(ob):
    """Get an object's bounds box dimensions in global coordinates"""

    # bound_box only usable when unrotated & unscaled
    if ob.rotation_euler != Euler((0.0, 0.0, 0.0), 'XYZ'):
        print(f"**** Object {ob.name} needs Apply Rotation!")
    if ob.scale != Vector((1.0, 1.0, 1.0)):
        print(f"**** Object {ob.name} needs Apply Scale!")

    mods_off = []
    for mod in ob.modifiers:
        if mod.type == 'ARRAY':
            ##print(f" temp turning off {ob.name} array {mod.name}")
            mods_off.append(mod.name)
            mod.show_viewport = False
    if len(mods_off) > 0:
        bpy.context.view_layer.update()

    bb = ob.bound_box
    Bs = ob.matrix_world @ Vector(bb[0])
    Be = ob.matrix_world @ Vector(bb[6])

    if len(mods_off) > 0:
        for name in mods_off:
            ob.modifiers[name].show_viewport = True
        bpy.context.view_layer.update()

    return Bs, Be


def overlap(ob1, ob2, extend=0):
    """Return a (Bs, Be) bound box trimmed to both objects, or None"""

    Ex = Vector((extend, extend, extend))
    B1l, B1h = bounds(ob1)
    B2l, B2h = bounds(ob2)
    ##print("overlap():", fv(B1l), fv(B1h))
    ##print("          ", fv(B2l), fv(B2h))
    B2l -= Ex
    B2h += Ex
    if not (
        (B1h.x >= B2l.x)
        and (B2h.x >= B1l.x)
        and (B1h.y >= B2l.y)
        and (B2h.y >= B1l.y)
        and (B1h.z >= B2l.z)
        and (B2h.z >= B1l.z)
    ):
        return None
    B1l.x = max(B1l.x, B2l.x)
    B1l.y = max(B1l.y, B2l.y)
    B1l.z = max(B1l.z, B2l.z)
    B1h.x = min(B1h.x, B2h.x)
    B1h.y = min(B1h.y, B2h.y)
    B1h.z = min(B1h.z, B2h.z)
    return (B1l, B1h)


def start_xcode():
    print("Telling XCode to run")
    ##scpt = 'tell application "XCode" to run active workspace document'
    scpt = """
        tell application "XCode"
            run workspace document "bfield.xcodeproj"
        end tell
        """
    p = Popen(
        ['osascript', '-'],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
    )
    stdout, stderr = p.communicate(scpt)
    print(p.returncode, stdout, stderr)


def get_fields_ob(context, create=True):
    name = "0Fields"
    obs = [ob for ob in context.visible_objects if ob.name.startswith(name)]
    if len(obs) == 1:
        ob = obs[0]
    elif len(obs) == 0:
        if not create:
            return
        if ob.verbose > 0:
            print("Creating", name)
        mesh = bpy.data.meshes.new(name)
        ob = bpy.data.objects.new(name, mesh)
        bpy.context.scene.objects.link(ob)
        sz = 16.0
        mesh.from_pydata(((0, 0, 0), (sz, sz, sz)), [], [])
        mesh.update()
        ob.show_texture_space = True

        # put this object in the Main collection
        ##ob.layers = [i == layerMain for i in range(20)]
        print(f"{name} to Main collection {coll_main.name()}")
        coll_main.get().objects.link(ob)

        ob.select_set(True)
    else:
        raise RuntimeError('Expected a single Field object in scene')
    return ob


class Block:
    props = {
        "snap": bp.BoolProperty(
            description="Snap bounds to sim grid", default=False
        ),
        "snapped_name": bp.StringProperty(
            description="Name of snapped-to-sim-grid copy of this object"
        ),
    }

    # cls.__annotations__ isn't used here because it's empty for subclasses

    @classmethod
    def create_types(cls):
        ##print(f"Block.create_types for {cls.__name__}: "
        ##      f"adding {', '.join(cls.props.keys())}")
        for key, value in cls.props.items():
            setattr(bpy.types.Object, key, value)

    @classmethod
    def del_types(cls):
        for key in cls.props.keys():
            delattr(bpy.types.Object, key)

    def get_field_mat(self):
        ob = self.ob
        slots = ob.material_slots
        mat = slots[0].material if slots else None
        if not (mat and mat in self.sim.fmats):
            raise ValueError(f"object {ob.name} requires an FDTD material")
        return mat

    def __init__(self, ob, sim):
        self.sim = sim  # simulation

        scn = bpy.context.scene
        view_layer = bpy.context.view_layer
        dx = sim.dx
        op = bpy.ops.object

        # object may optionally be snapped to grid
        sob = ob
        snapc = coll_snap.get()
        if ob.verbose > 0:
            print("Block.init: checking", ob.name, "snap=", ob.snap)
        if ob.snap:
            name = ob.snapped_name
            if name and scn.objects.get(name):
                # bounds come from existing snapped version of object
                sob = scn.objects.get(name)
            else:
                # or copy sim object to snap coll. and snap bounds to sim grid
                view_layer.objects.active = ob
                ob.select_set(True)
                op.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                Bs, Be = bounds(ob)
                Cu = Vector(ob.scale)
                Su = Be - Bs
                ##print(uob.name, "Bs, Be=", fv(Bs), fv(Be))
                # duplicate-linked the object, same as option-D
                op.add_named(linked=True, name=ob.name)
                sob = bpy.context.object
                # and rename the duplicate
                ob.snapped_name = sob.name = ob.name + "_snap"
                sob.display_type = 'WIRE'
                # sob.layers = [l==layerSnap for l in range(20)]
                coll_main.get().objects.unlink(sob)
                snapc.objects.link(sob)
                Iss = IGrid(Bs, dx)
                Ise = IGrid(Be, dx)
                ##print(uob.name, "Iss, Ise=", Iss, Ise)
                Gss = Vector((Iss.i * dx, Iss.j * dx, Iss.k * dx))
                Gse = Vector((Ise.i * dx, Ise.j * dx, Ise.k * dx))
                ##print(sob.name, "Gss, Gse=", fv(Gss), fv(Gse))
                Ss = Gse - Gss
                Ls = Vector(
                    (Gss.x + Ss.x / 2, Gss.y + Ss.y / 2, Gss.z + Ss.z / 2)
                )
                smin = dx / 10.0
                if Ss.x < smin:
                    Ss.x = smin
                    Ls.x = Gss.x
                if Ss.y < smin:
                    Ss.y = smin
                    Ls.y = Gss.y
                if Ss.z < smin:
                    Ss.z = smin
                    Ls.z = Gss.z
                if ob.verbose > 1:
                    print(sob.name, "Ss, Su=", fv(Ss), fv(Su))
                sob.location = Ls
                if not (Su.x == 0 or Su.y == 0 or Su.z == 0):
                    sob.scale = Vector(
                        (
                            Cu.x * Ss.x / Su.x,
                            Cu.y * Ss.y / Su.y,
                            Cu.z * Ss.z / Su.z,
                        )
                    )
                if ob.verbose > 1:
                    print(sob.name, "loc, scale=", fv(Ls), fv(sob.scale))

                # scn.layers[layerSnap] = True
                snapc.hide_viewport = False

                # recalc sob.matrix_world used by bounds()
                # scn.update()
                view_layer.update()

                Bs, Be = bounds(sob)
                if ob.verbose > 1:
                    print(sob.name, "new Bs, Be=", fv(Bs), fv(Be))

                if ob.blockType == 'PROBE':
                    # force probe to regenerate any image-- size may change
                    if sob.material_slots:
                        sob.material_slots[0].material = None
                else:
                    sob.display_type = 'WIRE'

                # insure that our snapped copy doesn't itself get snapped
                sob.snap = False
                sob.snapped_name = ""

        self.ob = sob
        self.Bs, self.Be = bounds(sob)
        ##print(f"{ob.name} bounds: {fv(self.Bs)} - {fv(self.Be)}")

    def prepare_gen(self):
        yield

    def send_def_gen(self):
        yield


class MatBlock(Block):
    """A simple volume of material in the sim world"""

    # used to select a MATCYLINDER axis or identify Block as a MatBlock
    mtype_codes = {
        'MATCUBE': 'C',
        'FIELDS': 'C',
        'MATMESH': 'C',
        'RESISTOR': 'C',
        'CAPACITOR': 'C',
        'MATCYLINDERX': 'X',
        'MATCYLINDERY': 'Y',
        'MATCYLINDERZ': 'Z',
    }

    @classmethod
    def draw_props(self, ob, layout, scene):
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")

    def prepare_gen(self):
        for _ in super().prepare_gen():
            yield

    def send_def_gen(self):
        ob = self.ob
        if ob.verbose > 0:
            print("MatBlock.send_def_gen")
        Bs, Be = self.Bs, self.Be
        mat = self.get_field_mat()
        mtype = self.mtype_codes[ob['blockType']]

        # if any array modifiers used, start with 0 index
        name = ob.name
        for mod in ob.modifiers:
            if mod.type == 'ARRAY':
                name = f"{ob.name}[000]"

        # send (first) block object's definition to simulator
        cmd = (
            f"B{mtype} {name} {mat.name} {Bs.x:g} "
            f"{Be.x:g} {Bs.y:g} {Be.y:g} {Bs.z:g} {Be.z:g}"
        )
        self.sim.send(cmd)
        if ob.verbose > 0:
            print(cmd)
        selection = [(Bs, Be)]
        arrayi = 1

        # handle any modifiers: arrays will send rest of objects, each
        # array sending copies of all the previous ones (in selection)
        for mod in ob.modifiers:
            t = mod.type
            if t not in ('BOOLEAN',):
                if t == 'ARRAY':
                    if ob.verbose > 0:
                        print(f" {ob.name} array modifier, {mod.count=}")
                    if not (
                        mod.use_constant_offset
                        and not (
                            mod.use_object_offset
                            or mod.use_relative_offset
                            or mod.use_merge_vertices
                        )
                    ):
                        print(f"**** {ob.name} array offsets not constant")
                    offset = mod.constant_offset_displace
                    new_selection = []
                    for i in range(1, mod.count):
                        for Bs, Be in selection:
                            nameo = f"{ob.name}[{arrayi:03}]"
                            Bso = Bs + i * offset
                            Beo = Be + i * offset
                            if ob.verbose > 0:
                                print(f" {mod.name} {i} ", end='')
                            cmd = (
                                f"B{mtype} {nameo} {mat.name} "
                                f"{Bso.x:g} {Beo.x:g} {Bso.y:g} "
                                f"{Beo.y:g} {Bso.z:g} {Beo.z:g}"
                            )
                            self.sim.send(cmd)
                            if ob.verbose > 0:
                                print(cmd)
                            new_selection.append((Bso, Beo))
                            arrayi += 1
                    selection.extend(new_selection)

                else:
                    print(f"**** {ob.name} needs Apply {t} Modifier!")

        yield

    def assert_material(self, ob, type, value, color):
        """Assert that component ob of given type has a material with a
        name matching package-value. Also assigns the color if created.
        Returns the material.
        """
        m = ob.active_material
        if m is None:
            pkg = ob.data.name[:5]
            mname = f"{pkg}-{value}"
            m = bpy.data.materials.get(mname)
            if m is None:
                m = bpy.data.materials.new(name=mname)
                m.diffuse_color = color
                print(f"{ob.name}: created {type} material {m.name}")
            if len(ob.material_slots) == 0:
                ob.data.materials.append(m)
            ob.material_slots[0].material = m
            ob.material_slots[0].link = 'OBJECT'
        print(f"{ob.name}: {type} material {m.name}")
        return m


class MeshMatBlock(MatBlock):
    """A volume of material in the sim world"""

    def prepare_gen(self):
        for _ in super().prepare_gen():
            yield

    def send_def_gen(self):
        """Write an IMBlock's mesh as a stack of image files for the server.
        Only alpha layer is used.
        """
        ob = self.ob
        sim = self.sim
        scn = bpy.context.scene
        dx = sim.dx
        if ob.verbose > 0:
            print("MeshMatBlock.send_def_gen", ob.name, "start")

        # need the snap layer active for Dynamic Paint
        snapc = coll_snap.get()
        snap_hide_save = snapc.hide_viewport
        snapc.hide_viewport = False

        # make plane that covers object, and square for a DP canvas
        Bs, Be = self.Bs, self.Be
        x, y = (Bs.x + Be.x) / 2, (Bs.y + Be.y) / 2
        D = ob.dimensions
        r = (D.x, D.y)[D.y > D.x] / 2
        ifactor = 4
        nx = max(sim.grid(2 * r) * ifactor, 1)
        nz = max(sim.grid(D.z) * ifactor, 1)
        ##print(f"Plane {ob.name}: {D.z=} {sim.grid(D.z)=} {nz=}")
        mat = self.get_field_mat()

        # only create directory of cached files if it doesn't exist
        dir = f"//cache_dp/{ob.name}_{nz:04}"
        dirp = bpy.path.abspath(dir)
        ##print(f"{cwd=} cache_dp {dirp=}")
        if not os.path.isdir(dirp):

            # delete any previous canvas object and brush modifier
            # (the [:] copies the list first)
            for o in scn.objects[:]:
                for m in o.modifiers:
                    if m.type == 'DYNAMIC_PAINT':
                        bpy.ops.object.select_all(action='DESELECT')
                        # scn.objects.active = o
                        bpy.context.view_layer.objects.active = o
                        ohide = o.hide_viewport
                        o.select_set(True)
                        o.hide_viewport = False
                        if m.brush_settings:
                            if ob.verbose > 1:
                                print("deleting", o.name, "Dynamic Paint mod")
                            bpy.ops.object.modifier_remove(modifier=m.name)
                            o.hide_viewport = ohide
                        elif m.canvas_settings:
                            if ob.verbose > 1:
                                print("deleting canvas plane", o.name)
                            bpy.ops.object.delete(use_global=False)
            yield

            # make this object the brush
            dp = ob.modifiers.get("Dynamic Paint")
            if dp is None:
                if ob.verbose > 1:
                    print("adding brush to", ob.name)
                bpy.context.view_layer.objects.active = ob
                bpy.ops.object.modifier_add(type='DYNAMIC_PAINT')
                dp = ob.modifiers.get("Dynamic Paint")
            if dp.brush_settings is None:
                dp.ui_type = 'BRUSH'
                bpy.ops.dpaint.type_toggle(type='BRUSH')

            # put canvas plane in X,Y center of object
            plane_name = coll_plane.name()
            vl = bpy.context.view_layer
            lc = vl.layer_collection.children[plane_name]
            vl.active_layer_collection = lc
            bpy.ops.mesh.primitive_plane_add()
            so = bpy.context.object
            print(
                f"canvas plane for {ob.name} is {so.name}, "
                f"verbose={ob.verbose}"
            )
            snapc.objects.link(so)
            so.scale = (r, r, 1)
            so.hide_viewport = False

            # set up a linear Z-slicing path for canvas through object
            try:
                scn.frame_set(1)
                print(f" {nz} slices at location=({x}, {y}, {Bs.z} -> {Be.z})")
                so.location = (x, y, Bs.z)
                so.keyframe_insert(data_path="location", frame=1)
                scn.frame_set(nz)
                so.location = (x, y, Be.z)
                so.keyframe_insert(data_path="location", frame=nz)
                zc = so.animation_data.action.fcurves[2]
                for kp in zc.keyframe_points:
                    kp.handle_left_type = 'VECTOR'
                    kp.handle_right_type = 'VECTOR'
                    kp.interpolation = 'LINEAR'
            except Exception as exc:
                raise RuntimeError(f"Z-slicing {ob.name}") from exc

            # make plane the receiving canvas; images will be square,
            #   min 16x16
            bpy.context.view_layer.objects.active = so
            bpy.ops.object.modifier_add(type='DYNAMIC_PAINT')
            dp = so.modifiers.get("Dynamic Paint")
            dp.ui_type = 'CANVAS'
            bpy.ops.dpaint.type_toggle(type='CANVAS')
            csurf = dp.canvas_settings.canvas_surfaces[0]
            csurf.surface_format = 'IMAGE'
            csurf.image_resolution = nx
            csurf.frame_end = nz
            csurf.dry_speed = 1
            csurf.use_dissolve = True
            csurf.dissolve_speed = 1
            csurf.use_antialiasing = True
            csurf.image_output_path = dir
            bpy.ops.mesh.uv_texture_add()

            # bake: write Z-slice images of object to png files
            last_fn = csurf.image_output_path + (f"/paintmap{nz:04}.png")
            if ob.verbose > 1:
                print("deleting any", last_fn)
            p = bpy.path.abspath(last_fn)
            if os.path.isfile(p):
                os.unlink(p)
            if ob.verbose > 1:
                print("sendVoxels_G: baking canvas", so.name)
            res = bpy.ops.dpaint.bake()
            if ob.verbose > 1:
                print("DP bake returned", res)
            yield
            # wait for last file to be (re)created
            while not os.path.isfile(p):
                if ob.verbose > 1:
                    print("waiting for", last_fn)
                yield
            ##so.hide_viewport = True
            bpy.data.objects.remove(so, do_unlink=True)

        # tell server to load image files
        bpy.context.view_layer.objects.active = ob
        nx = max(nx, 16)
        sim.send(
            f"HI {ob.name} {mat.name} {x-r:g} {x+r:g} {y-r:g} {y+r:g} "
            f"{Bs.z:g} {Be.z:g} {nx} {nz}"
        )
        if ob.verbose > 0:
            print("MeshMatBlock.send_def_gen", ob.name, "done.")
        snapc.hide_viewport = snap_hide_save


class LayerMatBlock(Block):
    """An image-defined layer material in the sim world"""

    props = {
        "fmat_name": bp.StringProperty(description="FDTD material"),
        "snap": bp.BoolProperty(
            description="Snap bounds to sim grid", default=False
        ),
    }

    @classmethod
    def draw_props(self, ob, layout, scene):
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")

        layout.prop_search(
            ob, 'fmat_name', bpy.data, 'materials', text="FDTD material"
        )

    def prepare_gen(self):
        for _ in super().prepare_gen():
            yield

    def send_def_gen(self):
        ob = self.ob
        if ob.verbose > 0:
            print("LayerMatBlock.send_def_gen", ob.name, "start")
        mesh = ob.data
        mat = mesh.materials[0]
        tex = mat.node_tree.nodes.get('Image Texture')
        if not tex:
            raise ValueError(f"object {ob.name} missing Image Texture node")
        img = tex.image
        img_file_path = bpy.path.abspath(img.filepath)
        Bs, Be = self.Bs, self.Be
        fmat_name = ob.get('fmat_name')
        self.sim.send(
            f"LI {ob.name} {fmat_name} {Bs.x:g} {Be.x:g} "
            f"{Bs.y:g} {Be.y:g} {Bs.z:g} {Be.z:g} {img_file_path}"
        )
        yield


class FieldsBlock(MatBlock):
    props = {
        "dx": bp.FloatProperty(
            description="Grid cell spacing, mm", min=0.0, default=1.0
        ),
        "stop_ps": bp.FloatProperty(
            description="When to stop sim, in ps", min=0.0, default=0.0
        ),
        "usPoll": bp.IntProperty(description="us/Step", min=0, default=50),
        "msRate": bp.IntProperty(description="ms/Update", min=0, default=500),
        "rec": bp.BoolProperty(
            description="Record as animation", default=True
        ),
        "pml_border": bp.IntProperty(
            description="PML border width, cells", min=0, default=4
        ),
        "verbose": bp.IntProperty(
            description="Sim verbosity level, 0-3", min=0, default=0
        ),
    }

    @classmethod
    def draw_props(self, ob, layout, scene):
        spit = layout.split()
        col = spit.column(align=True)
        col.label(text="Grid size (dx):")
        col.prop(ob, "dx", text="mm")
        dx = ob.get('dx')
        D = ob.dimensions
        if not dx is None:
            col.label(text="nx, ny, nz=")
            col.label(
                text=f"{ma.floor(D.x/dx)}, {ma.floor(D.y/dx)}, "
                f"{ma.floor(D.z/dx)}"
            )
        col = spit.column()
        col.prop(ob, 'usPoll', text="µs/Step")
        col.prop(ob, 'msRate', text="ms/Up")
        col.prop(ob, 'pml_border', text="PML cells")
        col.prop(ob, 'verbose', text="Verbosity")
        col.prop(ob, 'rec', text="Record")
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")

    def send_sim_def_gen(self):
        ob = self.ob
        if ob.verbose > 0:
            print("FieldsBlock.send_sim_def_gen")
        yield
        Bs, Be = self.Bs, self.Be
        mat = self.get_field_mat()
        sim = self.sim
        sim.send("A units mm")
        sim.send(f"A usPoll {sim.usPoll}")
        sim.send(f"A verbose {sim.verbose}")
        cmd = (
            f"F {ob.name} {mat.name} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
            f"{Bs.z:g} {Be.z:g} {sim.dx:g} 1 {sim.pml_border}"
        )
        if sim.verbose > 1:
            print(cmd)
        self.sim.send(cmd)


class Resistor(MatBlock):
    """A block of resistive material, creates the material if needed"""

    props = {
        "resistance": bp.FloatProperty(
            description="Resistance", min=0.0, default=1.0
        ),
        "res_units": bp.StringProperty(
            description="units for resistance", default='ohms'
        ),
        "axis": bp.StringProperty(
            description="axis of resistance (X/Y/Z)", default='X'
        ),
    }

    @classmethod
    def draw_props(self, ob, layout, scene):
        split = layout.split(factor=0.6)
        split.row().prop(ob, 'resistance')
        split.prop_search(ob, 'res_units', scene, 'res_units', text="")
        layout.prop_search(ob, 'axis', scene, 's_axes', text="Axis")
        layout.prop(ob, 'verbose', text="Verbosity")

    def prepare_gen(self):
        ob = self.ob
        ##print("Resistor.prepare_gen", ob.name, ob.axis)
        for _ in super().prepare_gen():
            yield
        R = ob.resistance * res_units[ob.res_units]
        sige = 1.0 / R
        # block dimensions [mm]
        D = self.Be - self.Bs
        if D.x < 0 or D.y < 0 or D.z < 0:
            print(f"**** Resistor {ob.name} rotated")
            return
        axis = ob.axis[-1]
        if axis == 'X':
            sige = sige * D.x / (D.y * D.z)
        elif axis == 'Y':
            sige = sige * D.y / (D.x * D.z)
        elif axis == 'Z':
            sige = sige * D.z / (D.x * D.y)
        sige = sige / mm

        value = f"{ob.resistance:g}{ob.res_units}"
        color = (0.025, 0.025, 0.025, 1.0)
        m = self.assert_material(ob, "resistor", value, color)
        m.mur = 1.0
        m.epr = 1.0
        m.sige = sige

    def send_def_gen(self):
        ob = self.ob
        if ob.verbose > 0:
            print("Resistor.send_def_gen", ob.name, "start")
        for _ in super().send_def_gen():
            yield


class Capacitor(MatBlock):
    """A capacitor formed from 2 plates and block of dielectric. Creates
    the material if needed.
    """

    props = {
        "capacitance": bp.FloatProperty(
            description="Capacitance", min=0.0, default=1.0
        ),
        "cap_units": bp.StringProperty(
            description="units for capacitance", default='pf'
        ),
        "axis": bp.StringProperty(
            description="axis of capacitor (X/Y/Z)", default='X'
        ),
    }

    @classmethod
    def draw_props(self, ob, layout, scene):
        split = layout.split(factor=0.6)
        split.row().prop(ob, 'capacitance')
        split.prop_search(ob, 'cap_units', scene, 'cap_units', text="")
        layout.prop_search(ob, 'axis', scene, 's_axes', text="Axis")
        layout.prop(ob, 'verbose', text="Verbosity")

    def prepare_gen(self):
        ob = self.ob
        ##print("Capacitor.prepare_gen", ob.name, ob.axis)
        for _ in super().prepare_gen():
            yield
        C = ob.capacitance * cap_units[ob.cap_units]
        # block dimensions [mm]
        D = self.Be - self.Bs
        if D.x < 0 or D.y < 0 or D.z < 0:
            print(f"**** Capacitor {ob.name} rotated")
            return
        axis = ob.axis[-1]
        # C = (e0*epr)*(A/d)
        # epr = C*(d/A)/e0
        # d/A is in 1/meter
        # d/A = r / mm
        if axis == 'X':
            r = D.x / (D.y * D.z)
        elif axis == 'Y':
            r = D.y / (D.x * D.z)
        elif axis == 'Z':
            r = D.z / (D.x * D.y)

        value = f"{ob.capacitance:g}{ob.cap_units}"
        color = (0.81, 0.75, 0.59, 1.0)
        m = self.assert_material(ob, "capacitor", value, color)
        m.mur = 1.0
        m.epr = C * r / (mm * e0)
        m.sige = 0.0

    def send_def_gen(self):
        ob = self.ob
        if ob.verbose > 0:
            print(f"Capacitor.send_def_gen{ob.name} start")
        for _ in super().send_def_gen():
            yield


class SubSpaceBlock(MatBlock):
    @classmethod
    def draw_props(self, ob, layout, scene):
        ##col.prop(ob, "dx", text="mm")  # assumed dx = parent.dx/2
        pass

    def send_sim_def_gen(self):
        ob = self.ob
        if ob.verbose > 0:
            print("SubSpaceBlock.send_sim_def_gen")
        yield
        Bs, Be = self.Bs, self.Be
        mat = self.get_field_mat()
        cmd = (
            f"G {ob.name} {mat.name} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
            f"{Bs.z:g} {Be.z:g} {ob.parent.name}"
        )
        if sim.verbose > 1:
            print(cmd)
        self.sim.send(cmd)


class Source(Block):
    props = {
        "s_axis": bp.StringProperty(description="Axis of positive voltage"),
        "s_excitation": bp.StringProperty(description="Excitation"),
        "s_function": bp.StringProperty(description="Function"),
        "s_hard": bp.BoolProperty(description="Hard source", default=False),
        "s_resistance": bp.FloatProperty(
            description="Soft source resistance (ohms)", min=0.0, default=50.0
        ),
        "s_scale": bp.FloatProperty(
            description="Signal height", min=0.0, default=1.0
        ),
        "s_tstart": bp.FloatProperty(
            description="Pulse start time", min=0.0, default=0.0
        ),
        "s_tstartUnits": bp.StringProperty(
            description="Pulse start time units", default='ps'
        ),
        "s_trise": bp.FloatProperty(
            description="Pulse rise time", min=0.0, default=10.0
        ),
        "s_triseUnits": bp.StringProperty(
            description="Pulse rise time units", default='ps'
        ),
        "s_duration": bp.FloatProperty(
            description="Pulse duration (after trise, before tfall)",
            min=0.0,
            default=0.0,
        ),
        "s_durationUnits": bp.StringProperty(
            description="Pulse duration units", default='sec'
        ),
        "s_tfall": bp.FloatProperty(
            description="Pulse fall time", min=0.0, default=10.0
        ),
        "s_tfallUnits": bp.StringProperty(
            description="Pulse fall time units", default='ps'
        ),
    }

    @classmethod
    def register_types(self):
        bpy.types.Scene.s_excitations = bp.CollectionProperty(
            type=bpy.types.PropertyGroup
        )
        bpy.types.Scene.s_axes = bp.CollectionProperty(
            type=bpy.types.PropertyGroup
        )
        bpy.types.Scene.s_functions = bp.CollectionProperty(
            type=bpy.types.PropertyGroup
        )
        bpy.types.Scene.time_units = bp.CollectionProperty(
            type=bpy.types.PropertyGroup
        )
        bpy.types.Scene.res_units = bp.CollectionProperty(
            type=bpy.types.PropertyGroup
        )
        bpy.types.Scene.cap_units = bp.CollectionProperty(
            type=bpy.types.PropertyGroup
        )

    @classmethod
    def unregister_types(self):
        del bpy.types.Scene.s_excitations
        del bpy.types.Scene.s_axes
        del bpy.types.Scene.s_functions
        del bpy.types.Scene.time_units
        del bpy.types.Scene.res_units
        del bpy.types.Scene.cap_units

    @classmethod
    def populate_types(self, scene):
        scene.s_excitations.clear()
        for k in ('Voltage', 'Current', 'Electrical', 'Magnetic'):
            scene.s_excitations.add().name = k
        scene.s_axes.clear()
        for k in (' X', ' Y', ' Z', '-X', '-Y', '-Z'):
            scene.s_axes.add().name = k
        scene.s_functions.clear()
        for k in ('Gaussian Pulse', 'Sine', 'Constant'):
            scene.s_functions.add().name = k
        scene.time_units.clear()
        for k in time_units.keys():
            scene.time_units.add().name = k
        scene.res_units.clear()
        for k in res_units.keys():
            scene.res_units.add().name = k
        scene.cap_units.clear()
        for k in cap_units.keys():
            scene.cap_units.add().name = k

    @classmethod
    def draw_props(self, ob, layout, scene):
        layout.prop_search(
            ob, 's_excitation', scene, 's_excitations', text="Excitation"
        )
        layout.prop_search(
            ob, 's_function', scene, 's_functions', text="Function"
        )
        split = layout.split(factor=0.33)
        split.row().prop(ob, 's_hard', text="Hard")
        if not ob.get('s_hard'):
            split.row().prop(ob, 's_resistance', text="Resistance")
        split = layout.split(factor=0.5)
        split.row().prop_search(ob, 's_axis', scene, 's_axes', text="Axis")
        split.row().prop(ob, 's_scale', text="Scale")
        if ob.s_function != 'Constant':
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_tstart', text="Start time")
            split.prop_search(
                ob, 's_tstartUnits', scene, 'time_units', text=""
            )
        if ob.s_function == 'Gaussian Pulse':
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_trise', text="Rise time")
            split.prop_search(ob, 's_triseUnits', scene, 'time_units', text="")
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_duration', text="Duration")
            split.prop_search(
                ob, 's_durationUnits', scene, 'time_units', text=""
            )
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_tfall', text="Fall time")
            split.prop_search(ob, 's_tfallUnits', scene, 'time_units', text="")
        elif ob.s_function == 'Sine':
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_duration', text="Period")
            split.prop_search(
                ob, 's_durationUnits', scene, 'time_units', text=""
            )
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")
        layout.prop(ob, 'verbose', text="Verbosity")

    def send_def_gen(self):
        ob = self.ob
        if ob.verbose > 0:
            print("Source.send_def_gen")
        yield
        Bs, Be = self.Bs, self.Be
        scale = ob.s_scale
        axis = ord(ob.s_axis[-1]) - ord('X')
        dist = (Be.x - Bs.x, Be.y - Bs.y, Be.z - Bs.z)[axis] * mm
        if ob.verbose > 1:
            print("Source", ob.name, "dist=", dist, "m")
        if ob.s_axis[0] == '-':
            scale *= -1
        ex = ob.s_excitation[0]
        if ex == 'V':
            # convert voltage across source to E field strength
            ex = 'E'
            scale /= dist
            if not ob.s_hard:
                scale *= 2  # assume load matches Rs, dividing Vs by two

        cmd = (
            f"S {ob.name} {ex} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
            f"{Bs.z:g} {Be.z:g} {ob.s_function.replace(' ', '_')} "
            f"{ob.s_hard:d} {ob.s_resistance:g} {axis} {scale:g} "
            f"{tu(ob,'s_tstart'):g} {tu(ob,'s_trise'):g} "
            f"{tu(ob,'s_duration'):g} {tu(ob,'s_tfall'):g}"
        )
        if ob.verbose > 1:
            print(cmd)
        self.sim.send(cmd)


class Figure:
    """A matplotlib figure window"""

    winx = None
    winy = None
    num = 1

    def __init__(self):
        # create a matplotlib plot
        print("creating Figure", Figure.num)
        self.figure = plt.figure(Figure.num, figsize=(4.7, 4))
        self.figure.clear()
        Figure.num += 1
        self.ax = plt.axes()
        self.max_x = 0.0
        self.min_y = 9999999.0
        self.max_y = -9999999.0
        self.ylabel = ""


class NFmtr(ticker.Formatter):
    def __init__(self, scale_x):
        self.scale_x = scale_x

    def __call__(self, x, pos=None):
        return "{0:g}".format(x / 10**self.scale_x)


class Probe(Block):
    props = {
        "p_field": bp.StringProperty(
            description="Field to measure", default='Electric'
        ),
        "p_axis": bp.StringProperty(
            description="Measurement axis", default='XYZ'
        ),
        "p_axisSign": bp.IntProperty(
            description="Measurement axis sign", default=1
        ),
        "p_verbose": bp.IntProperty(
            description="Probe verbosity level, 0-3", min=0, default=0
        ),
        "p_shape": bp.StringProperty(
            description="Display shape", default='Plane'
        ),
        "p_value": bp.FloatProperty(
            precision=6, description="Probe measured value"
        ),
        "p_value3": bp.FloatVectorProperty(
            description="Probe measured vector value"
        ),
        "p_dispScale": bp.FloatProperty(
            min=0, default=256.0, description="Display scale"
        ),
        "p_pixelRep": bp.IntProperty(
            min=1, default=1, description="Image pixel repeat factor"
        ),
        "p_imageAlpha": bp.FloatProperty(
            description="Image transparency alpha", min=0.0, default=1.0
        ),
        "p_sfactor": bp.IntProperty(
            description="Volume space factor, cells/sample", min=1, default=1
        ),
        "p_log": bp.BoolProperty(
            description="Log scale for magnitude", default=True
        ),
        "p_magScale": bp.FloatProperty(
            description="Magnitude multiplier", min=0.0, default=1.0
        ),
        "p_sum": bp.BoolProperty(description="Sum values", default=False),
        "p_avg": bp.BoolProperty(description="Average values", default=False),
        "p_dispIsMesh": bp.BoolProperty(
            description="Use mesh object for in-world chart", default=False
        ),
        "p_dispIsPlot": bp.BoolProperty(
            description="Use external MatPlotLib for chart", default=True
        ),
        "p_legendLoc": bp.StringProperty(
            description="Plot legend location", default="best"
        ),
        "p_plotScale": bp.FloatProperty(
            description="chart scale multiplier", min=0.0, default=1.0
        ),
        "p_dispColor": bp.FloatVectorProperty(
            description="Color",
            subtype='COLOR',
            size=4,
            min=0.0,
            max=1.0,
            default=(0.75, 0.0, 0.8, 1.0),
        ),
        "p_dispPos": bp.FloatProperty(
            min=0.0,
            max=1.0,
            description="Relative position in chart",
            default=0.5,
        ),
    }
    fieldNames = {
        'Electric': 'E',
        'Magnetic': 'H',
        'Voltage': 'E',
        'Current Density': 'J',
        'mE2': 'M',
        'mH2': 'N',
    }
    field_names_mag = {
        'Electric': 'E',
        'Magnetic': 'H',
        'Voltage': 'E',
        'mE2': 'M',
        'mH2': 'N',
        'SigE': 'S',
        'Current Density': 'J',
    }
    fieldUnits = {
        'V': "V",
        'E': "V/m",
        'M': "A/m",
        'S': "50MS",
        'T': "50MS",
        'C': "A/m^2",
    }

    @classmethod
    def register_types(self):
        ts = bpy.types.Scene
        ts.p_fields = bp.CollectionProperty(type=bpy.types.PropertyGroup)
        ts.p_fieldsMag = bp.CollectionProperty(type=bpy.types.PropertyGroup)
        ts.p_axes = bp.CollectionProperty(type=bpy.types.PropertyGroup)
        ts.p_shapes = bp.CollectionProperty(type=bpy.types.PropertyGroup)
        ts.p_legendLocs = bp.CollectionProperty(type=bpy.types.PropertyGroup)

    @classmethod
    def unregister_types(self):
        del bpy.types.Scene.p_fields
        del bpy.types.Scene.p_fieldsMag
        del bpy.types.Scene.p_axes
        del bpy.types.Scene.p_shapes
        del bpy.types.Scene.p_legendLocs

    @classmethod
    def populate_types(self, scene):
        scene.p_fields.clear()
        for k in self.fieldNames.keys():
            scene.p_fields.add().name = k
        scene.p_fieldsMag.clear()
        for k in self.field_names_mag.keys():
            scene.p_fieldsMag.add().name = k
        scene.p_axes.clear()
        for k in ('X', 'Y', 'Z', '-X', '-Y', '-Z', 'XYZ', 'Magnitude'):
            scene.p_axes.add().name = k
        scene.p_shapes.clear()
        for k in ('Point', 'Line', 'Plane', 'Volume'):
            scene.p_shapes.add().name = k
        scene.p_legendLocs.clear()
        for k in (
            'best',
            'upper right',
            'upper left',
            'lower left',
            'lower right',
            'right',
            'center left',
            'center right',
            'lower center',
            'upper center',
            'center',
        ):
            scene.p_legendLocs.add().name = k

    @classmethod
    def draw_props(self, ob, layout, scene):
        fields = ('p_fields', 'p_fieldsMag')[ob.p_axis == 'Magnitude']
        layout.prop_search(
            ob, 'p_shape', scene, 'p_shapes', text="Display Shape"
        )

        if ob.p_shape == 'Point':
            layout.prop_search(ob, 'p_axis', scene, 'p_axes', text="Axis")
            layout.prop_search(ob, 'p_field', scene, fields, text="Field")
            row = layout.row()
            row.prop(ob, 'p_dispIsMesh', text="Use Mesh")
            row.prop(ob, 'p_dispIsPlot', text="MatPlotLib")
            row = layout.row()
            row.prop(ob, 'p_sum', text="Sum values")
            row.prop(ob, 'p_verbose', text="Verbosity")
            if not (ob.p_dispIsMesh or ob.p_dispIsPlot):
                split = layout.split(factor=0.5)
                split.row().prop(ob, 'p_dispPos', text="Pos")
                split.row().prop(ob, 'p_dispColor', text="Color")
            box = layout.box()
            units = self.fieldUnits[ob.p_field[0]]
            box.label(text=f"Measurement ({units})")
            value = 'p_value3' if ob.p_axis == 'XYZ' else 'p_value'
            self.measurement_attr_name = value
            self.measurement_units = units
            box.row().prop(ob, value, text="")
            layout.prop(ob, 'p_plotScale', text="Plot Scale")
            layout.prop_search(
                ob,
                'p_legendLoc',
                scene,
                'p_legendLocs',
                text="Legend location",
            )

        elif ob.p_shape == 'Line':
            layout.prop_search(ob, 'p_axis', scene, 'p_axes', text="Axis")
            layout.prop_search(ob, 'p_field', scene, fields, text="Field")
            row = layout.row()
            row.prop(ob, 'p_sum', text="Sum values")
            row.prop(ob, 'p_avg', text="Average values")
            row = layout.row()
            row.prop(ob, 'p_verbose', text="Verbosity")

        elif ob.p_shape == 'Plane':
            layout.prop_search(ob, 'p_axis', scene, 'p_axes', text="Axis")
            layout.prop_search(ob, 'p_field', scene, fields, text="Field")
            row = layout.row()
            row.prop(ob, 'p_dispScale', text="Brightness")
            row.prop(ob, 'p_pixelRep', text="Repeat")
            row = layout.row()
            row.prop(ob, 'p_imageAlpha', text="Alpha")
            row.prop(ob, 'p_verbose', text="Verbosity")

        elif ob.p_shape == 'Volume':
            layout.prop_search(ob, 'p_field', scene, fields, text="Field")
            row = layout.row()
            row.prop(ob, 'p_sfactor', text="Cells/Sample")
            row.prop(ob, 'p_log', text="Log scale")
            row.prop(ob, 'p_verbose', text="Verbosity")
            layout.prop(ob, 'p_magScale', text="Mag scale")
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")

    def __init__(self, ob, sim):
        super().__init__(ob, sim)
        self.history = {}

    def set_plane_texture(self, data):
        ob = self.ob
        mesh = ob.data
        name = ob.name
        if not mesh:
            raise ValueError(f"object {name} missing a mesh")
        if not mesh.materials:
            raise ValueError(f"object {name} missing a material")
        mat = mesh.materials[0]
        if not mat or not mat.use_nodes:
            raise ValueError(f"object {name} missing material or nodes")
        tex = mat.node_tree.nodes.get('Image Texture')
        if not tex:
            raise ValueError(f"object {name} missing Image Texture node")
        img = tex.image
        if not img:
            raise ValueError(f"object {name} missing an image")
        mag_scale, rep = ob.p_dispScale, ob.p_pixelRep
        nix, niy = self.nix, self.niy
        di = 0
        if (
            img.generated_width == rep * nix
            and img.generated_height == rep * niy
        ):
            if ob.p_axis == 'XYZ':
                # RGBA as xyz vector
                adata = np.frombuffer(data, np.uint32).reshape((nix, niy)).T
                apix = np.frombuffer(adata.reshape((-1, 1)), np.uint8) / 256.0
            else:
                # greyscale
                adata = np.frombuffer(data, np.float32).reshape((nix, niy)).T
                adata = (np.abs(adata) * mag_scale).clip(0, 1.0)
                apix = np.zeros((nix * niy, 4)) + 1.0
                apix[:, 0:3] = adata.reshape((-1, 1))
            if rep > 1:
                apix = apix.reshape((niy, nix, 4))
                ##print(f"setPlaneTex: {rep=}, {nix=}, {niy=}")
                apix = apix.repeat(rep, axis=0).repeat(rep, axis=1)
            img.pixels = apix.reshape(-1)
        else:
            print(
                f"probe.P: image size mismatch: {img.name} "
                f"should be {nix}x{niy}"
            )

    def probe_plane_frame_handler(self, scene, depsgraph):
        """Timeline frame changed: update probe from history"""
        data = self.history.get(scene.frame_current)
        if not data is None:
            self.set_plane_texture(data)

    def probe_value_frame_handler(self, scene, depsgraph):
        data = self.history.get(scene.frame_current)
        ##print("probeValueFrHand:", self.ob.name, scn.frame_current, data)
        if not data is None:
            if type(data) == Vector:
                self.ob.p_value3 = data
            else:
                self.ob.p_value = data
        pass  # (fixes func pulldown indentation)

    # Create or update a probe's display image, objects, etc.

    def prepare_gen(self):
        ob = self.ob
        if ob.p_verbose > 0:
            print("Probe.prepare_gen start", self.ob.name)
        yield
        sim = self.sim
        scn = bpy.context.scene
        objs = bpy.data.objects
        posth = bpy.app.handlers.frame_change_post
        bmats = bpy.data.materials
        field_name = self.field_names_mag[ob.p_field]
        sfactor = ob.p_sfactor
        dx = sim.dx
        self.last_step = -1
        if ob.p_verbose > 0:
            print(f"Probe.prepare_gen for {ob.name}: {dx=:3} {sfactor=}")

        # get untrimmed probe dimensions
        B0l, B0u = self.Bs, self.Be
        ##print(f"{ob.name} bounds: {fv(B0l)}, {fv(B0u)}")

        # determine trimmed probe grid coords and size
        fields = sim.fields_block
        fob = fields.ob
        Bsf, Bef = bounds(fob)
        fover = overlap(ob, fob, fob.pml_border * dx)
        if fover is None:
            raise ValueError("Bug: probe doesn't overlap Fields!")
        B1l, B1u = fover
        Is = IGrid(B1l - Bsf, dx)
        ##print("Probe: ob=", ob.name, "ob.p_verbose=", ob.p_verbose)
        if ob.p_verbose > 0:
            print("B1l.x=", B1l.x, "Bsf.x=", Bsf.x, "dx=", dx)
        Ie = IGrid(B1u - Bsf, dx)
        nx = max(Ie.i - Is.i, 1)
        ny = max(Ie.j - Is.j, 1)
        ##print("ny:", Ie.j, Is.j, ny)
        nz = max(Ie.k - Is.k, 1)
        if ob.p_verbose > 0:
            print(ob.name, "Bsf=", fv(Bsf), "Is=", Is, "Ie=", Ie)
            print(" nx,ny,nz=", nx, ny, nz)
        self.N = IVector(nx, ny, nz)

        shape = ob.p_shape
        if shape == 'Point':
            if ob.type == 'EMPTY':
                n = 1
                if ob.p_axis == 'XYZ':
                    n = 3
                if ob.p_verbose > 0:
                    print(ob.name, "single point measurement,", n, "values")
                self.n = n
                if self.probe_value_frame_handler in posth:
                    del posth[self.probe_value_frame_handler]
                posth.append(self.probe_value_frame_handler)
            else:
                self.n = nx * ny * nz
                if ob.p_verbose > 0:
                    print(
                        ob.name, "extended (sum) point measurement, n=", self.n
                    )

        elif shape == 'Line':
            self.n = nx * ny * nz

        elif shape == 'Plane':
            nix, niy = nx, ny
            if nx == 1:
                nix = ny
                niy = nz
            elif ny == 1:
                niy = nz
            n = nix * niy
            if ob.p_verbose > 0:
                print(f"probe.plane {nix}x{niy}, total size = {n} elements")
            if n < 1 or n > 80000:
                raise ValueError(f"probe.plane: bad requested data size: {n}")
            self.n = n
            self.nix, self.niy = nix, niy

            # remove any wrong-sized image first
            mesh = ob.data
            name = ob.name
            mat = None
            rep = ob.p_pixelRep
            if len(mesh.materials) == 1:
                mat = mesh.materials[0]
                if mat and mat.use_nodes:
                    ##print(f"Probe.prepare_gen found mat {repr(mat)}")
                    tex = mat.node_tree.nodes.get('Image Texture')
                    if tex:
                        img = tex.image
                        if (
                            img is None
                            or img.generated_width != rep * nix
                            or img.generated_height != rep * niy
                        ):
                            if ob.p_verbose > 1:
                                print(
                                    f"removing old wrong-sized "
                                    f"{img.generated_width}x"
                                    f"{img.generated_height} image"
                                )
                            mat = None
                    else:
                        mat = None
                else:
                    mat = None

            if mat is None:
                # create a new material with an image texture
                if ob.p_verbose > 1:
                    print(f"Probe: creating {nix}x{niy} image plane")
                mesh.materials.clear()
                mat = bpy.data.materials.new(name)
                print(f"prepare_gen created mat {repr(mat)}")
                mat.use_nodes = True
                mat.specular_intensity = 0.0
                mesh.materials.append(mat)
                img_name = f"probe_{name}"
                img = bpy.data.images.new(
                    img_name, width=rep * nix, height=rep * niy
                )

                tex = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
                tex.image = img
                # mtex.texture_coords = 'UV'
                # mtex.use_map_color_diffuse = True
                # mtex.mapping = 'FLAT'
                # if len(mesh.uv_layers) == 0:
                #     bpy.ops.mesh.uv_texture_add()

                tex.location = (0, 0)
                # link the texture node to the material
                mat.node_tree.links.new(
                    mat.node_tree.nodes['Principled BSDF'].inputs[
                        'Base Color'
                    ],
                    tex.outputs['Color'],
                )

                # the following depends on mesh.polygons[0].vertices having
                # order: 0 1 3 2, and mesh.vertices being like
                # (0,0,0) (0,0,43) (0,60,0) (0,60,43)

            if mat:
                talpha = ob.p_imageAlpha
                ##print(f"{ob.name}: setting image alpha to {talpha}")
                mat.node_tree.nodes["Principled BSDF"].inputs[
                    'Alpha'
                ].default_value = talpha
                mat.blend_method = 'BLEND'

            # bounds, absolute, untrimmed: B0l to B0u, trimmed: B1l, B1u
            B1d = B1u - B1l
            uvmap = mesh.uv_layers.active
            ud = uvmap.data

            if ob.p_verbose > 0:
                print("nx,ny,nz=", nx, ny, nz)
            if nx == 1:  # project onto X-axis view
                ##print("assigning X-axis UV map", uvmap.name)
                Bnewl = Vector(
                    ((B0l.y - B1l.y) / B1d.y, (B0l.z - B1l.z) / B1d.z)
                )
                Bnewu = Vector(
                    ((B0u.y - B1l.y) / B1d.y, (B0u.z - B1l.z) / B1d.z)
                )
                ##print("uv trim coords:", fv(Bnewl), fv(Bnewu))
                ud[0].uv = Vector((Bnewl.x, Bnewl.y))
                ud[1].uv = Vector((Bnewl.x, Bnewu.y))
                ud[2].uv = Vector((Bnewu.x, Bnewu.y))
                ud[3].uv = Vector((Bnewu.x, Bnewl.y))
            elif ny == 1:  # project onto Y-axis view
                ##print("assigning Y-axis UV map", uvmap.name)
                Bnewl = Vector(
                    ((B0l.x - B1l.x) / B1d.x, (B0l.z - B1l.z) / B1d.z)
                )
                Bnewu = Vector(
                    ((B0u.x - B1l.x) / B1d.x, (B0u.z - B1l.z) / B1d.z)
                )
                ##print("uv trim coords:", fv(Bnewl), fv(Bnewu))
                ud[0].uv = Vector((Bnewl.x, Bnewu.y))
                ud[1].uv = Vector((Bnewu.x, Bnewu.y))
                ud[2].uv = Vector((Bnewu.x, Bnewl.y))
                ud[3].uv = Vector((Bnewl.x, Bnewl.y))
            else:  # project onto Z-axis view
                ##print("assigning Z-axis UV map", uvmap.name)
                Bnewl = Vector(
                    ((B0l.x - B1l.x) / B1d.x, (B0l.y - B1l.y) / B1d.y)
                )
                Bnewu = Vector(
                    ((B0u.x - B1l.x) / B1d.x, (B0u.y - B1l.y) / B1d.y)
                )
                ##print("uv trim coords:", fv(Bnewl), fv(Bnewu))
                ud[0].uv = Vector((Bnewl.x, Bnewl.y))
                ud[1].uv = Vector((Bnewu.x, Bnewl.y))
                ud[2].uv = Vector((Bnewu.x, Bnewu.y))
                ud[3].uv = Vector((Bnewl.x, Bnewu.y))

            if self.probe_plane_frame_handler in posth:
                del posth[self.probe_plane_frame_handler]
            posth.append(self.probe_plane_frame_handler)

        else:  # 'Volume'
            # create H and E field arrow objects if needed
            H = coll_H.get()
            H.hide_viewport = False
            E = coll_E.get()
            E.hide_viewport = False
            n = (
                ((nx + sfactor - 1) // sfactor)
                * ((ny + sfactor - 1) // sfactor)
                * ((nz + sfactor - 1) // sfactor)
            )
            ne = n * 3
            print(f"probe size {nx}x{ny}x{nz} *3 = {ne} elements")
            if ne < 1 or ne > 80000:
                raise ValueError(f"probe: bad requested data size: {ne}")
            self.n = ne
            if len(ob.children) != n:
                if ob.p_verbose > 1:
                    print(f"Probe: creating {n} {field_name} arrows")
                dx2, collection = ((0, E), (dx / 2, H))[field_name == 'H']
                # Is is parent lower-left index in full grid, Pp is coords
                Pp = Vector((Is.i * dx, Is.j * dx, Is.k * dx))
                # D2 is 1/2 cell extra for H, + offset of parent lower left
                D2 = Vector((dx2, dx2, dx2)) + Vector(ob.bound_box[0])
                if ob.p_verbose > 1:
                    print(ob.name, "D2=", fv(D2), "Is=", Is)

                # delete just probe's arrow-children
                for arrow in ob.children:
                    objs.remove(arrow, do_unlink=True)

                # get or create the Tmp collection, used to delete all arrows
                tmpc = bpy.data.collections.get('Tmp')
                if not tmpc:
                    tmpc = bpy.data.collections.new('Tmp')
                    if not tmpc:
                        raise RuntimeError(
                            f"failed to create Tmp collection for {ob.name}"
                        )
                print(f"probe.vol {ob.name}: {tmpc=}")

                r = dx * 0.05
                h = dx * 0.5
                verts = (
                    (0, r, r),
                    (0, r, -r),
                    (0, -r, -r),
                    (0, -r, r),
                    (h, 0, 0),
                )
                faces = (
                    (1, 0, 4),
                    (4, 2, 1),
                    (4, 3, 2),
                    (4, 0, 3),
                    (0, 1, 2, 3),
                )
                # use common mesh, since Outliner now fixed
                mesh = bpy.data.meshes.new(name='Arrow')
                mesh.from_pydata(verts, [], faces)
                mesh.update()
                mat = bmats[f"Field{field_name}"]
                mesh.materials.append(mat)
                for i in range(0, nx, sfactor):
                    for j in range(0, ny, sfactor):
                        for k in range(0, nz, sfactor):
                            # arrow name must be in sortable format
                            name = (
                                f"{field_name}{Is.i+i:03}"
                                f"{Is.j+j:03}{Is.k+k:03}"
                            )
                            arrow = objs.new(name, mesh)
                            # loc relative to parent
                            arrow.location = (
                                dx * i + D2.x,
                                dx * j + D2.y,
                                dx * k + D2.z,
                            )
                            ##scn.objects.link(arrow)
                            arrow.parent = ob
                            collection.objects.link(arrow)
                            tmpc.objects.link(arrow)
            self.arrows = list(ob.children)
            self.arrows.sort(key=lambda arrow: arrow.name)
            if ob.p_verbose > 1:
                print(f"Probe: {len(self.arrows)} arrows")
                print("Probe: removing keyframes from old arrows")
            for arrow in self.arrows:
                if arrow.animation_data:
                    arrow.animation_data.action = None

        ##print("Probe.prepare_gen done.")

    def send_def_gen(self, update=False):
        yield
        self.send_def(update)

    def send_def(self, update=False):
        ##print("Probe.send_def_gen start ", "PU"[update])
        ob = self.ob
        Bs, Be = self.Bs, self.Be
        field_name = self.field_names_mag[ob.p_field]
        ob.p_axisSign = 1
        axis = ob.p_axis
        if axis[0] == '-':
            ob.p_axisSign = -1
            axis = ob.p_axis[1]
        disp_type = 'Vec'
        disp_scale = ob.p_dispScale
        if ob.p_shape == 'Plane':
            if axis == 'XYZ':
                disp_type = 'RGB'
            else:
                disp_type = 'Mag'
        elif ob.p_shape == 'Line':
            if axis in 'ZYX':
                disp_type = ob.p_axis
                ##self.n = self.n + 1  # voltage sources include edges ???
        elif ob.p_shape == 'Point':
            disp_type = 'Mag'
            disp_scale = ob.p_plotScale
            if axis == 'XYZ':
                disp_type = 'Vec'
            if field_name == 'V':
                field_name = 'E'
            if axis in 'ZYX':
                if ob.p_sum:
                    disp_type = 'Sum'
                iaxis = ord(axis) - ord('X')
                ##self.dist = (Be.x-Bs.x, Be.y-Bs.y, Be.z-Bs.z)[iaxis]
                N = self.N
                if 0:  # doesn't work with sum, which has to see full block
                    if iaxis == 0:
                        Be.x = Bs.x = (Be.x + Bs.x) / 2.0
                        n = N.j * N.k
                    elif iaxis == 1:
                        Be.y = Bs.y = (Be.y + Bs.y) / 2.0
                        n = N.i * N.k
                    else:
                        Be.z = Bs.z = (Be.z + Bs.z) / 2.0
                        n = N.i * N.j
                    self.n = n

        cmd = (
            f"{'PU'[update]} {ob.name} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
            f"{Bs.z:g} {Be.z:g} {field_name} {disp_type[0]} "
            f"{disp_scale:g} {ob.p_sfactor} {ob.p_verbose}"
        )
        if ob.p_verbose > 1:
            print(cmd)
        self.sim.send(cmd)

    def get_data_from_server(self):
        """Get data values from server for one step"""

        scn = bpy.context.scene
        ob = self.ob
        sim = self.sim

        # send probe request to server
        s = sim.s
        cmd = f"Q {ob.name}"
        if ob.p_verbose > 2:
            print("get_data_from_server:", ob.name, f"cmd='{cmd}'")
        ack = sim.send(cmd, 7)
        if len(ack) < 1 or ack[0] != ord('A'):
            ##print("non-A ack:", ack)
            if len(ack) == 1 and ack[0] == ord('D'):
                print("server has ended simulation")
                raise StopIteration
            print("probe: bad ack:", ack)
            return
        step = int(ack[1:])
        scn.frame_set(step)

        # receive data
        n = self.n
        if ob.p_sum:
            n = 3
        bdata = b''
        dtype = np.dtype(np.float32)
        esize = dtype.itemsize
        rem = n * esize
        ##print("n=", n, "esize=", esize, "rem=", rem, dtype)
        rnbytes = s.recv(6)
        if len(rnbytes) != 6:
            raise IOError(f"expected 6 digit data length, got '{rnbytes}'")
        rnbytes = int(rnbytes)
        if ob.p_verbose > 1:
            print(f"Probe: expecting {rem} bytes, receiving {rnbytes}")
        if rnbytes != rem:
            raise IOError(
                f"probe {ob.name} expected {rem} " f"bytes, got {rnbytes}"
            )
        for i in range(20):
            r = s.recv(n * esize)
            bdata += r
            rem -= len(r)
            if rem <= 0:
                break
        if rem > 0:
            raise IOError

        if step == self.last_step:
            if ob.p_verbose > 1:
                print("Probe last step", ob.name)
            return None
        self.last_step = step
        data = np.frombuffer(bdata, dtype=dtype)

        if ob.p_sum:
            iaxis = ord(ob.p_axis[-1]) - ord('X')
            data = data[iaxis : iaxis + 1]
            if ob.p_verbose > 1:
                print("data=", np.frombuffer(data, np.float32))

        if ob.p_verbose > 1:
            print(ob.name, "data=", data.shape, data.dtype, "as uint32:")
            np.set_printoptions(formatter={'all': lambda x: f"0x{x:08x}"})
            print(np.frombuffer(data, np.uint32))
            np.set_printoptions(formatter=None)
            print(np.frombuffer(data, np.float32))
        return data

    def do_step(self):
        """Step probe 1 timestep, getting values from server"""

        ob = self.ob
        if ob.p_shape == 'Line' and not (ob.p_sum or ob.p_avg):
            # no history stored for normal line probes
            return

        data = self.get_data_from_server()
        if data is None:
            return
        scn = bpy.context.scene
        sim = self.sim
        dx = sim.dx * mm
        step = scn.frame_current
        t = step * sim.dt
        n = self.n

        if ob.p_shape == 'Point':
            axis = ob.p_axis
            v = data[0]
            vbase = v
            if ob.type == 'EMPTY':
                if len(data) > 1:
                    V = Vector(data)
                    v = V
                    ##print("Point.EMPTY.data>1: v=", v)
                    ob.p_value3 = V
                    iaxis = 0
                    if axis == 'Magnitude':
                        v = V.length
                    elif axis != 'XYZ':
                        iaxis = ord(axis) - ord('X')
                        v = V[iaxis]
                else:
                    pass
            else:
                if ob.p_field == 'Voltage':
                    # Voltage should only depend on E and dx.
                    if ob.p_verbose > 1:
                        print("Point", ob.name, "v=", v, "dx=", dx)
                    v *= -dx * ob.p_axisSign

                ob.p_value = v

            if ob.p_verbose:
                Bs, Be = self.Bs, self.Be
                i = sim.grid(Bs.x)
                j = sim.grid(Bs.y)
                k = sim.grid(Bs.z)
                field_name = self.field_names_mag[ob.p_field]
                units = ("V/m", "A/m")[ob.p_field == 'Magnetic']
                print(
                    f"{step}: t={t:9.2e}: {ob.name} "
                    f"{field_name}[{i},{j},{k}]",
                    end="",
                )
                if axis == 'XYZ':
                    print(f"=({V.x:.4g},{V.y:.4g},{V.z:.4g}) {units}")
                else:
                    print(f".{ob.p_axis}={vbase:.4g}{units}")
            data = v

        elif ob.p_shape == 'Line':
            # summation, line integral of border
            data_d = np.frombuffer(data[0:6], dtype=np.dtype(np.float64))
            S = data_d[0:3]
            data_i = np.frombuffer(data, dtype=np.dtype(np.int32))
            count = data_i[-1]
            A = S / count
            area = count * dx**2  # m^2
            # sum of H~ is z0*H, in V/m
            H = S / z0  # in A/m
            if ob.p_sum:
                ob.p_value3 = H
                data = (S, count, A, area, H)
                ##self.drawChartStep(H)
            else:
                # line integral in S.x
                Hx = H[0]
                dl = count * dx
                I = Hx * dl
                if ob.p_verbose:
                    print(ob.name, "sum=", S, "dl=", dl, "H=", Hx, "I=", I)
                if sim.state < PAUSED:
                    ob.p_value3 = I
                    data = (S, dl, Hx, I)
                    ##self.drawChartStep(I)

        elif ob.p_shape == 'Plane':
            self.set_plane_texture(data)
            ##print(f"storing history[{sstep}]")

        else:  # 'Volume'
            data = data.reshape((n // 3, 3))
            if 0:  # generate test data
                HEr = []
                for x in range(nx):
                    for y in range(ny):
                        for z in range(nz):
                            HEr.append((x / nx, y / ny, z / nz))
                for x in range(nx):
                    for y in range(ny):
                        for z in range(nz):
                            HEr.append((-x / nx, -y / ny, -z / nz))
                HE = np.array(HEr) * 32768 / 100.0

            log_mag, mag_scale = ob.p_log, ob.p_magScale
            for i, arrow in enumerate(self.arrows):
                arrow.rotation_euler.zero()
                pr = False
                ##pr = arrow.name in ('H040605', 'E040605')
                x, y, z = data[i]
                r2 = x * x + y * y + z * z
                r2 *= mag_scale**2
                ##arrow.show_name = r2 > 0
                if pr:
                    di = data[i]
                    print(
                        f"{step}: xyz=({di[0]:g},{di[1]:g},{di[2]:g})"
                        f"=({x:6.4}, {y:6.4}, {z:6.4}) r2={r2:6.4}"
                    )
                if r2 > 1e-12:
                    r = ma.sqrt(r2)
                    if log_mag:
                        ##r = 0.3*(ma.log(r)+4.7)
                        r = 0.25 * (ma.log10(r) + 6)
                    else:
                        if r > 30:
                            r = 30
                    r *= ob.p_sfactor
                    arrow.scale = (r, r, r)
                    if pr:
                        print(
                            f"{step}: {arrow.name} "
                            f"xyz=({x:6.4}, {y:6.4}, {z:6.4}) "
                            f"r={r:6.4} {mag_scale}"
                        )
                    M = Matrix(((x, 0, 0), (y, 0, 0), (z, 0, 0)))
                    # rely on rotate() to normalize matrix
                    arrow.rotation_euler.rotate(M)
                else:
                    arrow.scale = (0, 0, 0)
                if sim.rec:
                    arrow.keyframe_insert(data_path='rotation_euler')
                    arrow.keyframe_insert(data_path='scale')
                    ##arrow.keyframe_insert(data_path='show_name')

        # record data history
        if not isinstance(data, (int, float, tuple)):
            data = data.copy()
        ##print("Probe.do_step:", ob.name, "step", step, "data=", data)
        self.history[step] = data

    @classmethod
    def plot_all_set_up(self):
        self.figs = {}

    def get_plot_fig(self, data_shape):
        ob = self.ob
        color = None
        ##fig_type = ob.p_shape + ob.p_field + ob.p_axis + repr(data_shape)
        fig_type = ob.p_shape + ob.p_field + repr(data_shape)
        fig = self.figs.get(fig_type)
        if not fig:
            fig = Figure()
            self.figs[fig_type] = fig
            if ob.material_slots:
                mat = ob.material_slots[0].material
                color = mat.diffuse_color
            plt.figure(fig.figure.number)
            fig.title = ob.name
        return fig, color

    def plot(self):
        """Cmd-P: Plot all point probe value histories"""
        ob = self.ob
        sim = self.sim
        if ob.p_verbose > 0:
            print("Probe.plot", ob.name)
        if (ob.p_shape == 'Point' and self.history) or ob.p_shape == 'Line':
            if ob.p_shape == 'Point':
                # add plot of point probe value history to the plot
                # in separate arrays, otherwise numpy converts int keys to
                # floats and argsort may fail to sort those properly
                keys = list(self.history.keys())
                values = list(self.history.values())
                if type(values[0]) == Vector:
                    if ob.p_axis in 'ZYX':
                        ix = ob.p_axis - ord('X')
                        values = [V[ix] for V in values]
                    else:
                        values = [V.length for V in values]
                hk = np.array(keys)
                hv = np.array(values)
                # sort time keys, returning indices
                hi = hk[:].argsort()
                xs = hk[hi]
                ys = hv[hi]
                xs = xs.astype(np.double) * sim.dt
                fig, color = self.get_plot_fig(1)
                fig.xlabel = "Time"
                fig.xunit = "s"
            else:
                # 'Line'
                self.last_step = -1  # force data grab
                ys = self.get_data_from_server()
                Bs, Be = self.Bs, self.Be
                N = self.N
                s = Bs.x
                e = Be.x
                fig, color = self.get_plot_fig(N)
                if N.j > 1:
                    s = Bs.y
                    e = Be.y
                    fig.xlabel = "Y"
                elif N.k > 1:
                    s = Bs.z
                    e = Be.z
                    fig.xlabel = "Z"
                else:
                    fig.xlabel = "X"
                fig.xunit = "m"
                xs = np.linspace(s * mm, e * mm, self.n)
                ##print("Line plot: s=", s, "e=", e, "xs=", xs)

            marker = '.' if len(ys) < 50 else None
            label = ob.name
            if ob.p_plotScale != 1:
                label = f"{label} * {ob.p_plotScale:g}"
            plt.plot(
                xs.copy(), ys.copy(), marker=marker, color=color, label=label
            )
            plt.legend(loc=ob.p_legendLoc)
            fn = ob.p_field
            fig.ylabel = f"{fn.capitalize()} (%s{self.fieldUnits[fn[0]]})"
            fig.max_x = max(xs[-1], fig.max_x)
            fig.min_y = min(ys.min(), fig.min_y)
            fig.max_y = max(ys.max(), fig.max_y)

        else:
            print("not plottable")

    @classmethod
    def plot_all_finish(self):
        if not self.figs:
            print("no plottable probes")
            return

        for fig in self.figs.values():
            fnum = fig.figure.number
            print(f"plotting figure {fnum}")
            plt.figure(fnum)
            tm = time.localtime(time.time())
            title = (
                f"{tm.tm_year-2000:02}{tm.tm_mon:02}{tm.tm_mday:02}"
                f"-{fnum:02}-{fig.title.replace('.', '-')}"
            )
            fm = plt.get_current_fig_manager()
            fm.set_window_title(title)
            ##plt.title(title)
            # show the plot in a separate window
            plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
            x_si = si.si(fig.max_x)
            range_y = fig.max_y - fig.min_y
            y_si = si.si(range_y)
            units_x = x_si[-1]
            if units_x.isdigit():
                units_x = ''
            units_y = y_si[-1]
            if units_y.isdigit():
                units_y = ''
            scale_x = si.SIPowers.get(units_x)
            scale_y = si.SIPowers.get(units_y)
            ##print(f"{fig.min_y=} {fig.max_y=} {range_y=} {y_si=} "
            ##      f"{units_y=} {scale_y=}")
            if scale_x is not None:
                ##print(f"Scaling plot X axis by {10**scale_x:g} ({units_x})")
                if 0:
                    ticks_x = ticker.FuncFormatter(
                        lambda x, pos: '{0:g}'.format(x / 10**scale_x)
                    )
                    fig.ax.xaxis.set_major_formatter(ticks_x)
                else:
                    fig.ax.xaxis.set_major_formatter(NFmtr(scale_x))
            else:
                units_x = ''
            if scale_y is not None:
                fig.ax.yaxis.set_major_formatter(NFmtr(scale_y))
            else:
                units_y = ''
            plt.xlabel(f"{fig.xlabel} ({units_x}{fig.xunit})")
            yl = fig.ylabel % units_y
            yl = yl.replace("kV/m", "V/mm").replace("kA/m", "A/mm")
            plt.ylabel(yl)
            plt.grid(True)
            plt.subplots_adjust(left=0.15, top=0.95, right=0.95)
            if not is_linux:
                plt.show(block=False)
            output_dir = os.path.join(cwd, 'output')
            print(f"writing plots to {output_dir}")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            plt.savefig(os.path.join(output_dir, title))


block_classes = {
    'FIELDS': FieldsBlock,
    'MATCUBE': MatBlock,
    'MATCYLINDERX': MatBlock,
    'MATCYLINDERY': MatBlock,
    'MATCYLINDERZ': MatBlock,
    'MATMESH': MeshMatBlock,
    'MATLAYER': LayerMatBlock,
    'RESISTOR': Resistor,
    'CAPACITOR': Capacitor,
    'PROBE': Probe,
    'SOURCE': Source,
    'SUBSPACE': SubSpaceBlock,
}


class Sim:
    s = None
    mouse_pos = None

    def __init__(self, context):
        global sims
        sims[context.scene] = self
        bpy.app.driver_namespace['fields'] = sims

        obs = [
            ob
            for ob in context.visible_objects
            if ob.name.startswith("0Fields")
        ]
        if len(obs) != 1:
            raise KeyError("expected to find one '0Fields' object")
        preob = obs[0]
        self.state = INITIALIZING
        self.dx = preob.dx
        self.fields_block = FieldsBlock(preob, self)
        self.fields_ob = fob = self.fields_block.ob
        self.verbose = fob.verbose
        print("Verbosity level", self.verbose)
        if self.verbose > 0:
            print("Sim init: have fob", fob.name)
        self.tms = int(time.time() * 1000)
        ##print(f"sim {id(self)} init: state=0")
        self.history = {}
        self.dt = 0.0
        self.active_ob = context.object
        bpy.app.handlers.frame_change_post.clear()
        self.on_screen_init(context)
        self.gen = self.step_whole_fdtd_gen()

    def start_fdtd(self):
        """Open a connection to the server, as self.s"""

        import socket

        HOST = "localhost"  # The remote host
        PORT = 50007  # The same port as used by the server
        print("Looking for field server")
        self.s = s = None
        for res in socket.getaddrinfo(
            HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM
        ):
            af, socktype, proto, canonname, sa = res
            try:
                s = socket.socket(af, socktype, proto)
                s.settimeout(timeout)

            except socket.error as msg:
                s = None
                continue
            try:
                s.connect(sa)
            except socket.error as msg:
                s.close()
                s = None
                continue
            break
        if s is None:
            print("could not open socket")
            raise IOError

        self.s = s
        print("Connected")

    def send(self, text, nrecv=None, check=True):
        """Send a command to the FDTD server, returning nrecv bytes.
        The default is to expect an 'A' ack.
        """
        s = self.s
        s.send(text.encode("utf-8"))
        if nrecv is None:
            r = s.recv(1)
            if r == b'N':
                # server got an error: report it and stop (server doesn't)
                err_len = int(s.recv(2))
                err_msg = s.recv(err_len).decode("utf-8")
                raise RuntimeError(err_msg)
            elif check and r != b'A':
                raise IOError(f"Expected 'A' ack but got '{r}'")
        else:
            r = s.recv(nrecv)
        return r

    def grid(self, x):
        return round(x / self.dx)

    def new_material(self, name, value, alpha=None):
        """Create a material if it doesn't exist, given its name,
        (color, mur, epr, sige), and transparency.
        """
        bmats = bpy.data.materials
        m = bmats.get(name)
        if m is None:
            m = bmats.new(name=name)
        (m.diffuse_color, m['mur'], m['epr'], m['sige']) = value
        m.use_fake_user = True
        if alpha is None:
            alpha = m.diffuse_color[3]
        if alpha < 0.99:
            m.use_nodes = True
            ##print(f"{m.name}: setting alpha to {alpha}")
            node = m.node_tree.nodes["Principled BSDF"]
            node.inputs[0].default_value = m.diffuse_color
            node.inputs['Alpha'].default_value = alpha
            m.blend_method = 'BLEND'

    def get_materials_and_dims(self):
        # material name, color [R,G,B,A], mur, epr, sige
        mats = {
            'FieldE': ((0, 0, 1, 1.0), 0, 0, 0),
            'FieldH': ((1, 0, 0, 1.0), 0, 0, 0),
            'FieldM': ((0, 1, 0, 1.0), 0, 0, 0),
            'FieldJ': ((0, 1, 1, 1.0), 0, 0, 0),
            'Air': ((0.5, 0.5, 1, 0.1), 1.0, 1.0, 0.0),
        }
        mats_tr = {
            'Copper': ((0.45, 0.14, 0.06, 1.0), 1.0, 1.0, 9.8e7),
            'CopperLC': ((0.45, 0.14, 0.06, 1.0), 1.0, 1.0, 5.8e7),  # LC
            'Metal': ((0.45, 0.14, 0.06, 1.0), 1.0, 1.0, 3.27e7),  # LC
            'CopperTinned': ((0.42, 0.42, 0.42, 1.0), 1.0, 1.0, 9.8e7),
            'Solder': ((0.42, 0.42, 0.42, 1.0), 1.0, 1.0, 1.5e7),
            'SolderLC': ((0.42, 0.42, 0.42, 1.0), 1.0, 1.0, 7e6),  # LC
            'Brass': ((0.45, 0.30, 0.03, 1.0), 1.0, 1.0, 7e7),
            'BrassLC': ((0.45, 0.30, 0.03, 1.0), 1.0, 1.0, 1.5e7),  # LC
            'Teflon': ((0.99, 0.80, 0.78, 1.0), 1.0, 2.8, 0.0),  # LC
            'Epoxy': ((0.05, 0.05, 0.05, 1.0), 1.0, 2.7, 0.0),
            'FR4': ((0.73, 0.80, 0.39, 1.0), 1.0, 4.4, 0.0),
            'Ceramic': ((0.81, 0.75, 0.60, 1.0), 1.0, 38.0, 0.0),
            'Porcelain': ((0.81, 0.75, 0.60, 1.0), 1.0, 5.0, 1e-13),  # LC
            'Ferrite': ((0.81, 0.75, 0.60, 1.0), 1000.0, 1.0, 0.01),
            'Iron': ((0.81, 0.75, 0.60, 1.0), 3800.0, 1.0, 1e6),
            'Cast Iron': ((0.81, 0.75, 0.60, 1.0), 60.0, 1.0, 0.0),  # LC
        }
        for name, value in mats.items():
            self.new_material(name, value)
        for name, value in mats_tr.items():
            self.new_material(name, value, 1.0)
            self.new_material(f"{name}-T", value, 0.3)

        # get simulation area dimensions from parent Fields object
        scn = bpy.context.scene
        ob = self.fields_ob
        for key, value in FieldsBlock.props.items():
            setattr(self, key, getattr(ob, key))
        self.nx = ma.floor(ob.dimensions.x / ob.dx)
        self.ny = ma.floor(ob.dimensions.y / ob.dx)
        self.nz = ma.floor(ob.dimensions.z / ob.dx)
        if ob.verbose > 0:
            print("Fields nx,ny,nz=", self.nx, self.ny, self.nz)

        # start recording
        if ob.rec:
            scn.frame_set(1)
            self.frame_no = 1

    def create_blocks_gen(self):
        """Generator to create sim-blocks for all visible blocks within Fields."""
        fob = self.fields_ob
        if fob.verbose > 0:
            print("create_blocks_gen start")
        self.blocks = [self.fields_block]
        obs = list(bpy.context.visible_objects)
        obs.sort(key=lambda ob: ob.name)
        scn = bpy.context.scene
        coll_main.get().hide_viewport = False
        ##coll_snap.get().hide_viewport = True
        have_snap = False

        # create a block for each object
        for ob in obs:
            try:
                name = ob.name
            except ReferenceError:
                print("createBlocks: object reference error:", ob)
                continue

            if fob.verbose > 1:
                print("createBlocks object", ob.name)
            block = None
            bt = ob.get('blockType')

            block_class = block_classes.get(bt)
            if fob.verbose > 1:
                print(
                    "createBlocks:", ob.name, bt, block_class, ob.hide_viewport
                )
            if ob.snap:
                if fob.verbose > 1:
                    print("Block", ob.name, "is snap")
                have_snap = True
            if block_class and not ob.hide_viewport:
                if fob.verbose > 1:
                    print("  createBlocks verified", ob.name)
                over = overlap(fob, ob)
                if not over:
                    if fob.verbose > 1:
                        print("no overlap for", ob.name)
                if over:
                    # create block for object or optional snapped object
                    block = block_class(ob, self)
                    ob = block.ob
                    if fob.verbose > 1:
                        print("=====", ob.name, bt, block_class)
                        print("  overlap=", fv(over[0]), fv(over[1]))
                    for _ in block.prepare_gen():
                        yield
                    self.blocks.append(block)

        scn.frame_set(1)
        if have_snap:
            if fob.verbose > 1:
                print("createBlocks end: have_snap")
            ##coll_main.get().hide_viewport = True
            coll_snap.get().hide_viewport = False

    def find_block(self, ob):
        for block in self.blocks:
            if block.ob == ob:
                return block
        return None

    def send_f_material(self, ob, m):
        self.fmats.append(m)
        if ob.verbose > 1:
            print(" adding fmat", m.name)
        self.send(f"M {m.name} {m.mur:g} {m.epr:g} {m.sige:g} 0")

    def send_defs_gen(self):
        ##print("Sim.send_defs_gen start")

        # collect and send defs for Field Materials used by sim objects
        self.fmats = fmats = []
        for block in self.blocks:
            ob = block.ob
            if ob.verbose > 1:
                print("checking", ob.name, "for FDTD material")
            link = 'DATA'
            if ob.material_slots:
                link = ob.material_slots[0].link
            m = None
            if link == 'DATA':
                if ob.data and ob.data.materials:
                    if ob.blockType == 'MATLAYER':
                        m = bpy.data.materials.get(ob.fmat_name)
                    else:
                        m = ob.data.materials[0]
                        if not m and ob.material_slots:
                            m = ob.material_slots[0].material
            else:
                if ob.material_slots:
                    m = ob.material_slots[0].material

            if m and not m in fmats:
                self.send_f_material(ob, m)

        # send general sim parameters to server
        for _ in self.fields_block.send_sim_def_gen():
            yield

        # send defintions for sim objects
        for block in self.blocks:
            if block != self.fields_block:
                for _ in block.send_def_gen():
                    yield

    def do_step_blocks(self):
        """Process one timestep of each sim block"""

        scn = bpy.context.scene
        if self.verbose > 0:
            tms = int(time.time() * 1000)
            dtms = tms - self.tms
            self.tms = tms
            print(f"[{dtms}>{scn.frame_current}:] ", end="")
            sys.stdout.flush()
        rec = self.rec
        if rec:
            scn.frame_set(self.frame_no)
            self.frame_no += 1
        for block in self.blocks:
            if hasattr(block, 'do_step') and not block.ob.hide_viewport:
                block.do_step()
            stop_ps = self.fields_ob.get('stop_ps', 0)
            if stop_ps > 100000:
                # sim sends a 6-digit step number back with each ack
                print("**** Error: stop time limited to 100000 max.")
            current_ps = scn.frame_current * self.dt * 1e12
            ##print(f"{stop_ps=} {self.frame_no=} {self.dt=} {current_ps}")
            if stop_ps > 0 and current_ps > stop_ps:
                print("Reached stop time")
                field_operator.cancel(bpy.context)
            if self.state == STOPPED:
                ##print("do_step_blocks stopped")
                break
        scn.frame_set(scn.frame_current + 1)

    def step_whole_fdtd_gen(self):
        """Do one timer step: first initialize simulation, then step it"""

        ##print("step_whole_fdtd_gen start")
        self.get_materials_and_dims()

        try:
            self.start_fdtd()
        except IOError:
            start_xcode()
            time.sleep(5)
            self.start_fdtd()

        # tell simulator to chdir to blender file's directory
        cmd = f"C {cwd}"
        if self.verbose > 1:
            print(cmd)
        self.send(cmd)
        yield

        for _ in self.create_blocks_gen():
            yield
        for _ in self.send_defs_gen():
            yield
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = self.active_ob
        if self.active_ob:
            self.active_ob.select_set(True)
        self.dt = 0.5 * self.dx * mm / c0
        print(f"dx = {self.dx:0.3f} mm, dt = {self.dt / 1e-12:0.3g} ps")
        try:
            self.send('R')
        except IOError:
            print("*** Start Field Server first! ***")
            raise
        ##print(f"sim {id(self)} step_whole_fdtd_gen: set state=3, looping")
        self.state = RUNNING
        while self.state != STOPPED:
            yield
            if self.state == RUNNING:
                self.do_step_blocks()
        ##print("step_whole_fdtd_gen done")

    def pause(self, context):
        """Tell the server to pause/unpause simulation"""

        context.area.tag_redraw()  # update status line
        ack = self.send('D', 1)
        if len(ack) < 1 or ack[0] != ord('A'):
            ##print("non-A ack:", ack)
            return

    def on_screen_init(self, context):
        """On-screen status display"""

        # TODO: make on_screen_init work from "Run FDTD" button too
        area = context.area
        ##print(f"on_screen_init {id(self)} {area=}")
        if area and area.type == 'VIEW_3D':
            self.last_pos_meas = None
            oldh = bpy.app.driver_namespace.get('fields_handle')
            ##print(f"on_screen_init: in 3D area, {oldh=}")
            if oldh:
                bpy.types.SpaceView3D.draw_handler_remove(oldh, 'WINDOW')
            args = (context,)
            self.handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback, args, 'WINDOW', 'POST_PIXEL'
            )
            posth = bpy.app.handlers.frame_change_post
            if self.frame_handler in posth:
                ##print("on_screen_init: deleting old post handler")
                del posth[self.frame_handler]
            posth.append(self.frame_handler)
            self.area3d = context.area
            bpy.app.driver_namespace['fields_handle'] = self.handle
            bpy.app.handlers.load_pre.append(self.on_screen_remove)

    def draw_callback(self, context):
        ##try:

        ##print("trying draw status")
        scn = bpy.context.scene
        font_id = 0
        w = context.region.width
        font_scale = (2, 1)[is_linux]
        blf.position(font_id, w - 200 * font_scale, 10, 0)
        blf.color(font_id, 255, 255, 255, 255)
        blf.size(font_id, 12 * font_scale)
        status = f"{scn.frame_current * self.dt * 1e12:9.3f} ps"
        if self:
            ##print(f"sim {id(self)} draw_callback: state={self.state}")
            if self.state == PAUSED:
                status += '  PAUSED'
            elif self.state == STOPPED:
                status += '  STOPPED'
        blf.draw(font_id, status)

        # draw mobile-probe measurement value next to mouse pointer
        mpos = self.mouse_pos
        ob = bpy.context.object
        have_drawn_meas = False
        if mpos and ob:
            pblock = self.find_block(ob)
            if hasattr(pblock, 'measurement_attr_name'):
                aname = pblock.measurement_attr_name
                units = pblock.measurement_units
                if hasattr(ob, aname):
                    meas = getattr(ob, aname)
                    ##print(f"drawing measurement {meas:g}")
                    x = mpos[0] + 15
                    y = mpos[1] + 15
                    blf.position(font_id, x, y, 0)
                    self.last_pos_meas = [x, y]
                    blf.draw(font_id, f"{meas:g} {units}")
                    have_drawn_meas = True

        # erase measurement when another object is selected
        if not have_drawn_meas and self.last_pos_meas is not None:
            ##print("clearing measurement")
            x, y = self.last_pos_meas
            blf.position(font_id, x, y, 0)
            blf.draw(font_id, "")
            self.last_pos_meas = None

    def frame_handler(self, scene, depsgraph):
        ##print(f"frame_handler {id(self)}")
        self.area3d.tag_redraw()  # update status line

    def on_screen_remove(self, dummy1, dummy2):
        ##print(f"on_screen_remove {id(self)}")
        bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
        bpy.app.driver_namespace.pop('fields_handle')
        bpy.context.area.tag_redraw()


class FieldOperator(bpy.types.Operator):
    """Run FDTD simulation (Cmd-R)"""

    bl_idname = "fdtd.run"
    bl_label = "Run FDTD"

    timer = None
    sim = None

    def start_timer(self):
        """Timer routines for modal operator"""

        # Extra timers running in Blender are a pain. This happens
        # whenever bfield.py crashes while running, and there’s no way
        # to remove all previous timers, nor to distinguish which
        # timer sent the ‘TIMER’ event to compare to. Nor is there a
        # way to change the rate of the current timer, which would be
        # useful to speed up the initialization period where there are
        # a bunch of yields but the rate doesn’t need to be limited
        # then.

        sim = self.sim
        if not sim:
            return
        context = self.context
        context.window_manager.modal_handler_add(self)
        rate = sim.fields_ob.get('msRate') or 200
        rate = max(min(rate, 1000), 10)
        print(f"starting {rate} ms/tick timer")
        self.timer = context.window_manager.event_timer_add(
            rate / 1000.0, window=context.window
        )

    def stop_timer(self):
        self.context.window_manager.event_timer_remove(self.timer)
        self.timer = None

    def dyn_probe(self, event, action):
        """Dynamic probing"""

        ob = bpy.context.object
        name = ob.name if ob else "<no object>"
        ##print("dyn_probe:", action, name, ob.p_verbose)
        if ob and ob.p_verbose > 0:
            print("dyn_probe:", action, name)
        sim = self.sim
        if ob and ob.get('blockType') == 'PROBE':
            pblock = sim.find_block(ob)
            ##print("found block", pblock)
            if pblock:
                loc = get_mouse_3d(event)
                sim.mouse_pos = [event.mouse_region_x, event.mouse_region_y]
                if loc is None:
                    print(
                        f"No C.space_data: {event.type=} {event.value=} "
                        f"{event.mouse_region_x=}"
                    )
                else:
                    if action == 'START':
                        if ob.p_verbose > 0:
                            print("dyn_probe: starting move", ob.name)
                        self.probe_drag = ob
                        self.ob_rel_mouse = Vector(ob.location) - Vector(loc)
                    elif action == 'MOVE':
                        if ob.p_verbose > 0:
                            print("dyn_probe: moving", ob.name, pblock)
                        # ob_new_loc = self.ob_rel_mouse + Vector(loc)
                        ###print("ob_new_loc=", fv(ob_new_loc))
                        # if self.lock_axis == 0:
                        #    ob.location.x = ob_new_loc.x
                        # elif self.lock_axis == 1:
                        #    ob.location.y = ob_new_loc.y
                        # elif self.lock_axis == 2:
                        #    ob.location.z = ob_new_loc.z
                        # else:
                        #    ob.location = ob_new_loc
                        pblock.Bs, pblock.Be = bounds(ob)
                        pblock.send_def(update=True)
                        pblock.last_step -= 1
                        pblock.do_step()
                    else:
                        if ob.p_verbose > 1:
                            print("dyn_probe: move done")
                        self.probe_drag = None
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        sim = self.sim
        Pm = Vector((event.mouse_x, event.mouse_y))
        S = self.win_start
        E = self.win_end
        in_win = Pm.x >= S.x and Pm.x < E.x and Pm.y >= S.y and Pm.y < E.y
        ##print("mouse @", fv(Pm), in_win, fv(S), fv(E))

        ##if in_win and not event.type in ('TIMER'):
        ##    print("event:", event.value, event.type, "oskey=", event.oskey)

        if sim.verbose > 1 and event.type == 'P':
            print(f"'P', oskey={event.oskey} value={event.value}")
        if event.value == 'PRESS':
            if not in_win:
                return {'PASS_THROUGH'}

            if self.probe_drag:
                if event.type in ('LEFTMOUSE', 'RET', 'ESC'):
                    return self.dyn_probe(event, 'DONE')

            if sim.state >= RUNNING:
                ob = bpy.context.object
                if event.type == 'G':
                    if ob.blockType == 'PROBE' and ob.p_shape == 'Point':
                        self.lock_axis = None
                        self.dyn_probe(event, 'START')
                    return {'PASS_THROUGH'}

                elif event.type in 'XYZ':
                    if ob.blockType == 'PROBE' and ob.p_shape == 'Point':
                        self.lock_axis = ord(event.type) - ord('X')
                        # return {'RUNNING_MODAL'}
                    return {'PASS_THROUGH'}

                elif event.type == 'ESC':
                    print("ESC pressed while in sim")
                    return self.cancel(context)
                    # refresh viewport to update status line
                    # note: kills context.area, used in on_screen_init
                    # also prints "Warning: 1 x Draw window and swap..."
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            elif event.type == 'ESC':
                print("ESC pressed")
                return self.cancel(context)

        elif event.value == 'RELEASE':
            ##print(f"Release: {in_win=} {self.probe_drag=} {event.type=}")
            if not in_win:
                return {'PASS_THROUGH'}
            if self.probe_drag:
                # if event.type == 'MOUSEMOVE':
                #    return self.dyn_probe(event, 'MOVE')
                if event.type in ('LEFTMOUSE', 'RET', 'ESC'):
                    return self.dyn_probe(event, 'DONE')

        elif event.type == 'TIMER':
            if self.timer is None:
                return {'CANCELLED'}
            else:
                ##print(f"Timer: {in_win=} {self.probe_drag=} {event.type=}")
                if in_win and self.probe_drag:
                    return self.dyn_probe(event, 'MOVE')

                try:
                    sim.gen.__next__()
                except StopIteration:
                    if sim.verbose > 1:
                        print("modal: timer: step_whole_fdtd_gen done")
                except IOError:
                    print("Timer IOError exception")
                    if sim.s is None:
                        # server not found
                        print("Timer IOError: server not found")
                        return self.cancel(context)
                    else:
                        # usually a timeout: show where
                        print("Timer IOError: timeout?")
                        self.cancel(context)
                        raise
                except Exception as e:
                    print("Timer general exception")
                    self.cancel(context)
                    raise
                    ##self.report('ERROR', str(e))
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        global sims, field_operator
        field_operator = self
        sim = sims.get(context.scene)
        print("\n=== Starting BField FDTD simulation ===")
        self.context = context
        ##print(f" {context.area=}")
        winrgn = context.area.regions[-1]
        self.win_start = S = Vector((winrgn.x, winrgn.y))
        self.win_end = E = self.win_start + Vector(
            (winrgn.width, winrgn.height)
        )
        ##print("win @", fv(S), fv(E))
        if sim:
            if sim.state != STOPPED:
                print("Stopping current sim")
                sim.operator.cancel(context)
        self.sim = Sim(context)
        self.sim.operator = self
        self.probe_drag = None
        self.start_timer()
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ##print(f"FDTD execute: {context=}")
        return self.invoke(context, None)

    def cancel(self, context):
        ##print(f"cancel start: {context.area=}")
        sim = self.sim
        scn = context.scene
        s = sim.s
        if s:
            sim.send('Ex', check=False)
            s.close()
        if self.timer:
            self.stop_timer()
        ##print(f"sim {id(sim)} cancel: state=0")
        sim.state = STOPPED
        scn.frame_end = scn.frame_current

        print("FDTD stopped.")
        return {'CANCELLED'}


def clean_tmps():
    """Remove all probe-generated objects and images"""

    tmpc = bpy.data.collections.get('Tmp')
    if tmpc:
        objs = bpy.data.objects
        for ob in tmpc.all_objects.values():
            objs.remove(ob, do_unlink=True)

    # imgs = bpy.data.images
    # for name,img in imgs.items():
    #    if name.startswith('probe_'):
    #        imgs.remove(img, do_unlink=True)

    bpy.app.handlers.frame_change_post.clear()


class FieldCleanOperator(bpy.types.Operator):
    """Clean up after FDTD simulation (Cmd-K)"""

    bl_idname = "fdtd.clean"
    bl_label = "Clean FDTD: remove probe result objects"

    def invoke(self, context, event):
        print("Clean-FDTD invoke")
        clean_tmps()
        return {'FINISHED'}


class FieldPauseOperator(bpy.types.Operator):
    """Pause/unpause FDTD simulation (P)"""

    bl_idname = "fdtd.pause"
    bl_label = "Pause"

    def invoke(self, context, event):
        global sims
        sim = sims[context.scene]
        ##print("Pause-FDTD invoke")
        if not sim or sim.state < RUNNING:
            print("FDTD not running")
        elif sim.state == RUNNING:
            print("=== PAUSE ===")
            sim.pause(context)
            sim.state = PAUSED
        else:
            print("=== UNPAUSE ===")
            sim.pause(context)
            sim.state = RUNNING
        return {'FINISHED'}

    def execute(self, context):
        ##print("Pause-FDTD execute")
        return self.invoke(context, None)


class FieldPlotOperator(bpy.types.Operator):
    """Plot probes in FDTD simulation (Cmd-P)"""

    bl_idname = "fdtd.plot"
    bl_label = "Plot probes"

    def invoke(self, context, event):
        global sims
        sim = sims[context.scene]
        ##print("Plot-FDTD invoke")
        print("\nCmd-P: plot")
        if not sim:
            print("FDTD hasn't been run")
            return {'FINISHED'}
        Probe.plot_all_set_up()
        for block in sim.blocks:
            if type(block) == Probe:
                ob = block.ob
                block.plot()
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # wait for Cmd key to be released, or plot window messes oskey state
        if is_linux or (event.type == 'OSKEY' and event.value == 'RELEASE'):
            print("finishing plot")
            Probe.plot_all_finish()
            return {'FINISHED'}
        return {'PASS_THROUGH'}


class FieldCenterOperator(bpy.types.Operator):
    """Center object origin"""

    bl_idname = "fdtd.center"
    bl_label = "Center object origin"

    def invoke(self, context, event):
        print("FDTD Center invoke")
        op = bpy.ops.object
        op.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        return {'FINISHED'}


def export_scene_x3d():
    # unhide all collections and objects, keeping lists
    hidden = []
    hidden_vp = []
    coll_hidden_vp = []
    for coll in bpy.data.collections:
        if coll.hide_viewport:
            coll_hidden_vp.append(coll)
            coll.hide_viewport = False
    for ob in bpy.data.objects:
        if ob.hide_get():
            hidden.append(ob)
            ob.hide_set(False)
        if ob.hide_viewport:
            hidden_vp.append(ob)
            ob.hide_viewport = False

    # refresh viewport to update status line
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    # export as x3d file
    dir = os.path.join(cwd, 'scenes_x3d')
    if not os.path.exists(dir):
        os.mkdir(dir)
    base = os.path.basename(bpy.data.filepath)
    base = os.path.splitext(base)[0]
    name = f"{base}_{bpy.context.scene.name}.x3d"
    ##print(f"Writing {name} to {dir}")
    filepath = os.path.join(dir, name)
    bpy.ops.export_scene.x3d(filepath=filepath)

    # rehide hidden collections and objects
    for ob in hidden:
        ob.hide_set(True)
    for ob in hidden_vp:
        ob.hide_viewport = True
    for coll in coll_hidden_vp:
        coll.hide_viewport = True


class FieldExportOperator(bpy.types.Operator):
    """Export scene as .x3d file"""

    bl_idname = "fdtd.export"
    bl_label = "Export scene as x3d"

    def invoke(self, context, event):
        export_scene_x3d()
        return {'FINISHED'}


class FieldExportAllOperator(bpy.types.Operator):
    """Export all scenes as .x3d files"""

    bl_idname = "fdtd.export_all"
    bl_label = "Export all as x3ds"

    def invoke(self, context, event):
        cur_scene = bpy.context.window.scene
        for scene in bpy.data.scenes:
            bpy.context.window.scene = scene
            export_scene_x3d()
        bpy.context.window.scene = cur_scene
        return {'FINISHED'}


class FieldObjectPanel(bpy.types.Panel):
    """Creates a FDTD Panel in the Object properties window"""

    bl_label = "FDTD"
    bl_idname = "OBJECT_PT_FDTD"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        ob = context.object
        fob = get_fields_ob(bpy.context, create=False)
        layout.prop_search(
            ob, "blockType", scene, "block_types", text="Block type"
        )
        bt = ob.blockType
        if bt:
            block_classes[bt].draw_props(ob, layout, scene)
        else:
            if ob.material_slots:
                mat = ob.material_slots[0]
                if mat.name in ('FieldH', 'FieldE'):
                    field = mat.name[-1]
                    units = Probe.fieldUnits[('E', 'M')[field == 'H']]
                    Mr = ob.rotation_euler.to_matrix()
                    V = Mr @ Vector((1, 0, 0))
                    ap = ob.parent
                    r = ob.scale.x / ap.p_sfactor
                    if ap.p_log:
                        r = 10 ** (r * 4 - 6)
                    r /= ap.p_magScale
                    V.x *= r
                    V.y *= r
                    V.z *= r
                    box = layout.box()
                    box.label(text=f"{field} = {gv(V)} {units}")
                    box.label(text=f"   = {V.length:g} {units}")

        spit = layout.split()
        col = spit.column()
        col.operator("fdtd.run")
        col.operator("fdtd.pause")
        col = spit.column()
        if fob:
            col.prop(fob, "stop_ps", text="Stop ps")
        col.operator("fdtd.plot")

        layout.operator("fdtd.clean")
        layout.operator("fdtd.center")
        layout.operator("fdtd.export")
        layout.operator("fdtd.export_all")


def populate_types(scene):
    bpy.app.handlers.depsgraph_update_pre.remove(populate_types)
    scene.block_types.clear()
    for k, block in block_classes.items():
        scene.block_types.add().name = k
        if hasattr(block, 'populate_types'):
            block.populate_types(scene)


class FieldMatPanel(bpy.types.Panel):
    """Creates a FDTD Panel in the Material properties window"""

    bl_label = "FDTD"
    bl_idname = "MATERIAL_PT_FDTD"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    mur: bp.FloatProperty(
        description="Relative Permeability", min=0.0, default=1.0
    )
    epr: bp.FloatProperty(
        description="Relative Permittivity", min=0.0, default=1.0
    )
    sige: bp.FloatProperty(
        min=0.0, default=0.0, description="Eletrical Conductivity [S/m]"
    )

    @classmethod
    def create_types(cls):
        for key, value in cls.__annotations__.items():
            setattr(bpy.types.Material, key, value)

    @classmethod
    def del_types(cls):
        for key in cls.props.keys():
            delattr(bpy.types.Material, key)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        ob = context.object
        if ob.material_slots:
            mat = ob.material_slots[0].material
            if ob.blockType in MatBlock.mtype_codes.keys():
                layout.prop(mat, 'mur', text="Relative Permeability µr")
                layout.prop(mat, 'epr', text="Relative Permittivity εr")
                layout.prop(mat, 'sige', text="Conductivity σE")
            if mat.name in ('FieldH', 'FieldE'):
                field = mat.name[-1]
                units = Probe.fieldUnits[('E', 'M')[field == 'H']]
                Mr = ob.rotation_euler.to_matrix()
                V = Mr @ Vector((1, 0, 0))
                r = ob.scale.x
                ap = ob.parent
                if ap.p_log:
                    r = 10 ** (r * 4 - 6)
                r /= ap.p_magScale
                V.x *= r
                V.y *= r
                V.z *= r
                box = layout.box()
                box.label(text=f"{field} = {gv(V)} {units}")
                box.label(text=f"   = {V.length:g} {units}")


addon_keymaps = []
wm = bpy.context.window_manager

operators_panels = (
    FieldOperator,
    FieldCleanOperator,
    FieldPauseOperator,
    FieldPlotOperator,
    FieldCenterOperator,
    FieldExportOperator,
    FieldExportAllOperator,
    FieldObjectPanel,
    FieldMatPanel,
)


def register():
    for c in operators_panels:
        bpy.utils.register_class(c)

    # assign Cmd-R shortcut to 'Run FDTD', etc.
    km = wm.keyconfigs.addon.keymaps.new(
        name='Object Mode', space_type='EMPTY'
    )
    km.keymap_items.new("fdtd.run", 'R', 'PRESS', oskey=True)
    km.keymap_items.new("fdtd.clean", 'K', 'PRESS', oskey=True)
    km.keymap_items.new("fdtd.plot", 'P', 'PRESS', oskey=True)
    km.keymap_items.new("fdtd.pause", 'P', 'PRESS')
    addon_keymaps.append(km)

    bpy.types.Scene.block_types = bp.CollectionProperty(
        type=bpy.types.PropertyGroup
    )

    for k, block in block_classes.items():
        if hasattr(block, 'register_types'):
            block.register_types()
    bpy.types.Object.blockType = bp.StringProperty()
    bpy.app.handlers.depsgraph_update_pre.append(populate_types)

    for cls in block_classes.values():
        cls.create_types()
    FieldMatPanel.create_types()


def unregister():
    print("unregister:")
    for c in classes:
        bpy.utils.unregister_class(c)
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    addon_keymaps.clear()
    del bpy.types.Scene.block_types
    del bpy.types.Object.blockType
    for cls in block_classes.values():
        cls.del_types()
    for k, block in block_classes.items():
        if hasattr(block, 'unregister_types'):
            block.unregister_types()
    FieldMatPanel.del_types()
    print("unregister done.")


if __name__ == "__main__":
    register()
