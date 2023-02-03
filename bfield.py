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
##import matplotlib
# backend, chosen from /Applications/blender-2.79.app/Contents/Resources/
#      2.79/python/lib/python3.5/site-packages/matplotlib/backends
# Unfortunately that backend isn't in Blender
##matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
cwd = bpy.path.abspath("//")  # TODO: fix abspath("//") for import siunits
sys.path.append(cwd)
print(f"{cwd=}")
if 'siunits' in sys.modules:
    print("reloading module siunits")
    del sys.modules['siunits']
import siunits as si

isLinux = (os.uname()[0] == 'Linux')

timeout = 2500 # sec, server communication

c0 = 2.998e8    # m/s speed of light
z0 = 376.73     # ohms free space impedance
mm = 0.001      # m/mm

timeUnits = {'sec': 1., 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12,
             'fs': 1e-15}

#layerMain   = 0
#layerSnap   = 10
#layerH      = 11
#layerE      = 12

# Name of layer collection, of which there must be exactly one of
# each visible. Will be created if needed.

class StandardCollection(object):

    def __init__(self, baseName):
        self.baseName = baseName
        self._name = None

    def name(self):
        scene = bpy.context.scene
        collNames = [cn for cn in scene.collection.children.keys()
                     if cn.startswith(self.baseName)]
        if self._name is None or not self._name in collNames:
            if len(collNames) == 1:
                self._name = collNames[0]
            elif len(collNames) == 0:
                c = bpy.data.collections.new(self.baseName)
                scene.collection.children.link(c)
                self._name = c.name
            else:
                raise RuntimeError(
                    f"Expected a single {self.baseName} collection in scene."
                    f" Found {collNames}")
        ##print(f"StandardCollection {self.baseName}: {self._name}")
        return self._name

    def get(self):
        return bpy.data.collections[self.name()]

collMain = StandardCollection("Main")
collSnap = StandardCollection("Snap")
collPlane = StandardCollection("Plane")
collE = StandardCollection("E")
collH = StandardCollection("H")


sims = {}       # dictionary of FDTD simulations, indexed by scene
fieldOperator = None

def tu(ob, name):
    return getattr(ob, name) * timeUnits[getattr(ob, name + 'Units')]

# 3D mouse position display. From users lemon, batFINGER at stackexchange.

def getMouse3D(event):
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
        view_vector = view3d_utils.region_2d_to_vector_3d(region,
                                    region3D, mouse_pos)
        # the 3D location in this direction
        loc = view3d_utils.region_2d_to_location_3d(region,
                                    region3D, mouse_pos, view_vector)
        # the 3D location converted in object local coordinates
        ##loc = object.matrix_world.inverted() * loc
    else:
        print(f"getMouse3D: {space_data=} {object=} {region=} {mouse_pos=}")
    return loc


#==============================================================================

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

# Get an object's bounds box dimensions in global coordinates.

def bounds(ob):
    bb = ob.bound_box
    Bs = ob.matrix_world @ Vector(bb[0])
    Be = ob.matrix_world @ Vector(bb[6])
    return Bs, Be

# Return a (Bs, Be) bound box trimmed to both objects, or None.

def overlap(ob1, ob2, extend=0):
    Ex = Vector((extend, extend, extend))
    B1l, B1h = bounds(ob1)
    B2l, B2h = bounds(ob2)
    ##print("overlap():", fv(B1l), fv(B1h))
    ##print("          ", fv(B2l), fv(B2h))
    B2l -= Ex
    B2h += Ex
    if not ((B1h.x >= B2l.x) and (B2h.x >= B1l.x) and
            (B1h.y >= B2l.y) and (B2h.y >= B1l.y) and
            (B1h.z >= B2l.z) and (B2h.z >= B1l.z)):
        return None
    B1l.x = max(B1l.x, B2l.x)
    B1l.y = max(B1l.y, B2l.y)
    B1l.z = max(B1l.z, B2l.z)
    B1h.x = min(B1h.x, B2h.x)
    B1h.y = min(B1h.y, B2h.y)
    B1h.z = min(B1h.z, B2h.z)
    return (B1l, B1h)

#------------------------------------------------------------------------------

def xcodeRun():
    print("Telling XCode to run")
    ##scpt = 'tell application "XCode" to run active workspace document'
    scpt = """
        tell application "XCode"
            run workspace document "bfield.xcodeproj"
        end tell
        """
    p = Popen(['osascript', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE,
              universal_newlines=True)
    stdout, stderr = p.communicate(scpt)
    print(p.returncode, stdout, stderr)

#------------------------------------------------------------------------------

def getFieldsOb(context, create=True):
    name = "0Fields"
    obs = [ob for ob in context.visible_objects
              if ob.name.startswith(name)]
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
        sz = 16.
        mesh.from_pydata(((0,0,0), (sz,sz,sz)), [], [])
        mesh.update()
        ob.show_texture_space = True

        # put this object in the Main collection
        ##ob.layers = [i == layerMain for i in range(20)]
        print(f"{name} to Main collection {collMain.name()}")
        collMain.get().objects.link(ob)

        ob.select_set(True)
    else:
        raise RuntimeError('Expected a single Field object in scene')
    return ob


#==============================================================================

class Block:
    props = {
        "snap": bp.BoolProperty(description="Snap bounds to sim grid",
                                default=False),
        "snappedName": bp.StringProperty(description=
                        "Name of snapped-to-sim-grid copy of this object"),
    }

    # cls.__annotations__ isn't used here because it's empty for subclasses

    @classmethod
    def createTypes(cls):
        ##print(f"Block.createTypes for {cls.__name__}: "
        ##      f"adding {', '.join(cls.props.keys())}")
        for key, value in cls.props.items():
            setattr(bpy.types.Object, key, value)

    @classmethod
    def delTypes(cls):
        for key in cls.props.keys():
            delattr(bpy.types.Object, key)

    def getFieldMat(self):
        ob = self.ob
        slots = ob.material_slots
        mat = slots[0].material if slots else None
        if not (mat and mat in self.sim.fmats):
            raise ValueError(f"object {ob.name} requires an FDTD material")
        return mat

    #--------------------------------------------------------------------------

    def __init__(self, ob, sim):
        self.sim = sim  # simulation

        scn = bpy.context.scene
        view_layer = bpy.context.view_layer
        dx = sim.dx
        op = bpy.ops.object

        # object may optionally be snapped to grid
        sob = ob
        snapc = collSnap.get()
        if ob.verbose > 0:
            print("Block.init: checking", ob.name, "snap=", ob.snap)
        if ob.snap:
            name = ob.snappedName
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
                ob.snappedName = sob.name = ob.name + "_snap"
                sob.display_type = 'WIRE'
                # sob.layers = [l==layerSnap for l in range(20)]
                collMain.get().objects.unlink(sob)
                snapc.objects.link(sob)
                Iss = IGrid(Bs, dx)
                Ise = IGrid(Be, dx)
                ##print(uob.name, "Iss, Ise=", Iss, Ise)
                Gss = Vector((Iss.i*dx, Iss.j*dx, Iss.k*dx))
                Gse = Vector((Ise.i*dx, Ise.j*dx, Ise.k*dx))
                ##print(sob.name, "Gss, Gse=", fv(Gss), fv(Gse))
                Ss = Gse - Gss
                Ls = Vector((Gss.x+Ss.x/2, Gss.y+Ss.y/2, Gss.z+Ss.z/2))
                smin = dx / 10.
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
                    sob.scale = Vector((Cu.x*Ss.x/Su.x, Cu.y*Ss.y/Su.y,
                                        Cu.z*Ss.z/Su.z))
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
                sob.snappedName = ""

        self.ob = sob
        self.Bs, self.Be = bounds(sob)
        ##print(f"{ob.name} bounds: {fv(self.Bs)} - {fv(self.Be)}")

    #--------------------------------------------------------------------------

    def prepare_G(self):
        yield

    def sendDef_G(self):
        yield


#==============================================================================
# A simple volume of material in the sim world.

class MatBlock(Block):
    # used to select a MATCYLINDER axis or identify Block as a MatBlock
    mtypeCodes = { 'MATCUBE': 'C', 'FIELDS': 'C', 'MATMESH': 'C',
                   'RESISTOR': 'C', 'CAPACITOR': 'C',
                   'MATCYLINDERX': 'X' , 'MATCYLINDERY': 'Y' ,
                   'MATCYLINDERZ': 'Z' }

    @classmethod
    def drawProps(self, ob, layout, scene):
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")

    #--------------------------------------------------------------------------

    def prepare_G(self):
        for _ in super().prepare_G():
            yield

    #--------------------------------------------------------------------------

    def sendDef_G(self):
        ob = self.ob
        if ob.verbose > 0:
            print("MatBlock.sendDef_G")
        Bs, Be = self.Bs, self.Be
        mat = self.getFieldMat()
        mtype = self.mtypeCodes[ob['blockType']]
        self.sim.send(f"B{mtype} {ob.name} {mat.name} {Bs.x:g} {Be.x:g} "
                      f"{Bs.y:g} {Be.y:g} {Bs.z:g} {Be.z:g}\n")
        yield


#==============================================================================
# A volume of material in the sim world.

class MeshMatBlock(MatBlock):

    def prepare_G(self):
        for _ in super().prepare_G():
            yield

    #--------------------------------------------------------------------------
    # Write an IMBlock's mesh as a stack of image files for the server. Only
    # alpha layer is used.

    def sendDef_G(self):
        ob = self.ob
        sim = self.sim
        scn = bpy.context.scene
        dx = sim.dx
        if ob.verbose > 0:
            print("MeshMatBlock.sendDef_G", ob.name, "start")
        
        # need the snap layer active for Dynamic Paint
        # snapLayerSave = scn.layers[layerSnap]
        # scn.layers[layerSnap] = 1
        snapc = collSnap.get()
        snapHideSave = snapc.hide_viewport
        snapc.hide_viewport = False

        # make plane that covers object, and square for a DP canvas
        Bs, Be = self.Bs, self.Be
        x, y = (Bs.x+Be.x)/2, (Bs.y+Be.y)/2
        D = ob.dimensions
        r = (D.x, D.y)[D.y > D.x] / 2
        ifactor = 4
        nx = max(sim.grid(2*r) * ifactor, 1)
        nz = max(sim.grid(D.z) * ifactor, 1)
        mat = self.getFieldMat()

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
                                print("deleting", o.name,
                                      "Dynamic Paint mod")
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
            bpy.ops.mesh.primitive_plane_add()
            so = bpy.context.object
            print(f"canvas plane for {ob.name} is {so.name}, "
                  f"verbose={ob.verbose}")
            snapc.objects.link(so)
            so.scale = (r, r, 1)
            so.hide_viewport = False
            collPlane.get().objects.link(so)
            scn.collection.objects.unlink(so)
            # why?
            ##bpy.ops.object.transform_apply(scale=True)

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
            lastfn = csurf.image_output_path + (f"/paintmap{nz:04}.png")
            if ob.verbose > 1:
                print("deleting any", lastfn)
            p = bpy.path.abspath(lastfn)
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
                    print("waiting for", lastfn)
                yield
            ##so.hide_viewport = True
            bpy.data.objects.remove(so, do_unlink=True)

        # tell server to load image files
        bpy.context.view_layer.objects.active = ob
        nx = max(nx, 16)
        sim.send(f"HI {ob.name} {mat.name} {x-r:g} {x+r:g} {y-r:g} {y+r:g} "
                 f"{Bs.z:g} {Be.z:g} {nx} {nz}\n")
        if ob.verbose > 0:
            print("MeshMatBlock.sendDef_G", ob.name, "done.")
        snapc.hide_viewport = snapHideSave


#==============================================================================
# An image-defined layer material in the sim world.

class LayerMatBlock(Block):
    props = {
        "fmatName": bp.StringProperty(description="FDTD material"),
        "snap": bp.BoolProperty(description="Snap bounds to sim grid",
                        default=False),
    }

    @classmethod
    def drawProps(self, ob, layout, scene):
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")

        layout.prop_search(ob, 'fmatName', bpy.data, 'materials',
                        text="FDTD material")

    #--------------------------------------------------------------------------

    def prepare_G(self):
        for _ in super().prepare_G():
            yield

    #--------------------------------------------------------------------------

    def sendDef_G(self):
        ob = self.ob
        if ob.verbose > 0:
            print("LayerMatBlock.sendDef_G", ob.name, "start")
        mesh = ob.data
        mat = mesh.materials[0]
        tex = mat.node_tree.nodes.get('Image Texture')
        if not tex:
            raise ValueError(f"object {ob.name} missing Image Texture node")
        img = tex.image
        imgFilePath = bpy.path.abspath(img.filepath)
        Bs, Be = self.Bs, self.Be
        fmatName = ob.get('fmatName')
        self.sim.send(f"LI {ob.name} {fmatName} {Bs.x:g} {Be.x:g} "
                      f"{Bs.y:g} {Be.y:g} {Bs.z:g} {Be.z:g} {imgFilePath}\n")
        yield


#==============================================================================

class FieldsBlock(MatBlock):
    props = {
        "dx": bp.FloatProperty(description="Grid cell spacing, mm",
                    min=0., default=1.),
        "usPoll": bp.IntProperty(description="us/Step",
                    min=0, default=50),
        "msRate": bp.IntProperty(description="ms/Update",
                    min=0, default=500),
        "rec": bp.BoolProperty(description="Record as animation",
                    default=True),
        "pmlBorder": bp.IntProperty(description=
                    "PML border width, cells", min=0, default=4),
        "verbose": bp.IntProperty(description=
                    "Sim verbosity level, 0-3", min=0, default=0),
    }

    @classmethod
    def drawProps(self, ob, layout, scene):
        spit = layout.split()
        col = spit.column(align=True)
        col.label(text="Grid size (dx):")
        col.prop(ob, "dx", text="mm")
        dx = ob.get('dx')
        D = ob.dimensions
        if not dx is None:
            col.label(text="nx, ny, nz=")
            col.label(text=f"{ma.floor(D.x/dx)}, {ma.floor(D.y/dx)}, "
                           f"{ma.floor(D.z/dx)}")
        col = spit.column()
        col.prop(ob, 'usPoll', text="µs/Step")
        col.prop(ob, 'msRate', text="ms/Up")
        col.prop(ob, 'pmlBorder', text="PML cells")
        col.prop(ob, 'verbose', text="Verbosity")
        col.prop(ob, 'rec', text="Record")
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")


    def sendSimDef_G(self):
        ob = self.ob
        if ob.verbose > 0:
            print("FieldsBlock.sendSimDef_G")
        yield
        Bs, Be = self.Bs, self.Be
        mat = self.getFieldMat()
        sim = self.sim
        sim.send("A units mm\n")
        sim.send(f"A usPoll {sim.usPoll}\n")
        sim.send(f"A verbose {sim.verbose}\n")
        cmd = (f"F {ob.name} {mat.name} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
               f"{Bs.z:g} {Be.z:g} {sim.dx:g} 1 {sim.pmlBorder}")
        if sim.verbose > 1:
            print(cmd)
        self.sim.send(cmd)


#==============================================================================
# A block of resistive material. Creates the material if needed.

class Resistor(MatBlock):
    props = {
             "resistance": bp.FloatProperty(description="Resistance, ohms",
                                            min=0., default=1.),
             "axis": bp.StringProperty(description="axis of resistance (X/Y/Z)",
                                       default='X'),
    }

    @classmethod
    def drawProps(self, ob, layout, scene):
        layout.prop(ob, 'resistance', text="Resistance (ohms)")
        layout.prop_search(ob, 'axis', scene, 's_axes', text="Axis")
        layout.prop(ob, 'verbose', text="Verbosity")

    def prepare_G(self):
        ob = self.ob
        print("Resistor.prepare_G", ob.name, ob.axis)
        for _ in super().prepare_G():
            yield
        R = ob.resistance
        sige = 1. / R
        N = self.Be - self.Bs
        axis = ob.axis[-1]
        if axis == 'X':
            sige = sige * N.x / (N.y * N.z)
        elif axis == 'Y':
            sige = sige * N.y / (N.x * N.z)
        elif axis == 'Z':
            sige = sige * N.z / (N.x * N.y)
            ##print("sige=", N.z, "/ (", N.x, N.y, R, ") /", mm)
        sige = sige / mm

        m = ob.active_material
        if m is None:
            m = bpy.data.materials.new(name=f"Carbon-{ob.name}")
        m.mur = 1.0
        m.epr = 1.0
        m.sige = sige
        print("material:", m.name)

    def sendDef_G(self):
        ob = self.ob
        if ob.verbose > 0:
            print("Resistor.sendDef_G", ob.name, "start")
        for _ in super().sendDef_G():
            yield


#==============================================================================
# A capacitor formed from 2 plates and block of dielectric. Creates the
# material if needed.

class Capacitor(MatBlock):
    props = {
             "capacitance": bp.FloatProperty(description="Capacitance, farads",
                                             min=0., default=1.),
             "axis": bp.StringProperty(description="axis of capacitor (X/Y/Z)",
                                       default='X'),
    }

    @classmethod
    def drawProps(self, ob, layout, scene):
        layout.prop(ob, 'capacitance', text="Capacitance (farads)")
        layout.prop_search(ob, 'axis', scene, 's_axes', text="Axis")
        layout.prop(ob, 'verbose', text="Verbosity")

    def prepare_G(self):
        ob = self.ob
        print("Capacitor.prepare_G", ob.name, ob.axis)
        for _ in super().prepare_G():
            yield
        C = ob.capacitance
        N = self.Be - self.Bs
        axis = ob.axis[-1]
        if axis == 'X':
            sige = sige * N.x / (N.y * N.z)
        elif axis == 'Y':
            sige = sige * N.y / (N.x * N.z)
        elif axis == 'Z':
            sige = sige * N.z / (N.x * N.y)
            ##print("sige=", N.z, "/ (", N.x, N.y, R, ") /", mm)
        sige = sige / mm

        m = ob.active_material
        if m is None:
            m = bpy.data.materials.new(name=f"Carbon-{ob.name}")
        m.mur = 1.0
        m.epr = 1.0
        m.sige = sige
        print("material:", m.name)

    def sendDef_G(self):
        ob = self.ob
        if ob.verbose > 0:
            print(f"Capacitor.sendDef_G{ob.name} start")
        for _ in super().sendDef_G():
            yield


#==============================================================================

class SubSpaceBlock(MatBlock):

    @classmethod
    def drawProps(self, ob, layout, scene):
        ##col.prop(ob, "dx", text="mm")         # assumed dx = parent.dx/2
        pass

    def sendSimDef_G(self):
        ob = self.ob
        if ob.verbose > 0:
            print("SubSpaceBlock.sendSimDef_G")
        yield
        Bs, Be = self.Bs, self.Be
        mat = self.getFieldMat()
        cmd = (f"G {ob.name} {mat.name} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
               f"{Bs.z:g} {Be.z:g} {ob.parent.name}\n")
        if sim.verbose > 1:
            print(cmd)
        self.sim.send(cmd)


#==============================================================================

class Source(Block):
    props = {
        "s_axis": bp.StringProperty(description=
                    "Axis of positive voltage"),
        "s_excitation": bp.StringProperty(description="Excitation"),
        "s_function": bp.StringProperty(description="Function"),
        "s_hard": bp.BoolProperty(description=
                    "Hard source", default=False),
        "s_resistance": bp.FloatProperty(description=
                    "Soft source resistance (ohms)", min=0., default=50.),
        "s_scale": bp.FloatProperty(description=
                    "Signal height", min=0., default=1.),
        "s_tstart": bp.FloatProperty(description=
                    "Pulse start time", min=0., default=0.),
        "s_tstartUnits": bp.StringProperty(description=
                    "Pulse start time units", default='ps'),
        "s_trise": bp.FloatProperty(description=
                    "Pulse rise time", min=0., default=10.),
        "s_triseUnits": bp.StringProperty(description=
                    "Pulse rise time units", default='ps'),
        "s_duration": bp.FloatProperty(description=
                    "Pulse duration (after trise, before tfall)",
                    min=0., default=0.),
        "s_durationUnits": bp.StringProperty(description=
                    "Pulse duration units", default='sec'),
        "s_tfall": bp.FloatProperty(description=
                    "Pulse fall time", min=0., default=10.),
        "s_tfallUnits": bp.StringProperty(description=
                    "Pulse fall time units", default='ps'),
    }

    @classmethod
    def registerTypes(self):
        bpy.types.Scene.s_excitations = bp.CollectionProperty(
                                            type=bpy.types.PropertyGroup)
        bpy.types.Scene.s_axes = bp.CollectionProperty(
                                            type=bpy.types.PropertyGroup)
        bpy.types.Scene.s_functions = bp.CollectionProperty(
                                            type=bpy.types.PropertyGroup)
        bpy.types.Scene.timeUnits = bp.CollectionProperty(
                                            type=bpy.types.PropertyGroup)

    @classmethod
    def unregisterTypes(self):
        del bpy.types.Scene.s_excitations
        del bpy.types.Scene.s_axes
        del bpy.types.Scene.s_functions
        del bpy.types.Scene.timeUnits

    @classmethod
    def populateTypes(self, scene):
        scene.s_excitations.clear()
        for k in ('Voltage', 'Current', 'Electrical', 'Magnetic'):
            scene.s_excitations.add().name = k
        scene.s_axes.clear()
        for k in (' X', ' Y', ' Z', '-X', '-Y', '-Z'):
            scene.s_axes.add().name = k
        scene.s_functions.clear()
        for k in ('Gaussian Pulse', 'Sine', 'Constant'):
            scene.s_functions.add().name = k
        scene.timeUnits.clear()
        for k in timeUnits.keys():
            scene.timeUnits.add().name = k

    @classmethod
    def drawProps(self, ob, layout, scene):
        layout.prop_search(ob, 's_excitation', scene, 's_excitations',
                           text="Excitation")
        layout.prop_search(ob, 's_function', scene, 's_functions',
                           text="Function")
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
            split.prop_search(ob, 's_tstartUnits', scene, 'timeUnits', text="")
        if ob.s_function == 'Gaussian Pulse':
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_trise', text="Rise time")
            split.prop_search(ob, 's_triseUnits', scene, 'timeUnits', text="")
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_duration', text="Duration")
            split.prop_search(ob, 's_durationUnits', scene, 'timeUnits',
                              text="")
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_tfall', text="Fall time")
            split.prop_search(ob, 's_tfallUnits', scene, 'timeUnits', text="")
        elif ob.s_function == 'Sine':
            split = layout.split(factor=0.7)
            split.row().prop(ob, 's_duration', text="Period")
            split.prop_search(ob, 's_durationUnits', scene, 'timeUnits',
                              text="")
        layout.prop(ob, 'snap', text="Snap bounds to sim grid")
        layout.prop(ob, 'verbose', text="Verbosity")

    def sendDef_G(self):
        ob = self.ob
        if ob.verbose > 0:
            print("Source.sendDef_G")
        yield
        Bs, Be = self.Bs, self.Be
        scale = ob.s_scale
        axis = ord(ob.s_axis[-1]) - ord('X')
        dist = (Be.x-Bs.x, Be.y-Bs.y, Be.z-Bs.z)[axis] * mm
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
                scale *= 2 # assume load matches Rs, dividing Vs by two

        cmd = (f"S {ob.name} {ex} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
               f"{Bs.z:g} {Be.z:g} {ob.s_function.replace(' ', '_')} "
               f"{ob.s_hard:d} {ob.s_resistance:g} {axis} {scale:g} "
               f"{tu(ob,'s_tstart'):g} {tu(ob,'s_trise'):g} "
               f"{tu(ob,'s_duration'):g} {tu(ob,'s_tfall'):g}")
        if ob.verbose > 1:
            print(cmd)
        self.sim.send(cmd)

#==============================================================================
# A matplotlib figure window.

class Figure:
    winx = None
    winy = None
    num = 1

    def __init__(self):
        # create a matplotlib plot
        print("creating Figure", Figure.num)
        self.figure = plt.figure(Figure.num, figsize=(4.7, 4))
        self.figure.clear()
        ##win = plt.get_current_fig_manager().window
        ##m = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)", win.geometry())
        ##dx, dy, x, y = [x for x in m.groups()]
        ##if not Figure.winx is None:
        ##    x = Figure.winx + 10
        ##    y = Figure.winy + 10
        ##    win.geometry(f"{dx}x{dy}+{x}+{y}")
        ##Figure.winx = x
        ##Figure.winy = y
        Figure.num += 1
        self.ax = plt.axes()
        self.max_x = 0.
        self.min_y = 9999999.
        self.max_y = -9999999.
        ##self.units_y = 'V'
        self.ylabel = ""

#==============================================================================

class NFmtr(ticker.Formatter):
    def __init__(self, scale_x):
        self.scale_x = scale_x

    def __call__(self, x, pos=None):
        return "{0:g}".format(x/10**self.scale_x)

#==============================================================================

class Probe(Block):
    props = {
        "p_field": bp.StringProperty(description=
                    "Field to measure", default='Electric'),
        "p_axis": bp.StringProperty(description=
                    "Measurement axis", default='XYZ'),
        "p_axisSign": bp.IntProperty(description=
                    "Measurement axis sign", default=1),
        "p_verbose": bp.IntProperty(description=
                    "Probe verbosity level, 0-3", min=0, default=0),
        "p_shape": bp.StringProperty(description=
                    "Display shape", default='Plane'),
        "p_value": bp.FloatProperty(precision=6, description=
                    "Probe measured value"),
        "p_value3": bp.FloatVectorProperty(description=
                    "Probe measured vector value"),
        ##"p_inColor": bp.BoolProperty(description=
        ##            "Display is in color", default=True),
        "p_dispScale": bp.FloatProperty(min=0, default=256., description=
                    "Display scale"),
        "p_pixelRep": bp.IntProperty(min=1, default=1, description=
                    "Image pixel repeat factor"),
        "p_imageAlpha": bp.FloatProperty(description=
                    "Image transparency alpha", min=0., default=1.),
        "p_sfactor": bp.IntProperty(description=
                    "Volume space factor, cells/sample", min=1, default=1),
        "p_log": bp.BoolProperty(description=
                    "Log scale for magnitude", default=True),
        "p_magScale": bp.FloatProperty(description=
                    "Magnitude multiplier", min=0., default=1.),
        "p_sum": bp.BoolProperty(description=
                    "Sum values", default=False),
        "p_avg": bp.BoolProperty(description=
                    "Average values", default=False),
        "p_dispIsMesh": bp.BoolProperty(description=
                    "Use mesh object for in-world chart", default=False),
        "p_dispIsPlot": bp.BoolProperty(description=
                    "Use external MatPlotLib for chart", default=True),
        "p_plotScale": bp.FloatProperty(description=
                    "chart scale multiplier", min=0., default=1.),
        "p_dispColor": bp.FloatVectorProperty(description=
                    "Color", subtype='COLOR',
                    size=4, min=0., max=1., default=(0.75, 0., 0.8, 1.)),
        "p_dispPos": bp.FloatProperty(min=0., max=1., description=
                    "Relative position in chart", default=0.5),
    }
    fieldNames = {'Electric': 'E', 'Magnetic': 'H', 'Voltage': 'E',
                     'Current Density': 'J', 'mE2': 'M', 'mH2': 'N'}
    fieldNamesMag = {'Electric': 'E', 'Magnetic': 'H', 'Voltage': 'E',
                     'mE2': 'M', 'mH2': 'N', 'SigE': 'S',
                     'Current Density': 'J'}
    fieldUnits = {'V': "V", 'E': "V/m", 'M': "A/m", 'S': "50MS", 'T': "50MS",
                  'C': "A/m^2"}

    @classmethod
    def registerTypes(self):
        ts = bpy.types.Scene
        ts.p_fields = bp.CollectionProperty(type=bpy.types.PropertyGroup)
        ts.p_fieldsMag = bp.CollectionProperty(type=bpy.types.PropertyGroup)
        ts.p_axes   = bp.CollectionProperty(type=bpy.types.PropertyGroup)
        ts.p_shapes = bp.CollectionProperty(type=bpy.types.PropertyGroup)

    @classmethod
    def unregisterTypes(self):
        del bpy.types.Scene.p_fields
        del bpy.types.Scene.p_fieldsMag
        del bpy.types.Scene.p_axes
        del bpy.types.Scene.p_shapes

    @classmethod
    def populateTypes(self, scene):
        scene.p_fields.clear()
        for k in self.fieldNames.keys():
            scene.p_fields.add().name = k
        scene.p_fieldsMag.clear()
        for k in self.fieldNamesMag.keys():
            scene.p_fieldsMag.add().name = k
        scene.p_axes.clear()
        for k in ('X', 'Y', 'Z', '-X', '-Y', '-Z', 'XYZ', 'Magnitude'):
            scene.p_axes.add().name = k
        scene.p_shapes.clear()
        for k in ('Point', 'Line', 'Plane', 'Volume'):
            scene.p_shapes.add().name = k

    @classmethod
    def drawProps(self, ob, layout, scene):
        fields = ('p_fields', 'p_fieldsMag')[ob.p_axis == 'Magnitude']
        layout.prop_search(ob, 'p_shape', scene, 'p_shapes',
                           text="Display Shape")

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
            ##row.prop(ob, 'p_inColor', text="In color")
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

    #--------------------------------------------------------------------------

    def setPlaneTexture(self, data):
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
        magScale, rep = ob.p_dispScale, ob.p_pixelRep
        nix, niy = self.nix, self.niy
        di = 0
        if img.generated_width == rep*nix and img.generated_height == rep*niy:
            if ob.p_axis == 'XYZ':
                # RGBA as xyz vector
                adata = np.frombuffer(data, np.uint32).reshape((nix,niy)).T
                apix = np.frombuffer(adata.reshape((-1,1)), np.uint8) / 256.
            else:
                # greyscale
                adata = np.frombuffer(data, np.float32).reshape((nix,niy)).T
                adata = (np.abs(adata) * magScale).clip(0, 1.)
                apix = np.zeros((nix*niy, 4)) + 1.
                apix[:,0:3] = adata.reshape((-1,1))
            if rep > 1:
                apix = apix.reshape((niy,nix,4))
                ##print(f"setPlaneTex: {rep=}, {nix=}, {niy=}")
                apix = apix.repeat(rep, axis=0).repeat(rep, axis=1)
            ##if ob.p_verbose > 1:
            ##    print("apix=", apix.shape, apix.dtype)
            ##    print(apix)
            img.pixels = apix.reshape(-1)
        else:
            print(f"probe.P: image size mismatch: {img.name} "
                  f"should be {nix}x{niy}")

    #--------------------------------------------------------------------------
    # Timeline frame changed: update probe from history.

    def probePlaneFrameHandler(self, scene, depsgraph):
        data = self.history.get(scene.frame_current)
        if not data is None:
            self.setPlaneTexture(data)

    def probeValueFrameHandler(self, scene, depsgraph):
        data = self.history.get(scene.frame_current)
        ##print("probeValueFrHand:", self.ob.name, scn.frame_current, data)
        if not data is None:
            if type(data) == Vector:
                self.ob.p_value3 = data
            else:
                self.ob.p_value = data
        pass # (fixes func pulldown indentation)

    #--------------------------------------------------------------------------
    # Create or update a probe's display image, objects, etc.

    def prepare_G(self):
        ob = self.ob
        if ob.p_verbose > 0:
            print("Probe.prepare_G start", self.ob.name)
        yield
        sim = self.sim
        scn = bpy.context.scene
        objs = bpy.data.objects
        posth = bpy.app.handlers.frame_change_post
        bmats = bpy.data.materials
        fieldName = self.fieldNamesMag[ob.p_field]
        sfactor = ob.p_sfactor
        dx = sim.dx
        self.lastStep = -1
        if ob.p_verbose > 1:
            print(f"Probe.prepare_G for {ob.name}: {dx=} {sfactor=}")

        # get untrimmed probe dimensions
        B0l, B0u = self.Bs, self.Be
        ##print(f"{ob.name} bounds: {fv(B0l)}, {fv(B0u)}")

        # determine trimmed probe grid coords and size
        fields = sim.fieldsBlock
        fob = fields.ob
        Bsf, Bef = bounds(fob)
        fover = overlap(ob, fob, fob.pmlBorder * dx)
        if fover is None:
            raise ValueError("Bug: probe doesn't overlap Fields!")
        B1l, B1u = fover
        Is = IGrid(B1l-Bsf, dx)
        ##print("Probe: ob=", ob.name, "ob.p_verbose=", ob.p_verbose)
        if ob.p_verbose > 1:
            print("B1l.x=", B1l.x, "Bsf.x=", Bsf.x, "dx=", dx)
        Ie = IGrid(B1u-Bsf, dx)
        nx = max(Ie.i - Is.i, 1)
        ny = max(Ie.j - Is.j, 1)
        ##print("ny:", Ie.j, Is.j, ny)
        nz = max(Ie.k - Is.k, 1)
        if ob.p_verbose > 1:
            print(ob.name, "Bsf=", fv(Bsf), "Is=", Is, "Ie=", Ie)
            print(" nx,ny,nz=", nx, ny, nz)
        self.N = IVector(nx,ny,nz)

        shape = ob.p_shape
        if shape == 'Point':
            if ob.type == 'EMPTY':
                n = 1
                if ob.p_axis == 'XYZ':
                    n = 3
                if ob.p_verbose > 0:
                    print(ob.name, "single point measurement,", n, "values")
                self.n = n
                if self.probeValueFrameHandler in posth:
                    del posth[self.probeValueFrameHandler]
                posth.append(self.probeValueFrameHandler)
            else:
                self.n = nx*ny*nz
                if ob.p_verbose > 0:
                    print(ob.name, "extended (sum) point measurement, n=",
                          self.n)

        elif shape == 'Line':
            self.n = nx*ny*nz

        elif shape == 'Plane':
            nix, niy = nx, ny
            if nx == 1:
                nix = ny
                niy = nz
            elif ny == 1:
                niy = nz
            n = nix * niy
            if ob.p_verbose > 1:
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
                    print(f"prepare_G found mat {repr(mat)}")
                    tex = mat.node_tree.nodes.get('Image Texture')
                    if tex:
                        img = tex.image
                        if (img is None or
                            img.generated_width != rep*nix or
                            img.generated_height != rep*niy):
                            if ob.p_verbose > 1:
                                print(f"removing old wrong-sized "
                                      f"{img.generated_width}x"
                                      f"{img.generated_height} image")
                            mesh.materials.clear()

            if len(mesh.materials) < 1 or mesh.materials[0] is None:
                # create a new material with an image texture
                if ob.p_verbose > 1:
                    print(f"Probe: creating {nix}x{niy} image plane")
                mesh.materials.clear()
                mat = bpy.data.materials.new(name)
                print(f"prepare_G created mat {repr(mat)}")
                mat.use_nodes = True
                mat.specular_intensity = 0.
                mesh.materials.append(mat)
                imgName = f"probe_{name}"
                img = bpy.data.images.new(imgName, width=rep*nix, height=rep*niy)

                tex = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
                tex.image = img
                # mtex.texture_coords = 'UV'
                # mtex.use_map_color_diffuse = True
                # mtex.mapping = 'FLAT'
                # if len(mesh.uv_layers) == 0:
                #     bpy.ops.mesh.uv_texture_add()

                tex.location = (0, 0)
                # link the texture node to the material
                mat.node_tree.links.new(mat.node_tree.nodes['Principled BSDF']
                                  .inputs['Base Color'], tex.outputs['Color'])

                # the following depends on mesh.polygons[0].vertices having
                # order: 0 1 3 2, and mesh.vertices being like
                # (0,0,0) (0,0,43) (0,60,0) (0,60,43)

            if mat:
                talpha = ob.p_imageAlpha
                ##print(f"{ob.name}: setting image alpha to {talpha}")
                mat.node_tree.nodes["Principled BSDF"
                                    ].inputs['Alpha'].default_value = talpha
                mat.blend_method = 'BLEND'

            # bounds, absolute, untrimmed: B0l to B0u, trimmed: B1l, B1u
            B1d = B1u - B1l
            uvmap = mesh.uv_layers.active
            ud = uvmap.data

            if ob.p_verbose > 1:
                print("nx,ny,nz=", nx,ny,nz)
            if nx == 1: # project onto X-axis view
                ##print("assigning X-axis UV map", uvmap.name)
                Bnewl = Vector(((B0l.y - B1l.y)/B1d.y, (B0l.z - B1l.z)/B1d.z))
                Bnewu = Vector(((B0u.y - B1l.y)/B1d.y, (B0u.z - B1l.z)/B1d.z))
                ##print("uv trim coords:", fv(Bnewl), fv(Bnewu))
                ud[0].uv = Vector((Bnewl.x, Bnewl.y))
                ud[1].uv = Vector((Bnewl.x, Bnewu.y))
                ud[2].uv = Vector((Bnewu.x, Bnewu.y))
                ud[3].uv = Vector((Bnewu.x, Bnewl.y))
            elif ny == 1: # project onto Y-axis view
                ##print("assigning Y-axis UV map", uvmap.name)
                Bnewl = Vector(((B0l.x - B1l.x)/B1d.x, (B0l.z - B1l.z)/B1d.z))
                Bnewu = Vector(((B0u.x - B1l.x)/B1d.x, (B0u.z - B1l.z)/B1d.z))
                ##print("uv trim coords:", fv(Bnewl), fv(Bnewu))
                ud[0].uv = Vector((Bnewl.x, Bnewu.y))
                ud[1].uv = Vector((Bnewu.x, Bnewu.y))
                ud[2].uv = Vector((Bnewu.x, Bnewl.y))
                ud[3].uv = Vector((Bnewl.x, Bnewl.y))
            else:         # project onto Z-axis view
                ##print("assigning Z-axis UV map", uvmap.name)
                Bnewl = Vector(((B0l.x - B1l.x)/B1d.x, (B0l.y - B1l.y)/B1d.y))
                Bnewu = Vector(((B0u.x - B1l.x)/B1d.x, (B0u.y - B1l.y)/B1d.y))
                ##print("uv trim coords:", fv(Bnewl), fv(Bnewu))
                ud[0].uv = Vector((Bnewl.x, Bnewl.y))
                ud[1].uv = Vector((Bnewu.x, Bnewl.y))
                ud[2].uv = Vector((Bnewu.x, Bnewu.y))
                ud[3].uv = Vector((Bnewl.x, Bnewu.y))

            if self.probePlaneFrameHandler in posth:
                del posth[self.probePlaneFrameHandler]
            posth.append(self.probePlaneFrameHandler)

        else: # 'Volume'
            # create H and E field arrow objects if needed
            H = collH.get()
            H.hide_viewport = False
            E = collE.get()
            E.hide_viewport = False
            n = (((nx+sfactor-1)//sfactor) * ((ny+sfactor-1)//sfactor)
                 * ((nz+sfactor-1)//sfactor))
            ne = n * 3
            print(f"probe size {nx}x{ny}x{nz} *3 = {ne} elements")
            if ne < 1 or ne > 80000:
                raise ValueError(f"probe: bad requested data size: {ne}")
            self.n = ne
            if len(ob.children) != n:
                if ob.p_verbose > 1:
                    print(f"Probe: creating {n} {fieldName} arrows")
                dx2, collection = ((0, E), (dx/2, H))[fieldName == 'H']
                # Is is parent lower-left index in full grid, Pp is coords
                Pp = Vector((Is.i*dx, Is.j*dx, Is.k*dx))
                # D2 is 1/2 cell extra for H, + offset of parent lower left
                D2 = Vector((dx2,dx2,dx2)) + Vector(ob.bound_box[0])
                if ob.p_verbose > 1:
                    print(ob.name, "D2=", fv(D2), "Is=", Is)

                # this delete is too dangerous
                ##op = bpy.ops.object
                ##op.select_by_layer(layers=layer+1) # layers 1-20
                ##op.delete()
                # instead we delete just probe's arrow-children
                for arrow in ob.children:
                    objs.remove(arrow, do_unlink=True)

                # get or create the Tmp collection, used to delete all arrows
                tmpc = bpy.data.collections.get('Tmp')
                if not tmpc:
                    tmpc = bpy.data.collections.new('Tmp')
                    if not tmpc:
                        raise RuntimeError(
                            f"failed to create Tmp collection for {ob.name}")
                print(f"probe.vol {ob.name}: {tmpc=}")

                r = dx * 0.05
                h = dx * 0.5
                verts = ((0,r,r), (0,r,-r), (0,-r,-r), (0,-r,r), (h,0,0))
                faces = ((1,0,4), (4,2,1), (4,3,2), (4,0,3), (0,1,2,3))
                # TODO: use common mesh, since Outliner now fixed
                ##mesh = bpy.data.meshes.new(name='Arrow')
                ##mesh.from_pydata(verts, [], faces)
                ##mesh.update()
                mat = bmats[f"Field{fieldName}"]
                for i in range(0, nx, sfactor):
                    for j in range(0, ny, sfactor):
                        for k in range(0, nz, sfactor):
                            # arrow name must be in sortable format
                            name = (f"{fieldName}{Is.i+i:03}"
                                    f"{Is.j+j:03}{Is.k+k:03}")
                            mesh = bpy.data.meshes.new(name)
                            mesh.from_pydata(verts, [], faces)
                            mesh.update()
                            arrow = objs.new(name, mesh)
                            # loc relative to parent
                            arrow.location = (dx*i+D2.x, dx*j+D2.y, dx*k+D2.z)
                            arrow.data.materials.append(mat)
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

        ##print("Probe.prepare_G done.")

    #--------------------------------------------------------------------------

    def sendDef_G(self, update=False):
        yield
        self.sendDef(update)

    def sendDef(self, update=False):
        ##print("Probe.sendDef_G start ", "PU"[update])
        ob = self.ob
        Bs, Be = self.Bs, self.Be
        fieldName = self.fieldNamesMag[ob.p_field]
        ob.p_axisSign = 1
        axis = ob.p_axis
        if axis[0] == '-':
            ob.p_axisSign = -1
            axis = ob.p_axis[1]
        dispType = 'Vec'
        dispScale = ob.p_dispScale
        if ob.p_shape == 'Plane':
            if axis == 'XYZ':
                dispType = 'RGB'
            else:
                dispType = 'Mag'
        elif ob.p_shape == 'Line':
            if axis in 'ZYX':
                dispType = ob.p_axis
                ##self.n = self.n + 1  # voltage sources include edges ???
        elif ob.p_shape == 'Point':
            dispType = 'Mag'
            dispScale = ob.p_plotScale
            if axis == 'XYZ':
                dispType = 'Vec'
            if fieldName == 'V':
                fieldName = 'E'
            if axis in 'ZYX':
                if ob.p_sum:
                    dispType = 'Sum'
                iaxis = ord(axis) - ord('X')
                ##self.dist = (Be.x-Bs.x, Be.y-Bs.y, Be.z-Bs.z)[iaxis]
                N = self.N
                if 0:   # doesn't work with sum, which has to see full block
                    if iaxis == 0:
                        Be.x = Bs.x = (Be.x + Bs.x) / 2.
                        n = N.j * N.k
                    elif iaxis == 1:
                        Be.y = Bs.y = (Be.y + Bs.y) / 2.
                        n = N.i * N.k
                    else:
                        Be.z = Bs.z = (Be.z + Bs.z) / 2.
                        n = N.i * N.j
                    self.n = n

        cmd = (f"{'PU'[update]} {ob.name} {Bs.x:g} {Be.x:g} {Bs.y:g} {Be.y:g} "
               f"{Bs.z:g} {Be.z:g} {fieldName} {dispType[0]} "
               f"{dispScale:g} {ob.p_sfactor} {ob.p_verbose}\n")
        if ob.p_verbose > 1:
            print(cmd)
        self.sim.send(cmd)

    #--------------------------------------------------------------------------
    # Get data values from server for one step.

    def getDataFromServer(self):
        scn = bpy.context.scene
        ob = self.ob
        sim = self.sim

        # send probe request to server
        s = sim.s
        cmd = f"Q {ob.name}"
        if ob.p_verbose:
            print("getDataFromServer:", ob.name, f"cmd='{cmd}'")
        ack = sim.send(cmd, 5)
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
        rem = n*esize
        ##print("n=", n, "esize=", esize, "rem=", rem, dtype)
        rnbytes = s.recv(6)
        if len(rnbytes) != 6:
            raise IOError(f"expected 6 digit data length, got '{rnbytes}'")
        rnbytes = int(rnbytes)
        if ob.p_verbose > 1:
            print(f"Probe: expecting {rem} bytes, receiving {rnbytes}")
        if rnbytes != rem:
            raise IOError(f"probe {ob.name} expected {rem} "
                          f"bytes, got {rnbytes}")
        for i in range(20):
            r = s.recv(n*esize)
            bdata += r
            rem -= len(r)
            if rem <= 0:
                break
        if rem > 0:
            raise IOError

        if step == self.lastStep:
            if ob.p_verbose > 1:
                print("Probe last step", ob.name)
            return None
        self.lastStep = step
        data = np.frombuffer(bdata, dtype=dtype)

        if ob.p_sum:
            iaxis = ord(ob.p_axis[-1]) - ord('X')
            data = data[iaxis:iaxis+1]
            if ob.p_verbose > 1:
                print("data=", np.frombuffer(data, np.float32))

        if ob.p_verbose > 1:
            print(ob.name, "data=", data.shape, data.dtype, "as uint32:")
            np.set_printoptions(formatter={'all':lambda x: f"0x{x:08x}"})
            print(np.frombuffer(data, np.uint32))
            np.set_printoptions(formatter=None)
            print(np.frombuffer(data, np.float32))
        return data

    #--------------------------------------------------------------------------
    # Step probe 1 timestep, getting values from server.

    def doStep(self):
        ob = self.ob
        if ob.p_shape == 'Line' and not (ob.p_sum or ob.p_avg):
            # no history stored for normal line probes
            return

        data = self.getDataFromServer()
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
                fieldName = self.fieldNamesMag[ob.p_field]
                units = ("V/m", "A/m")[ob.p_field == 'Magnetic']
                print(f"{step}: t={t:9.2e}: {ob.name} "
                      f"{fieldName}[{i},{j},{k}]", end="")
                if axis == 'XYZ':
                    print(f"=({V.x:.4g},{V.y:.4g},{V.z:.4g}) {units}")
                else:
                    print(f".{ob.p_axis}={vbase:.4g}{units}")
                ##if ob.p_field == 'Voltage':
                ##    print(f" d={self.dist:5.2}mm v={v:9.4}V")
                ##else:
                ##    print()
            ##if sim.state < 4:
                ### not paused (why only not-paused???)
                ##data = v
                ##print("Point: data=v=", data)
            data = v

        elif ob.p_shape == 'Line':
            # summation, line integral of border
            data_d = np.frombuffer(data[0:6], dtype=np.dtype(np.float64))
            S = data_d[0:3]
            data_i = np.frombuffer(data, dtype=np.dtype(np.int32))
            count = data_i[-1]
            A = S / count
            area = count * dx**2 # m^2
            # sum of H~ is z0*H, in V/m
            H = S / z0   # in A/m
            if ob.p_sum:
                if ob.p_verbose:
                    print(ob.name, "sum=", S, "count=", count, "\navg=", A,
                    "area=", area, "H=", H)
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
                if sim.state < 4:
                    ob.p_value3 = I
                    data = (S, dl, Hx, I)
                    ##self.drawChartStep(I)

        elif ob.p_shape == 'Plane':
            self.setPlaneTexture(data)
            ##print(f"storing history[{sstep}]")

        else: # 'Volume'
            data = data.reshape((n//3, 3))
            if 0: # generate test data
                HEr = []
                for x in range(nx):
                    for y in range(ny):
                        for z in range(nz):
                            HEr.append((x/nx, y/ny, z/nz))
                for x in range(nx):
                    for y in range(ny):
                        for z in range(nz):
                            HEr.append((-x/nx, -y/ny, -z/nz))
                HE = np.array(HEr) * 32768/100.

            logMag, magScale = ob.p_log, ob.p_magScale
            for i,arrow in enumerate(self.arrows):
                arrow.rotation_euler.zero()
                pr = False
                ##pr = arrow.name in ('H040605', 'E040605')
                x,y,z = data[i]
                r2 = x*x + y*y + z*z
                r2 *= magScale**2
                ##arrow.show_name = r2 > 0
                if pr:
                    di = data[i]
                    print(f"{step}: xyz=({di[0]:g},{di[1]:g},{di[2]:g})"
                          f"=({x:6.4}, {y:6.4}, {z:6.4}) r2={r2:6.4}")
                if r2 > 1e-12:
                    r = ma.sqrt(r2)
                    if logMag:
                        ##r = 0.3*(ma.log(r)+4.7)
                        r = 0.25*(ma.log10(r)+6)
                    else:
                        if r > 30:
                            r = 30
                    r *= ob.p_sfactor
                    arrow.scale = (r,r,r)
                    if pr:
                        print(f"{step}: {arrow.name} "
                              f"xyz=({x:6.4}, {y:6.4}, {z:6.4}) "
                              f"r={r:6.4} {magScale}")
                    M = Matrix(((x, 0, 0), (y, 0, 0), (z, 0, 0)))
                    # rely on rotate() to normalize matrix
                    arrow.rotation_euler.rotate(M)
                else:
                    arrow.scale = (0,0,0)
                if sim.rec:
                    arrow.keyframe_insert(data_path='rotation_euler')
                    arrow.keyframe_insert(data_path='scale')
                    ##arrow.keyframe_insert(data_path='show_name')

        # record data history
        if not isinstance(data, (int, float, tuple)):
            data = data.copy()
        ##print("Probe.doStep:", ob.name, "step", step, "data=", data)
        self.history[step] = data

    #--------------------------------------------------------------------------
    # Cmd-P: Plot all point probe value histories.

    @classmethod
    def plotAllSetUp(self):
        self.figs = {}

    def getPlotFig(self, dataShape):
        ob = self.ob
        color = None
        ##figType = ob.p_shape + ob.p_field + ob.p_axis + repr(dataShape)
        figType = ob.p_shape + ob.p_field + repr(dataShape)
        fig = self.figs.get(figType)
        if not fig:
            fig = Figure()
            self.figs[figType] = fig
            if ob.material_slots:
                mat = ob.material_slots[0].material
                color = mat.diffuse_color
            plt.figure(fig.figure.number)
            fig.title = ob.name
        return fig, color

    def plot(self):
        ob = self.ob
        sim = self.sim
        if ob.p_verbose > 0:
            print("Probe.plot", ob.name)
        if (ob.p_shape == 'Point' and self.history) or ob.p_shape == 'Line':
            if ob.p_shape == 'Point':
                # add plot of point probe value history to the plot
                items = list(self.history.items())
                ##print("plot Point", ob.p_axis, type(items[0][1]))
                if type(items[0][1]) == Vector:
                    if ob.p_axis in 'ZYX':
                        ix = ob.p_axis - ord('X')
                        ##print(f"plot Point V[{ix}]")
                        items = [(t, V[ix]) for (t,V) in items]
                    else:
                        items = [(t, V.length) for (t,V) in items]
                h = np.array(items)
                xs, ys = h[h[:,0].argsort()].T
                xs *= sim.dt
                fig, color = self.getPlotFig(1)
                fig.xlabel = "Time"
                fig.xunit = "s"
            else:
                # 'Line'
                self.lastStep = -1 # force data grab
                ys = self.getDataFromServer()
                Bs, Be = self.Bs, self.Be
                N = self.N
                s = Bs.x; e = Be.x
                fig, color = self.getPlotFig(N)
                if N.j > 1:
                    s = Bs.y; e = Be.y
                    fig.xlabel = "Y"
                elif N.k > 1:
                    s = Bs.z; e = Be.z
                    fig.xlabel = "Z"
                else:
                    fig.xlabel = "X"
                fig.xunit = "m"
                xs = np.linspace(s*mm, e*mm, self.n)
                ##print("Line plot: s=", s, "e=", e, "xs=", xs)

            if ob.p_verbose > 1:
                print("ys[-1]=", ys[-1], "xs[0]=", xs[0],
                      "xs[-1]=", xs[-1], "dt=", sim.dt)

            marker = '.' if len(ys) < 50 else None
            label = ob.name
            if ob.p_plotScale != 1:
                label = f"{label} * {ob.p_plotScale:g}"
            plt.plot(xs.copy(), ys.copy(), marker=marker,
                     color=color, label=label)
            fn = ob.p_field
            fig.ylabel = f"{fn.capitalize()} (%s{self.fieldUnits[fn[0]]})"
            fig.max_x = max(xs[-1], fig.max_x)
            fig.min_y = min(ys.min(), fig.min_y)
            fig.max_y = max(ys.max(), fig.max_y)
                
        else:
            print("not plottable")

    @classmethod
    def plotAllFinish(self):
        if not self.figs:
            print("no plottable probes")
            return

        for fig in self.figs.values():
            fnum = fig.figure.number
            print(f"plotting figure {fnum}")
            plt.figure(fnum)
            tm = time.localtime(time.time())
            title = (f"{tm.tm_year-2000:02}{tm.tm_mon:02}{tm.tm_mday:02}"
                     f"-{fnum:02}-{fig.title.replace('.', '-')}")
            fm = plt.get_current_fig_manager()
            fm.set_window_title(title)
            ##plt.title(title)
            # show the plot in a separate window
            plt.ticklabel_format(axis='both', style='sci',
                                 scilimits=(-2,2))
            x_si = si.si(fig.max_x)
            range_y = fig.max_y - fig.min_y
            y_si = si.si(range_y)
            units_x = x_si[-1]
            units_y = y_si[-1]
            scale_x = si.SIPowers.get(units_x)
            scale_y = si.SIPowers.get(units_y)
            if scale_x:
                print(f"Scaling plot X axis by {10**scale_x:g} ({units_x})")
                if 0:
                    ticks_x = ticker.FuncFormatter(lambda x,
                                    pos: '{0:g}'.format(x/10**scale_x))
                    fig.ax.xaxis.set_major_formatter(ticks_x)
                else:
                    fig.ax.xaxis.set_major_formatter(NFmtr(scale_x))
            else:
                units_x = ''
            if scale_y:
                print(f"Scaling plot Y axis by {10**scale_y:g} ({units_y})")
                ##ticks_y = ticker.FuncFormatter(lambda y,
                ##                pos: '{0:g}'.format(y/10**scale_y))
                ##fig.ax.yaxis.set_major_formatter(ticks_y)
                fig.ax.yaxis.set_major_formatter(NFmtr(scale_y))
            else:
                units_y = ''
            plt.xlabel(f"{fig.xlabel} ({units_x}{fig.xunit})")
            plt.ylabel(fig.ylabel % units_y)
            plt.grid(True)
            plt.legend()
            plt.subplots_adjust(left=0.15, top=0.95, right=0.95)
            if not isLinux:
                plt.show(block=False)
            outputDir = os.path.join(cwd, 'output')
            print(f"writing plots to {outputDir}")
            if not os.path.exists(outputDir):
                os.mkdir(outputDir)
            plt.savefig(os.path.join(outputDir, title))


blockClasses = {'FIELDS':        FieldsBlock,
                'MATCUBE':       MatBlock,
                'MATCYLINDERX':  MatBlock,
                'MATCYLINDERY':  MatBlock,
                'MATCYLINDERZ':  MatBlock,
                'MATMESH':       MeshMatBlock,
                'MATLAYER':      LayerMatBlock,
                'RESISTOR':      Resistor,
                'CAPACITOR':     Capacitor,
                'PROBE':         Probe,
                'SOURCE':        Source,
                'SUBSPACE':      SubSpaceBlock}

#==============================================================================

class Sim:
    s = None
    mouse_pos = None

    def __init__(self, context):
        global sims
        sims[context.scene] = self
        bpy.app.driver_namespace['fields'] = sims

        ##self.fieldsOb = ob = getFieldsOb(context)
        ##bpy.context.scene.layers[layerSnap] = 0
        obs = [ob for ob in context.visible_objects
          if ob.name.startswith("0Fields")]
        if len(obs) != 1:
            raise KeyError("expected to find one '0Fields' object")
        preob = obs[0]
        self.dx = preob.dx
        self.fieldsBlock = FieldsBlock(preob, self)
        self.fieldsOb = fob = self.fieldsBlock.ob
        self.verbose = fob.verbose
        print("Verbosity level", self.verbose)
        if self.verbose > 0:
            print("Sim init: have fob", fob.name)
        self.tms = int(time.time()*1000)
        self.state = 0
        self.history = {}
        self.dt = 0.
        self.activeOb = context.object
        bpy.app.handlers.frame_change_post.clear()
        self.onScreenInit(context)
        self.gen = self.stepWholeFDTD_G()

    #--------------------------------------------------------------------------
    # Open a connection to the server, as self.s

    def startFDTD(self):
        import socket

        HOST = "localhost"    # The remote host
        PORT = 50007              # The same port as used by the server
        print("Looking for field server")
        self.s = s = None
        for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC,
                                      socket.SOCK_STREAM):
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

    #--------------------------------------------------------------------------
    # Send a command to the FDTD server, returning nrecv bytes.
    # The default is to expect an 'A' ack.

    def send(self, text, nrecv=None, check=True):
        s = self.s
        s.send(text.encode("utf-8"))
        if nrecv is None:
            r = s.recv(1)
            if r == b'N':
                # server got an error: report it and stop (server doesn't)
                errLen = int(s.recv(2))
                errMsg = s.recv(errLen).decode("utf-8")
                raise RuntimeError(errMsg)
            elif check and r != b'A':
                raise IOError(f"Expected 'A' ack but got '{r}'")
        else:
            r = s.recv(nrecv)
        return r

    def grid(self, x):
        return ma.floor(x/self.dx)

    #--------------------------------------------------------------------------

    def getMaterialsAndDims(self):
        bmats = bpy.data.materials
        mats = {
            'FieldE':   ((0, 0, 1, 1.),          0, 0, 0),
            'FieldH':   ((1, 0, 0, 1.),          0, 0, 0),
            'FieldM':   ((0, 1, 0, 1.),          0, 0, 0),
            'FieldJ':   ((0, 1, 1, 1.),          0, 0, 0),
            'Air':      ((0.5, 0.5, 1, 0.1),     1., 1., 0.),
            'Copper':   ((0.66, 0.17, .02, 1.),  1., 1., 9.8e7),
            'Copper-T': ((0.66, 0.17, .02, 0.3), 1., 1., 9.8e7),
        }
        for k,v in mats.items():
            m = bmats.get(k)
            if m is None:
                m = bmats.new(name=k)
            ##print(f"getMaterialsAndDims: {k}")
            #print(f"m.diffuse_color={m.diffuse_color}")
            #m.diffuse_color = (0, 0, 1)
            #m['mur'] = 0
            #m['epr'] = 0
            #m['sige'] = 0
            (m.diffuse_color, m['mur'], m['epr'], m['sige']) = v
            # TODO: replace mat.use_transparency with PrincipledBSDFWrapper?
            ##if m.diffuse_color[3] < 1:
            ##    m.use_transparency = True

        # get simulation area dimensions from parent Fields object
        scn = bpy.context.scene
        ob = self.fieldsOb
        # for k in [p[1]['name'] for p in FieldsBlock.props]:
        #     setattr(self, k, getattr(ob, k))
        for key, value in FieldsBlock.props.items():
            setattr(self, key, getattr(ob, key))
        self.nx = ma.floor(ob.dimensions.x/ob.dx)
        self.ny = ma.floor(ob.dimensions.y/ob.dx)
        self.nz = ma.floor(ob.dimensions.z/ob.dx)
        if ob.verbose > 0:
            print("Fields nx,ny,nz=", self.nx, self.ny, self.nz)

        # start recording
        if ob.rec:
            scn.frame_set(1)
            self.frame_no = 1

    #--------------------------------------------------------------------------
    # Generator to create sim-blocks for all visible blocks within Fields.

    def createBlocks_G(self):
        fob = self.fieldsOb
        if fob.verbose > 0:
            print("createBlocks_G start")
        self.blocks = [self.fieldsBlock]
        obs = list(bpy.context.visible_objects)
        obs.sort(key=lambda ob: ob.name)
        scn = bpy.context.scene
        collMain.get().hide_viewport = False
        ##collSnap.get().hide_viewport = True
        haveSnap = False

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

            blockClass = blockClasses.get(bt)
            if fob.verbose > 1:
                print("createBlocks:", ob.name, bt, blockClass,
                      ob.hide_viewport)
            if ob.snap:
                if fob.verbose > 1:
                    print("Block", ob.name, "is snap")
                haveSnap = True
            if blockClass and not ob.hide_viewport:
                if fob.verbose > 1:
                    print("  createBlocks verified", ob.name)
                over = overlap(fob, ob)
                if not over:
                    if fob.verbose > 1:
                        print("no overlap for", ob.name)
                if over:
                    # create block for object or optional snapped object
                    block = blockClass(ob, self)
                    ob = block.ob
                    if fob.verbose > 1:
                        print("=====", ob.name, bt, blockClass)
                        print("  overlap=", fv(over[0]), fv(over[1]))
                    for _ in block.prepare_G():
                        yield
                    self.blocks.append(block)

        scn.frame_set(1)
        if haveSnap:
            if fob.verbose > 1:
                print("createBlocks end: haveSnap")
            ##collMain.get().hide_viewport = True
            collSnap.get().hide_viewport = False

    #--------------------------------------------------------------------------

    def findBlock(self, ob):
        for block in self.blocks:
            if block.ob == ob:
                return block
        return None

    #--------------------------------------------------------------------------

    def sendDefs_G(self):
        ##print("Sim.sendDefs_G start")

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
                        m = bpy.data.materials.get(ob.fmatName)
                    else:
                        m = ob.data.materials[0]
                        if not m and ob.material_slots:
                            m = ob.material_slots[0].material
            else:
                if ob.material_slots:
                    m = ob.material_slots[0].material

            if m and not m in fmats:
                fmats.append(m)
                if ob.verbose > 1:
                    print(" adding fmat", m.name)
                self.send(f"M {m.name} {m.mur:g} {m.epr:g} {m.sige:g} 0\n")

        # send general sim parameters to server
        for _ in self.fieldsBlock.sendSimDef_G():
            yield

        # send defintions for sim objects
        for block in self.blocks:
            block.state = 0
            if block != self.fieldsBlock:
                for _ in block.sendDef_G():
                    yield

    #--------------------------------------------------------------------------
    # Process one timestep of each sim block.

    def doStepBlocks(self):
        scn = bpy.context.scene
        if self.verbose > 0:
            tms = int(time.time()*1000)
            dtms = tms - self.tms
            self.tms = tms
            print(f"[{dtms}>{scn.frame_current}:] ", end="")
            sys.stdout.flush()
        rec = self.rec
        if rec:
            scn.frame_set(self.frame_no)
            self.frame_no += 1
        for block in self.blocks:
            if hasattr(block, 'doStep') and not block.ob.hide_viewport:
                block.doStep()
            if self.state == 0:
                print("doStepBlocks stopped")
                break
        scn.frame_set(scn.frame_current + 1)

    #--------------------------------------------------------------------------
    # Do one timer step: first initialize simulation, then step it.

    def stepWholeFDTD_G(self):
        print("stepWholeFDTD_G start")
        self.getMaterialsAndDims()

        try:
            self.startFDTD()
        except IOError:
            xcodeRun()
            time.sleep(5)
            self.startFDTD()

        # tell simulator to chdir to blender file's directory
        cmd = f"C {cwd}\n"
        if self.verbose > 1:
            print(cmd)
        self.send(cmd)
        yield

        for _ in self.createBlocks_G():
            yield
        for _ in self.sendDefs_G():
            yield
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = self.activeOb
        if self.activeOb:
            self.activeOb.select_set(True)
        self.dt = 0.5 * self.dx*mm / c0
        try:
            self.send('R')
        except IOError:
            print("*** Start Field Server first! ***")
            raise
        self.state = 3
        while self.state > 0:
            yield
            if self.state == 3:
                self.doStepBlocks()
        print("stepWholeFDTD_G done")
        raise StopIteration

    #--------------------------------------------------------------------------
    # Tell the server to pause/unpause simulation.

    def pause(self, context):
        context.area.tag_redraw()   # update status line
        ack = self.send('D', 1)
        if len(ack) < 1 or ack[0] != ord('A'):
            ##print("non-A ack:", ack)
            return

    #--------------------------------------------------------------------------
    # On-screen status display.

    def onScreenInit(self, context):
        print("onScreenInit")
        # TODO: make onScreenInit work from "Run FDTD" button too
        area = context.area
        if area and area.type == 'VIEW_3D':
            self.lastPosMeas = None
            oldh = bpy.app.driver_namespace.get('fields_handle')
            if oldh:
                bpy.types.SpaceView3D.draw_handler_remove(oldh, 'WINDOW')
            args = (context,)
            self.handle = bpy.types.SpaceView3D.draw_handler_add(
                        self.drawCallback, args, 'WINDOW', 'POST_PIXEL')
            posth = bpy.app.handlers.frame_change_post
            if self.frameHandler in posth:
                del posth[self.frameHandler]
            posth.append(self.frameHandler)
            self.area3d = context.area
            bpy.app.driver_namespace['fields_handle'] = self.handle
            bpy.app.handlers.load_pre.append(self.onScreenRemove)

    def drawCallback(self, context):
        ##try:
 
        ##print("trying draw status")
        ##print("self=", self)
        scn = bpy.context.scene
        font_id = 0
        w = context.region.width
        font_scale = (2, 1)[isLinux]
        blf.position(font_id, w-200*font_scale, 10, 0)
        blf.color(font_id, 255, 255, 255, 255)
        blf.size(font_id, 12*font_scale)
        status = f"{scn.frame_current * self.dt * 1e12:9.3f} ps"
        if self and self.state > 3:
            status += '  PAUSED'
        blf.draw(font_id, status)
 
        ##except:       # !!! which error to ignore? not all of them!
        ##    pass

        # draw mobile-probe measurement value next to mouse pointer
        mpos = self.mouse_pos
        ob = bpy.context.object
        haveDrawnMeas = False
        if mpos and ob:
            pblock = self.findBlock(ob)
            if hasattr(pblock, 'measurement_attr_name'):
                aname = pblock.measurement_attr_name
                units = pblock.measurement_units
                if hasattr(ob, aname):
                    meas = getattr(ob, aname)
                    ##print(f"drawing measurement {meas:g}")
                    x = mpos[0] + 15
                    y = mpos[1] + 15
                    blf.position(font_id, x, y, 0)
                    self.lastPosMeas = [x, y]
                    blf.draw(font_id, f"{meas:g} {units}")
                    haveDrawnMeas = True

        # erase measurement when another object is selected
        if not haveDrawnMeas and self.lastPosMeas is not None:
            ##print("clearing measurement")
            x, y = self.lastPosMeas
            blf.position(font_id, x, y, 0)
            blf.draw(font_id, "")
            self.lastPosMeas = None

    def frameHandler(self, scene, depsgraph):
        self.area3d.tag_redraw()   # update status line

    def onScreenRemove(self, dummy1, dummy2):
        ##print("onScreenRemove", self)
        bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
        bpy.app.driver_namespace.pop('fields_handle')
        bpy.context.area.tag_redraw()


#==============================================================================

class FieldOperator(bpy.types.Operator):
    """Run FDTD simulation (Cmd-R)"""
    bl_idname = "fdtd.run"
    bl_label = "Run FDTD"

    timer = None
    sim = None

    #--------------------------------------------------------------------------
    # Timer routines for modal operator.

    def startTimer(self):
        sim = self.sim
        if not sim:
            return
        context = self.context
        context.window_manager.modal_handler_add(self)
        rate = (sim.fieldsOb.get('msRate') or 200)
        rate = max(min(rate, 1000), 10)
        print(f"starting {rate} ms/tick timer")
        self.timer = context.window_manager.event_timer_add(
                                            rate/1000., window=context.window)

    def stopTimer(self):
        self.context.window_manager.event_timer_remove(self.timer)
        self.timer = None

    #--------------------------------------------------------------------------
    # Dynamic probing.

    def dynProbe(self, event, action):
        ob = bpy.context.object
        name = ob.name if ob else "<no object>"
        ##print("dynProbe:", action, name, ob.p_verbose)
        if ob and ob.p_verbose > 0:
            print("dynProbe:", action, name)
        sim = self.sim
        if ob and ob.get('blockType') == 'PROBE':
            pblock = sim.findBlock(ob)
            ##print("found block", pblock)
            if pblock:
                loc = getMouse3D(event)
                sim.mouse_pos = [event.mouse_region_x, event.mouse_region_y]
                if loc is None:
                    print(f"No C.space_data for event {event}")
                else:
                    if action == 'START':
                        if ob.p_verbose > 0:
                            print("dynProbe: starting move", ob.name)
                        self.probeDrag = ob
                        self.obRelMouse = Vector(ob.location) - Vector(loc)
                    elif action == 'MOVE':
                        if ob.p_verbose > 0:
                            print("dynProbe: moving", ob.name, pblock)
                        #obNewLoc = self.obRelMouse + Vector(loc)
                        ###print("obNewLoc=", fv(obNewLoc))
                        #if self.lockAxis == 0:
                        #    ob.location.x = obNewLoc.x
                        #elif self.lockAxis == 1:
                        #    ob.location.y = obNewLoc.y
                        #elif self.lockAxis == 2:
                        #    ob.location.z = obNewLoc.z
                        #else:
                        #    ob.location = obNewLoc
                        pblock.Bs, pblock.Be = bounds(ob)
                        pblock.sendDef(update=True)
                        pblock.lastStep -= 1
                        pblock.doStep()
                    else:
                        if ob.p_verbose > 1:
                            print("dynProbe: move done")
                        self.probeDrag = None
        return {'RUNNING_MODAL'}

    #--------------------------------------------------------------------------

    def modal(self, context, event):
        sim = self.sim
        Pm = Vector((event.mouse_x, event.mouse_y))
        S = self.winStart; E = self.winEnd
        inWin = (Pm.x >= S.x and Pm.x < E.x and Pm.y >= S.y and Pm.y < E.y)
        ##print("mouse @", fv(Pm), inWin, fv(S), fv(E))

        ##if inWin and not event.type in ('TIMER'):
        ##    print("event:", event.value, event.type, "oskey=", event.oskey)

        if sim.verbose > 1 and event.type == 'P':
            print(f"'P', oskey={event.oskey} value={event.value}")
        if event.value == 'PRESS':
            if not inWin:
                return {'PASS_THROUGH'}

            if self.probeDrag:
                if event.type in ('LEFTMOUSE', 'RET', 'ESC'):
                    return self.dynProbe(event, 'DONE')

            if sim.state >= 3:
                ob = bpy.context.object
                if event.type == 'G':
                    if ob.blockType == 'PROBE' and ob.p_shape == 'Point':
                        self.lockAxis = None
                        self.dynProbe(event, 'START')
                    return {'PASS_THROUGH'}

                elif event.type in 'XYZ':
                    if ob.blockType == 'PROBE' and ob.p_shape == 'Point':
                        self.lockAxis = ord(event.type) - ord('X')
                        # return {'RUNNING_MODAL'}
                    return {'PASS_THROUGH'}

                elif event.type == 'ESC':
                    ##print("ESC pressed while in sim")
                    return self.cancel(context)

            elif event.type == 'ESC':
                ##print("ESC pressed")
                return self.cancel(context)

        elif event.value == 'RELEASE':
            ##print(f"Release: {inWin=} {self.probeDrag=} {event.type=}")
            if not inWin:
                return {'PASS_THROUGH'}
            if self.probeDrag:
                #if event.type == 'MOUSEMOVE':
                #    return self.dynProbe(event, 'MOVE')
                if event.type in ('LEFTMOUSE', 'RET', 'ESC'):
                    return self.dynProbe(event, 'DONE')

        elif event.type == 'TIMER':
            if self.timer is None:
                return {'CANCELLED'}
            else:
                ##print(f"Timer: {inWin=} {self.probeDrag=} {event.type=}")
                if inWin and self.probeDrag:
                    return self.dynProbe(event, 'MOVE')

                try:
                    sim.gen.__next__()
                except StopIteration:
                    if sim.verbose > 1:
                        print("modal: timer: stepWholeFDTD_G done")
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
        global sims, fieldOperator
        fieldOperator = self
        sim = sims.get(context.scene)
        print("\n=== Starting BField FDTD simulation ===")
        self.context = context
        winrgn = context.area.regions[-1]
        self.winStart = S = Vector((winrgn.x, winrgn.y))
        self.winEnd = E = self.winStart + Vector((winrgn.width, winrgn.height))
        ##print("win @", fv(S), fv(E))
        if sim:
            if sim.state > 0:
                print("Stopping current sim")
                sim.operator.cancel(context)
        self.sim = Sim(context)
        self.sim.operator = self
        self.probeDrag = None
        self.startTimer()
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ##print(f"FDTD execute: {context=}")
        return self.invoke(context, None)

    def cancel(self, context):
        sim = self.sim
        scn = context.scene
        s = sim.s
        if s:
            sim.send('Ex', check=False)
            s.close()
        if self.timer:
            self.stopTimer()
        sim.state = 0
        scn.frame_end = scn.frame_current

        # refresh viewport to remove any "PAUSED" on status line
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        print("FDTD stopped.")
        return {'CANCELLED'}


#==============================================================================

def cleanTmps():
    """Remove all probe-generated objects and images"""

    tmpc = bpy.data.collections.get('Tmp')
    if tmpc:
        objs = bpy.data.objects
        for ob in tmpc.all_objects.values():
            objs.remove(ob, do_unlink=True)

    #imgs = bpy.data.images
    #for name,img in imgs.items():
    #    if name.startswith('probe_'):
    #        imgs.remove(img, do_unlink=True)

    bpy.app.handlers.frame_change_post.clear()

class FieldCleanOperator(bpy.types.Operator):
    """Clean up after FDTD simulation (Cmd-K)"""
    bl_idname = "fdtd.clean"
    bl_label = "Clean FDTD: remove probe result objects"

    def invoke(self, context, event):
        print("Clean-FDTD invoke")
        cleanTmps()
        return {'FINISHED'}


class FieldPauseOperator(bpy.types.Operator):
    """Pause/unpause FDTD simulation (P)"""
    bl_idname = "fdtd.pause"
    bl_label = "Pause"

    def invoke(self, context, event):
        global sims
        sim = sims[context.scene]
        ##print("Pause-FDTD invoke")
        if not sim or sim.state < 3:
            print("FDTD not running")
        elif sim.state == 3:
            print("=== PAUSE ===")
            sim.pause(context)
            sim.state = 4
        else:
            print("=== UNPAUSE ===")
            sim.pause(context)
            sim.state = 3
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
        Probe.plotAllSetUp()
        for block in sim.blocks:
            if type(block) == Probe:
                ob = block.ob
                block.plot()
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # wait for Cmd key to be released, or plot window messes oskey state
        if isLinux or (event.type == 'OSKEY' and event.value == 'RELEASE'):
            print("finishing plot")
            Probe.plotAllFinish()
            return {'FINISHED'}
        return {'PASS_THROUGH'}


#==============================================================================

class FieldCenterOperator(bpy.types.Operator):
    """Center object origin"""
    bl_idname = "fdtd.center"
    bl_label = "Center object origin"

    def invoke(self, context, event):
        print("FDTD Center invoke")
        op = bpy.ops.object
        op.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        return {'FINISHED'}


#==============================================================================

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
        layout.prop_search(ob, "blockType", scene, "blockTypes",
                           text="Block type")
        bt = ob.blockType
        if bt:
            blockClasses[bt].drawProps(ob, layout, scene)
        else:
            if ob.material_slots:
                mat = ob.material_slots[0]
                if mat.name in ('FieldH', 'FieldE'):
                    field = mat.name[-1]
                    units = Probe.fieldUnits[('E', 'M')[field == 'H']]
                    Mr = ob.rotation_euler.to_matrix()
                    V = Mr @ Vector((1,0,0))
                    ap = ob.parent
                    r = ob.scale.x / ap.p_sfactor
                    if ap.p_log:
                        r = 10**(r*4 - 6)
                    r /= ap.p_magScale
                    V.x *= r
                    V.y *= r
                    V.z *= r
                    box = layout.box()
                    box.label(text=f"{field} = {gv(V)} {units}")
                    box.label(text=f"   = {V.length:g} {units}")

        layout.operator("fdtd.run")
        layout.operator("fdtd.pause")
        layout.operator("fdtd.plot")
        layout.operator("fdtd.center")
        layout.operator("fdtd.clean")

def populateTypes(scene):
    bpy.app.handlers.depsgraph_update_pre.remove(populateTypes)
    scene.blockTypes.clear()
    for k,block in blockClasses.items():
        scene.blockTypes.add().name = k
        if hasattr(block, 'populateTypes'):
            block.populateTypes(scene)

#==============================================================================

class FieldMatPanel(bpy.types.Panel):
    """Creates a FDTD Panel in the Material properties window"""
    bl_label = "FDTD"
    bl_idname = "MATERIAL_PT_FDTD"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    mur: bp.FloatProperty(description="Relative Permeability",
                          min=0., default=1.)
    epr: bp.FloatProperty(description="Relative Permittivity",
                          min=0., default=1.)
    sige: bp.FloatProperty(min=0., default=0.,
                           description="Eletrical Conductivity [S/m]")

    @classmethod
    def createTypes(cls):
        # for prop in cls.props:
        #     setattr(bpy.types.Material, prop[1]['name'], prop)
        ##print(f"FMP.createTypes: .__annotations__={cls.__annotations__}")
        for key, value in cls.__annotations__.items():
            ##print(f"FMP.createTypes: {key}={value}")
            setattr(bpy.types.Material, key, value)

    @classmethod
    def delTypes(cls):
        # for prop in cls.props:
        #     delattr(bpy.types.Material, prop[1]['name'])
        for key in cls.props.keys():
            delattr(bpy.types.Material, key)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        ob = context.object
        if ob.material_slots:
            mat = ob.material_slots[0].material
            if ob.blockType in MatBlock.mtypeCodes.keys():
                layout.prop(mat, 'mur', text="Relative Permeability µr")
                layout.prop(mat, 'epr', text="Relative Permittivity εr")
                layout.prop(mat, 'sige', text="Conductivity σE")
            if mat.name in ('FieldH', 'FieldE'):
                field = mat.name[-1]
                units = Probe.fieldUnits[('E', 'M')[field == 'H']]
                Mr = ob.rotation_euler.to_matrix()
                V = Mr @ Vector((1,0,0))
                r = ob.scale.x
                ap = ob.parent
                if ap.p_log:
                    r = 10**(r*4 - 6)
                r /= ap.p_magScale
                V.x *= r
                V.y *= r
                V.z *= r
                box = layout.box()
                box.label(text=f"{field} = {gv(V)} {units}")
                box.label(text=f"   = {V.length:g} {units}")

#==============================================================================
# Register.

addon_keymaps = []
wm = bpy.context.window_manager

operatorsPanels = (
    FieldOperator,
    FieldCleanOperator,
    FieldPauseOperator,
    FieldPlotOperator,
    FieldCenterOperator,
    FieldObjectPanel,
    FieldMatPanel
)

def register():
    ##bpy.utils.register_module(__name__)
    for c in operatorsPanels:
        bpy.utils.register_class(c)

    # assign Cmd-R shortcut to 'Run FDTD', etc.
    km = wm.keyconfigs.addon.keymaps.new(name='Object Mode',space_type='EMPTY')
    km.keymap_items.new("fdtd.run",   'R', 'PRESS', oskey=True)
    km.keymap_items.new("fdtd.clean", 'K', 'PRESS', oskey=True)
    km.keymap_items.new("fdtd.plot",  'P', 'PRESS', oskey=True)
    km.keymap_items.new("fdtd.pause", 'P', 'PRESS')
    addon_keymaps.append(km)

    bpy.types.Scene.blockTypes = bp.CollectionProperty(
                                            type=bpy.types.PropertyGroup)
    for k,block in blockClasses.items():
        if hasattr(block, 'registerTypes'):
            block.registerTypes()
    bpy.types.Object.blockType = bp.StringProperty()
    bpy.app.handlers.depsgraph_update_pre.append(populateTypes)

    for cls in blockClasses.values():
        ##print(f"creating types for {cls.__name__}:")
        cls.createTypes()
    FieldMatPanel.createTypes()

def unregister():
    print("unregister:")
    ##bpy.utils.unregister_module(__name__)
    for c in classes:
       bpy.utils.unregister_class(c)
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    addon_keymaps.clear()
    del bpy.types.Scene.blockTypes
    del bpy.types.Object.blockType
    for cls in blockClasses.values():
        cls.delTypes()
    for k,block in blockClasses.items():
        if hasattr(block, 'unregisterTypes'):
            block.unregisterTypes()
    FieldMatPanel.delTypes()
    print("unregister done.")

if __name__ == "__main__":
    register()
    ##getFieldsOb(bpy.context, create=False)
