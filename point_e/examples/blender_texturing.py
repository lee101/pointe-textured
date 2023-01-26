
import bpy
import bmesh
import math

def stripeMaterial():
    # create a new 3x1 image
    i1 = bpy.data.images.new("stripe", 3, 1)

    # set the pixels of the tiny image
    i1.pixels = [
            #R,G,B,A,
    1,0,0,1, # red
    1,1,1,1, # white
    0,0,1,1] # blue

    # if we don't pack the image into the .blend, it will be lost when you re-load the project
    i1.pack(True)

    # create a texture for the image
    t1 = bpy.data.textures.new("stripe", 'IMAGE')
    t1.image = i1

    # give it sharp uninterpolated edges.
    t1.use_interpolation = False
    t1.filter_type = 'BOX'

    # put the texture on a material with UV coordinates
    m1 = bpy.data.materials.new('stripe')
    m1.texture_slots.add()
    m1.texture_slots[0].texture = t1
    m1.texture_slots[0].texture_coords = 'UV'
    m1.texture_slots[0].uv_layer = 'spiral'

    return m1


def pole(name, nFaces, z1):
    # this makes a simple open-ended cylinder.
    mesh = bpy.data.meshes.new(name)

    verts = []
    faces = []
    for i in range(nFaces):
        theta = math.pi*2 *i/nFaces
        c = math.cos(theta)
        s = math.sin(theta)
        verts.append( [ c,s,0])
        verts.append( [ c,s,z1])
        v1 = i*2
        v2 = v1+1
        v3 = v1+2
        v4 = v1+3
        if (v3>=2*nFaces):
            v3 = 0
            v4 = 1
        faces.append( [ v1, v3, v4, v2] )
    mesh.from_pydata(verts, [], faces)
    mesh.validate(True, verbose=True)
    mesh.show_normal_face = True

    obj = bpy.data.objects.new(name, mesh)

    return obj


def spiralUVs(mesh, xPlus):
    # add a UV layer called "spiral" and make it slanted.
    mesh.uv_textures.new("spiral")
    bm = bmesh.new()
    bm.from_mesh(mesh)

    bm.faces.ensure_lookup_table()

    uv_layer = bm.loops.layers.uv[0]

    nFaces = len(bm.faces)
    for fi in range(nFaces):
        x0 = fi*2/nFaces
        x1 = (fi+1)*2/nFaces
        bm.faces[fi].loops[0][uv_layer].uv = (x0, 0)
        bm.faces[fi].loops[1][uv_layer].uv = (x1, 0)
        bm.faces[fi].loops[2][uv_layer].uv = (xPlus+x1, 1)
        bm.faces[fi].loops[3][uv_layer].uv = (xPlus+x0, 1)
    bm.to_mesh(mesh)

# put the image behind the UV layer for the times when you're using the Multitexture shader on the 3D view.
def setPreviewTex(mesh, img):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    tl = bm.faces.layers.tex.active
    if tl:
        for f in bm.faces:
            f[tl].image = img
    bm.to_mesh(mesh)


def cliche():
    # tie it all together
    obj = pole("stripes", 20, 5)

    bpy.context.scene.objects.link(obj)

    obj.data.materials.append(stripeMaterial())

    spiralUVs(obj.data,2)

    setPreviewTex(obj.data, obj.data.materials[0].texture_slots[0].texture.image)

#
#

cliche()
