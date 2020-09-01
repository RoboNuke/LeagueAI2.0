import bpy
import os
from math import radians
from time import sleep

context = bpy.context

#create a scene
scene = bpy.data.scenes.new("League AI Data Generator")

# do the same for lights etc


objects = bpy.data.objects
# Add cube
bpy.ops.mesh.primitive_cube_add( location= (0,0,0), rotation=(0,0,0))

# Add Circle
bpy.ops.curve.primitive_bezier_circle_add(radius = 5, location= (0,0,0), rotation=(0,0,0))

# Add Empty
bpy.ops.object.empty_add(type='CUBE', location= (0,0,5), rotation=(0,0,0))
# constrain empty to curve

empty = objects['Empty']
bpy.ops.object.constraint_add(type="FOLLOW_PATH")
cCon = empty.constraints['Follow Path']
cCon.target = objects['BezierCircle']
cCon.use_fixed_location = True
cCon.offset_factor = 0


# Add camera
bpy.ops.object.camera_add( location= (0,0,0), rotation=(0,0,0))

#print(objects)
cam = objects['Camera']

cam.parent = empty
bpy.ops.object.constraint_add(type="TRACK_TO")
camCon = cam.constraints['Track To']
camCon.target = objects['Cube']
camCon.track_axis = 'TRACK_NEGATIVE_Z'
camCon.up_axis = 'UP_Y'

scene.update()
context.scene.camera = cam
sceney = context.scene
sceney.render.filepath = 'cube_test'
sceney.render.image_settings.file_format = 'AVI_JPEG'
sceney.frame_end = 24
sceney.frame_start = 0
bpy.ops.render.render(animation=True)

"""
for model_path in models:
    scene.camera = camera
    path = os.path.join(models_path, model_path)
    # make a new scene with cam and lights linked
    context.screen.scene = scene
    bpy.ops.scene.new(type='LINK_OBJECTS')
    context.scene.name = model_path
    cams = [c for c in context.scene.objects if c.type == 'CAMERA']
    #import model
    bpy.ops.import_scene.obj(filepath=path, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl")
    for c in cams:
        context.scene.camera = c                                    
        print("Render ", model_path, context.scene.name, c.name)
        context.scene.render.filepath = "somepathmadeupfrommodelname"
        bpy.ops.render.render(write_still=True)
"""
