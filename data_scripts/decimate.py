import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]

input_path = argv[0]
output_path = argv[1]
angle_limit = float(argv[2])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.obj(filepath=input_path)

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
modifier.decimate_type = 'DISSOLVE'
modifier.angle_limit = angle_limit * (3.14159 / 180.0)  # degrees to radians

bpy.ops.object.modifier_apply(modifier="Decimate")
bpy.ops.export_scene.obj(filepath=output_path)
