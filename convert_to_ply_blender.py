import glob
import os
import mathutils
import bpy, bmesh
def import_one(filename):
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass

    bpy.data.objects['Cube'].select_set(True)  # Blender 2.8x

    bpy.ops.object.delete()
    meshes_to_join = {}

    # Import the objs, keeping track of existing ones
    orig_existing_obj_names = set([obj.name for obj in bpy.data.objects])
    print("Importing " + filename + "...")

    # Keep track of existing objects.
    exist_obj_names_tmp = set([obj.name for obj in bpy.data.objects])

    # Load in new objects.
    if filename.upper().endswith(".OBJ"):
        bpy.ops.import_scene.obj(filepath=filename)
        # Not sure why OBJ imported from VMD are rotated.
        initial_rotation = (90, 0, 0)
    else:
        bpy.ops.import_scene.x3d(filepath=filename)
        # Not sure why WRL imported from PyMol are rotated.
        initial_rotation = (270, 0, 180)

    # Get new objects.
    new_obj_names_tmp = set([
        obj.name for obj in bpy.data.objects
    ]) - exist_obj_names_tmp

    new_objs_tmp = [bpy.data.objects[obj_name]
                    for obj_name in new_obj_names_tmp
                    if bpy.data.objects[obj_name].type == "MESH"]
    meshes_to_join[filename] = new_objs_tmp

    # Get a list of the names of objects just added.


    new_obj_names = set([
        obj.name for obj in bpy.data.objects
    ]) - orig_existing_obj_names

    # Keep the ones that are meshes
    new_objs = [bpy.data.objects[obj_name]
                for obj_name in new_obj_names
                if bpy.data.objects[obj_name].type == "MESH"]

    # Delete the ones that aren't meshes
    # See https://blender.stackexchange.com/questions/27234/python-how-to-completely-remove-an-object
    for obj_name in new_obj_names:
        obj = bpy.data.objects[obj_name]
        if obj.type != "MESH":
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(state=True)
            bpy.ops.object.delete()
    new_obj_names = [o.name for o in new_objs]

    # Apply the rotations of all meshes
    for obj in new_objs:
        bpy.ops.object.select_all(action='DESELECT')
        # bpy.context.scene.objects.active = obj
        bpy.context.view_layer.objects.active = obj
        obj.select_set(state=True)
        bpy.ops.object.transform_apply(
            location=False, scale=False, rotation=True
        )

    # Join some of the objects
    for filename in meshes_to_join.keys():
        objs_to_merge = meshes_to_join[filename]
        if len(objs_to_merge) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in objs_to_merge:
                obj.select_set(state=True)
            bpy.context.view_layer.objects.active = objs_to_merge[0]
            bpy.ops.object.join()
        if len(objs_to_merge) > 0:
            objs_to_merge[0].name = "BldMl__" + os.path.basename(filename)
        else:
            # Sometimes PyMol (at least) doesn't save a file at all,
            # perhaps because the selection was empty?
            pass

    # Make sure origins of all new meshes is 0, 0, 0
    # See https://blender.stackexchange.com/questions/35825/changing-object-origin-to-arbitrary-point-without-origin-set
    for obj in bpy.data.objects:
        if obj.name.startswith("BldMl__"):
            loc = obj.location
            try:
                obj.data.transform(mathutils.Matrix.Translation(loc))
                obj.matrix_world.translation -= loc
            except  Exception as exc:
                print(exc)
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = obj
            obj.select_set(state=True)
            # bpy.ops.export_scene.obj(filepath='converted/{}.obj'.format(obj.name))
            bpy.ops.export_mesh.ply(filepath='converted/{}.ply'.format(obj.name), use_selection=True)





    #
def import_all_mesh_files(nanometers=True, remove_doubles=True):
    """
    Import all the meshes produced by the external visualization program
    (VMD or PyMol), saved to the temporary directory.

    :param ??? my_operator: The operator, used to access user-parameter
                variables.

    :returns: List of the names of the added meshes.
    :rtype: :class:`str[]`
    """


    mask1 = 'objects/' + "*.obj"
    mask2 = 'to_convert/' + "*.wrl"
    for filename in  glob.glob(mask2)+ glob.glob(mask2):
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects['Cube'].select_set(True)  # Blender 2.8x
        except:
            pass



        bpy.ops.object.delete()
        meshes_to_join = {}

        # Import the objs, keeping track of existing ones
        orig_existing_obj_names = set([obj.name for obj in bpy.data.objects])
        print("Importing " + filename + "...")

        # Keep track of existing objects.
        exist_obj_names_tmp = set([obj.name for obj in bpy.data.objects])

        # Load in new objects.
        if filename.upper().endswith(".OBJ"):
            bpy.ops.import_scene.obj(filepath=filename)
            # Not sure why OBJ imported from VMD are rotated.
            initial_rotation = (90, 0, 0)
        else:
            bpy.ops.import_scene.x3d(filepath=filename)
            # Not sure why WRL imported from PyMol are rotated.
            initial_rotation = (270, 0, 180)

        # Get new objects.
        new_obj_names_tmp = set([
            obj.name for obj in bpy.data.objects
        ]) - exist_obj_names_tmp

        new_objs_tmp = [bpy.data.objects[obj_name]
                        for obj_name in new_obj_names_tmp
                        if bpy.data.objects[obj_name].type == "MESH"]
        meshes_to_join[filename] = new_objs_tmp

        # Get a list of the names of objects just added.
        new_obj_names = set([
            obj.name for obj in bpy.data.objects
        ]) - orig_existing_obj_names

        # Keep the ones that are meshes
        new_objs = [bpy.data.objects[obj_name]
                    for obj_name in new_obj_names
                    if bpy.data.objects[obj_name].type == "MESH"]

        # Delete the ones that aren't meshes
        # See https://blender.stackexchange.com/questions/27234/python-how-to-completely-remove-an-object
        for obj_name in new_obj_names:
            obj = bpy.data.objects[obj_name]
            if obj.type != "MESH":
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(state=True)
                bpy.ops.object.delete()
        new_obj_names = [o.name for o in new_objs]

        # Apply the rotations of all meshes
        for obj in new_objs:
            bpy.ops.object.select_all(action='DESELECT')
            # bpy.context.scene.objects.active = obj
            bpy.context.view_layer.objects.active = obj
            obj.select_set(state=True)
            bpy.ops.object.transform_apply(
                location=False, scale=False, rotation=True
            )

        # Join some of the objects
        for filename in meshes_to_join.keys():
            objs_to_merge = meshes_to_join[filename]
            if len(objs_to_merge) > 1:
                bpy.ops.object.select_all(action='DESELECT')
                for obj in objs_to_merge:
                    obj.select_set(state=True)
                bpy.context.view_layer.objects.active = objs_to_merge[0]
                bpy.ops.object.join()
            if len(objs_to_merge) > 0:
                objs_to_merge[0].name = "BldMl__" + os.path.basename(filename)
            else:
                # Sometimes PyMol (at least) doesn't save a file at all,
                # perhaps because the selection was empty?
                pass

        # Make sure origins of all new meshes is 0, 0, 0
        # See https://blender.stackexchange.com/questions/35825/changing-object-origin-to-arbitrary-point-without-origin-set
        for obj in bpy.data.objects:
            if obj.name.startswith("BldMl__"):
                loc = obj.location
                try:
                    obj.data.transform(mathutils.Matrix.Translation(loc))
                    obj.matrix_world.translation -= loc
                except  Exception as exc:
                    print(exc)
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = obj
                obj.select_set(state=True)
                # bpy.ops.export_scene.obj(filepath='converted/{}.obj'.format(obj.name))
                bpy.ops.export_mesh.ply(filepath='converted/{}.ply'.format(obj.name.replace("BldMl__",'')), use_selection=True)

        if nanometers == True:
            for obj in bpy.data.objects:
                if obj.name.startswith("BldMl__"):
                    obj.scale = [0.1, 0.1, 0.1]

    return new_obj_names
def convert_one(name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if  'Camera' and 'Cube' in bpy.context.scene.objects:
        objs = [bpy.context.scene.objects['Camera'], bpy.context.scene.objects['Cube']]
        bpy.ops.object.delete({"selected_objects": objs})
    bpy.ops.import_scene.x3d(filepath='to_convert/'+name+'.wrl')
    objs_to_merge=[]
    for obj in bpy.data.objects:
        if bpy.data.objects[obj.name].type == "MESH":
              objs_to_merge.append(bpy.data.objects[obj.name])

    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs_to_merge:
            obj.select_set(state=True)
    bpy.context.view_layer.objects.active = objs_to_merge[0]
    bpy.ops.object.join()

    for obj in bpy.data.objects:
        if bpy.data.objects[obj.name].type == "MESH":

            coords=[(obj.matrix_world @ v.co) for v in obj.data.vertices]

            bpy.ops.export_mesh.ply(filepath='converted/'+name+'.ply', use_selection=True,use_mesh_modifiers =True,
                                    use_normals=True,use_uv_coords=True,use_colors =True)

    bpy.ops.export_scene.obj(filepath='converted/'+name+'.obj', use_edges=True)

import_all_mesh_files()
