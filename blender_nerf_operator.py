import os
import math
import json
import datetime
import bpy
import bmesh
import struct


# global addon script variables
OUTPUT_TRAIN = 'train'
OUTPUT_TEST = 'test'
CAMERA_NAME = 'BlenderNeRF Camera'
TMP_VERTEX_COLORS = 'blendernerf_vertex_colors_tmp'


# blender nerf operator parent class
class BlenderNeRF_Operator(bpy.types.Operator):

    # camera intrinsics
    def get_camera_intrinsics(self, scene, camera):
        camera_angle_x = camera.data.angle_x
        camera_angle_y = camera.data.angle_y

        # camera properties
        f_in_mm = camera.data.lens # focal length in mm
        scale = scene.render.resolution_percentage / 100
        width_res_in_px = scene.render.resolution_x * scale # width
        height_res_in_px = scene.render.resolution_y * scale # height
        optical_center_x = width_res_in_px / 2
        optical_center_y = height_res_in_px / 2

        # pixel aspect ratios
        size_x = scene.render.pixel_aspect_x * width_res_in_px
        size_y = scene.render.pixel_aspect_y * height_res_in_px
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        # sensor fit and sensor size (and camera angle swap in specific cases)
        if camera.data.sensor_fit == 'AUTO':
            sensor_size_in_mm = camera.data.sensor_height if width_res_in_px < height_res_in_px else camera.data.sensor_width
            if width_res_in_px < height_res_in_px:
                sensor_fit = 'VERTICAL'
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
            elif width_res_in_px > height_res_in_px:
                sensor_fit = 'HORIZONTAL'
            else:
                sensor_fit = 'VERTICAL' if size_x <= size_y else 'HORIZONTAL'

        else:
            sensor_fit = camera.data.sensor_fit
            if sensor_fit == 'VERTICAL':
                sensor_size_in_mm = camera.data.sensor_height if width_res_in_px <= height_res_in_px else camera.data.sensor_width
                if width_res_in_px <= height_res_in_px:
                    camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

        # focal length for horizontal sensor fit
        if sensor_fit == 'HORIZONTAL':
            sensor_size_in_mm = camera.data.sensor_width
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

        # focal length for vertical sensor fit
        if sensor_fit == 'VERTICAL':
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

        camera_intr_dict = {
            'camera_angle_x': camera_angle_x,
            'camera_angle_y': camera_angle_y,
            'fl_x': s_u,
            'fl_y': s_v,
            'k1': 0.0,
            'k2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'cx': optical_center_x,
            'cy': optical_center_y,
            'w': width_res_in_px,
            'h': height_res_in_px,
            'aabb_scale': scene.aabb
        }

        return {'camera_angle_x': camera_angle_x} if scene.nerf else camera_intr_dict

    # camera extrinsics (transform matrices)
    def get_camera_extrinsics(self, scene, camera, mode='TRAIN', method='SOF'):
        assert mode == 'TRAIN' or mode == 'TEST'
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        if scene.splats and scene.splats_test_dummy and mode == 'TEST':
            return []

        initFrame = scene.frame_current
        step = scene.train_frame_steps if (mode == 'TRAIN' and method == 'SOF') else scene.frame_step
        if (mode == 'TRAIN' and method == 'COS'):
            end = scene.frame_start + scene.cos_nb_frames - 1
        elif (mode == 'TRAIN' and method == 'TTC'):
            end = scene.frame_start + scene.ttc_nb_frames - 1
        else:
            end = scene.frame_end

        camera_extr_dict = []
        for frame in range(scene.frame_start, end + 1, step):
            scene.frame_set(frame)
            filename = os.path.basename( scene.render.frame_path(frame=frame) )
            filedir = OUTPUT_TRAIN * (mode == 'TRAIN') + OUTPUT_TEST * (mode == 'TEST')

            frame_data = {
                'file_path': os.path.join(filedir, os.path.splitext(filename)[0] if scene.splats else filename),
                'transform_matrix': self.listify_matrix(camera.matrix_world)
            }

            camera_extr_dict.append(frame_data)

        scene.frame_set(initFrame) # set back to initial frame

        return camera_extr_dict

    # # export vertex colors for each visible mesh
    # def save_splats_ply(self, scene, directory):
    #     # create temporary vertex colors
    #     for obj in scene.objects:
    #         if obj.type == 'MESH':
    #             if not obj.data.vertex_colors:
    #                 obj.data.vertex_colors.new(name=TMP_VERTEX_COLORS)

    #     if bpy.context.object is None:
    #         self.report({'INFO'}, 'No object active. Setting first object as active.')
    #         bpy.context.view_layer.objects.active = bpy.data.objects[0]

    #     init_mode = bpy.context.object.mode
    #     bpy.ops.object.mode_set(mode='OBJECT')

    #     init_active_object = bpy.context.active_object
    #     init_selected_objects = bpy.context.selected_objects
    #     bpy.ops.object.select_all(action='DESELECT')

    #     # select only visible meshes
    #     for obj in scene.objects:
    #         if obj.type == 'MESH' and self.is_object_visible(obj):
    #             obj.select_set(True)

    #     # save ply file
    #     bpy.ops.wm.ply_export(filepath=os.path.join(directory, 'points3d.ply'), export_normals=True, export_attributes=False, ascii_format=True)

    #     # remove temporary vertex colors
    #     for obj in scene.objects:
    #         if obj.type == 'MESH' and self.is_object_visible(obj):
    #             if obj.data.vertex_colors:
    #                 obj.data.vertex_colors.remove(obj.data.vertex_colors[TMP_VERTEX_COLORS])

    #     bpy.context.view_layer.objects.active = init_active_object
    #     bpy.ops.object.select_all(action='DESELECT')

    #     # reselect previously selected objects
    #     for obj in init_selected_objects:
    #         obj.select_set(True)

    #     bpy.ops.object.mode_set(mode=init_mode)
    def save_splats_ply(self, scene, directory):
        # Create temporary vertex colors
        for obj in scene.objects:
            if obj.type == 'MESH':
                if not obj.data.vertex_colors:
                    obj.data.vertex_colors.new(name=TMP_VERTEX_COLORS)

        if bpy.context.object is None:
            self.report({'INFO'}, 'No object active. Setting first object as active.')
            bpy.context.view_layer.objects.active = bpy.data.objects[0]

        # Store the initial mode to restore later
        init_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Store the initial active and selected objects to restore later
        init_active_object = bpy.context.active_object
        init_selected_objects = scene.objects.copy()
        bpy.ops.object.select_all(action='DESELECT')

        # Prepare a new collection to store temporary meshes
        temp_collection = bpy.data.collections.new("Temp_PLY_Export")
        scene.collection.children.link(temp_collection)

        # Iterate over all objects to process visible meshes
        for obj in scene.objects:
            if obj.type == 'MESH' and self.is_object_visible(obj):
                # Duplicate the object to avoid modifying the original
                temp_obj = obj.copy()
                temp_obj.data = obj.data.copy()
                temp_collection.objects.link(temp_obj)

                # Enter Edit Mode for the temporary object
                bpy.context.view_layer.objects.active = temp_obj
                bpy.ops.object.mode_set(mode='EDIT')

                # Initialize BMesh
                bm = bmesh.from_edit_mesh(temp_obj.data)

                total_vertices = len(bm.verts)
                step = max(1, total_vertices // 20)

                # Deselect all vertices first
                for vert in bm.verts:
                    vert.select = False

                # Select every 'step' vertex to get approximately 1/20 of the vertices
                for i, vert in enumerate(bm.verts):
                    if i % step == 0:
                        vert.select = True

                # Update the mesh with the selected vertices
                bmesh.update_edit_mesh(temp_obj.data)
                bpy.ops.object.mode_set(mode='OBJECT')

        # Now export the temporary collection's objects to PLY
        # Note: Blender's built-in PLY exporter does not support exporting multiple objects directly.
        # Therefore, we'll join all temporary objects into one mesh before exporting.

        # Select all temporary objects
        bpy.ops.object.select_all(action='DESELECT')
        for temp_obj in temp_collection.objects:
            temp_obj.select_set(True)

        # Join selected temporary objects into a single mesh
        bpy.context.view_layer.objects.active = temp_collection.objects[0]
        bpy.ops.object.join()

        # Export the joined mesh to PLY
        ply_filepath = os.path.join(directory, 'points3d.ply')
        bpy.ops.wm.ply_export(
            filepath=ply_filepath,
            use_selection=True,
            use_normals=True,
            use_attributes=False,
            use_ascii=True
        )

        # Clean up: Remove the temporary collection and its objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in temp_collection.objects:
            obj.select_set(True)
        bpy.ops.object.delete()

        bpy.data.collections.remove(temp_collection)

        # Remove temporary vertex colors
        for obj in scene.objects:
            if obj.type == 'MESH':
                if TMP_VERTEX_COLORS in obj.data.vertex_colors:
                    obj.data.vertex_colors.remove(obj.data.vertex_colors[TMP_VERTEX_COLORS])

        # Restore the initial active object and selection
        bpy.context.view_layer.objects.active = init_active_object
        bpy.ops.object.select_all(action='DESELECT')

        for obj in init_selected_objects:
            obj.select_set(True)

        # Restore the initial mode
        bpy.ops.object.mode_set(mode=init_mode)





    def save_json(self, directory, filename, data, indent=4):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=indent)

    def is_power_of_two(self, x):
        return math.log2(x).is_integer()

    # function from original nerf 360_view.py code for blender
    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    # check whether an object is visible in render
    def is_object_visible(self, obj):
        if obj.hide_render:
            return False

        for collection in obj.users_collection:
            if collection.hide_render:
                return False

        return True

    # assert messages
    def asserts(self, scene, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        camera = scene.camera
        train_camera = scene.camera_train_target
        test_camera = scene.camera_test_target

        sof_name = scene.sof_dataset_name
        ttc_name = scene.ttc_dataset_name
        cos_name = scene.cos_dataset_name

        error_messages = []

        if (method == 'SOF' or method == 'COS') and not camera.data.type == 'PERSP':
            error_messages.append('Only perspective cameras are supported!')

        if method == 'TTC' and not (train_camera.data.type == 'PERSP' and test_camera.data.type == 'PERSP'):
           error_messages.append('Only perspective cameras are supported!')

        if method == 'COS' and CAMERA_NAME in scene.objects.keys():
            sphere_camera = scene.objects[CAMERA_NAME]
            if not sphere_camera.data.type == 'PERSP':
                error_messages.append('BlenderNeRF Camera must remain a perspective camera!')

        if (method == 'SOF' and sof_name == '') or (method == 'TTC' and ttc_name == '') or (method == 'COS' and cos_name == ''):
            error_messages.append('Dataset name cannot be empty!')

        if method == 'COS' and any(x == 0 for x in scene.sphere_scale):
            error_messages.append('The BlenderNeRF Sphere cannot be flat! Change its scale to be non zero in all axes.')

        if not scene.nerf and not self.is_power_of_two(scene.aabb):
            error_messages.append('AABB scale needs to be a power of two!')

        if scene.save_path == '':
            error_messages.append('Save path cannot be empty!')

        if scene.splats and not scene.test_data:
            error_messages.append('Gaussian Splatting requires test data!')

        if scene.splats and scene.render.image_settings.file_format != 'PNG':
            error_messages.append('Gaussian Splatting requires PNG file extensions!')

        return error_messages

    def save_log_file(self, scene, directory, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'
        now = datetime.datetime.now()

        logdata = {
            'BlenderNeRF Version': scene.blendernerf_version,
            'Date and Time' : now.strftime("%d/%m/%Y %H:%M:%S"),
            'Train': scene.train_data,
            'Test': scene.test_data,
            'AABB': scene.aabb,
            'Render Frames': scene.render_frames,
            'File Format': 'NeRF' if scene.nerf else 'NGP',
            'Save Path': scene.save_path,
            'Method': method
        }

        if method == 'SOF':
            logdata['Frame Step'] = scene.train_frame_steps
            logdata['Camera'] = scene.camera.name
            logdata['Dataset Name'] = scene.sof_dataset_name

        elif method == 'TTC':
            logdata['Train Camera Name'] = scene.camera_train_target.name
            logdata['Test Camera Name'] = scene.camera_test_target.name
            logdata['Frames'] = scene.ttc_nb_frames
            logdata['Dataset Name'] = scene.ttc_dataset_name

        else:
            logdata['Camera'] = scene.camera.name
            logdata['Location'] = str(list(scene.sphere_location))
            logdata['Rotation'] = str(list(scene.sphere_rotation))
            logdata['Scale'] = str(list(scene.sphere_scale))
            logdata['Radius'] = scene.sphere_radius
            logdata['Lens'] = str(scene.focal) + ' mm'
            logdata['Seed'] = scene.seed
            logdata['Frames'] = scene.cos_nb_frames
            logdata['Upper Views'] = scene.upper_views
            logdata['Outwards'] = scene.outwards
            logdata['Dataset Name'] = scene.cos_dataset_name

        self.save_json(directory, filename='log.txt', data=logdata)