import bpy
from mathutils import Vector
from src.macro import *


def set_camera_lookat(camera, target: Vector):
    direction = target - camera.location
    quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = quat.to_euler()


class SimScene:
    def __init__(self, resolution, sun_rx_radian, sun_ry_radian, sun_density, background_color, background_density):
        self.clear_scene()
        self.scene = bpy.context.scene
        self.object_init_location = (0, 0, 0)
        self.gripper_init_location = (0, 0, 0)
        self.object = self.create_object()
        self.gripper = self.create_gripper()
        self.create_floor()
        self.bin_place = self.create_destination()
        self.create_lights(sun_rx_radian, sun_ry_radian, sun_density, background_color, background_density)
        camera_lookat = (0, 0, OBJECT_INIT_Z)
        self.front_view_camera = self.create_camera(camera_name="FrontViewCamera", camera_position=(50, 0, OBJECT_INIT_Z), camera_lookat=camera_lookat)
        self.side_view_camera = self.create_camera(camera_name="SideViewCamera", camera_position=(0, 50, OBJECT_INIT_Z), camera_lookat=camera_lookat)
        self.top_view_camera = self.create_camera(camera_name="TopViewCamera", camera_position=(0, 0, 35), camera_lookat=camera_lookat)
        self.set_rendering(resolution)
        
    def reset(self, object_init_location, gripper_init_location):
        self.object_init_location = object_init_location
        self.gripper_init_location = gripper_init_location
        self.object.location = object_init_location
        self.gripper.location = gripper_init_location
        
    @staticmethod
    def clear_scene():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        # 清理残余数据
        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block, do_unlink=True)
        for block in bpy.data.materials:
            bpy.data.materials.remove(block, do_unlink=True)
        for block in bpy.data.images:
            if block.name != "Render Result":
                bpy.data.images.remove(block, do_unlink=True)

    @staticmethod
    def create_object():
        # ---------------------- 创建蓝色立方体 ----------------------
        bpy.ops.mesh.primitive_cube_add(size=OBJECT_SIZE, location=(0, 0, 0))
        cubic_object = bpy.context.active_object
        cubic_object.name = "BlueCube"

        mat_cubic_object = bpy.data.materials.new(name="BlueMaterial")
        mat_cubic_object.use_nodes = True
        bsdf_cubic_object = mat_cubic_object.node_tree.nodes["Principled BSDF"]
        bsdf_cubic_object.inputs['Base Color'].default_value = (0.0, 0.0, 1.0, 1)
        bsdf_cubic_object.inputs['Roughness'].default_value = 0.4
        bsdf_cubic_object.inputs['Metallic'].default_value = 0.0
        cubic_object.data.materials.append(mat_cubic_object)
        return cubic_object

    @staticmethod
    def create_gripper():
        # ---------------------- 创建红色夹爪 ----------------------
        bpy.ops.mesh.primitive_cube_add(size=OBJECT_SIZE, location=(0, 0, 0))
        gripper = bpy.context.active_object
        gripper.name = "RobotArm"
        gripper.scale = (1, 1, GRIPPER_HEIGHT / OBJECT_SIZE)

        material_gripper = bpy.data.materials.new(name="RedMaterial")
        material_gripper.use_nodes = True
        bsdf_gripper = material_gripper.node_tree.nodes["Principled BSDF"]
        bsdf_gripper.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1)
        bsdf_gripper.inputs['Roughness'].default_value = 0.4
        bsdf_gripper.inputs['Metallic'].default_value = 0.0
        gripper.data.materials.append(material_gripper)
        return gripper

    @staticmethod
    def create_floor():
        # ---------------------- 创建灰色地面 ----------------------
        bpy.ops.mesh.primitive_plane_add(size=FLOOR_SIZE, location=(0, 0, 0))
        floor = bpy.context.active_object
        floor.name = "GrayFloor"
        floor.scale = (1, 1, 1)

        material_floor = bpy.data.materials.new(name="GrayMaterial")
        material_floor.use_nodes = True
        bsdf_floor = material_floor.node_tree.nodes["Principled BSDF"]
        bsdf_floor.inputs['Base Color'].default_value = (0.0, 0.5, 0.0, 1)
        bsdf_floor.inputs['Roughness'].default_value = 0.4
        bsdf_floor.inputs['Metallic'].default_value = 0.0
        floor.data.materials.append(material_floor)
        return floor

    @staticmethod
    def create_destination():
        # ---------------------- 创建物体放置区域 ----------------------
        bpy.ops.mesh.primitive_plane_add(size=1, location=BIN_LOCATION)
        destination = bpy.context.active_object
        destination.name = "BinPlace"
        destination.scale = (BIN_SIZE, BIN_SIZE, 1)

        material_destination = bpy.data.materials.new(name="GreenMaterial")
        material_destination.use_nodes = True
        bsdf_destination = material_destination.node_tree.nodes["Principled BSDF"]
        bsdf_destination.inputs['Base Color'].default_value = (1.0, 1.0, 0.0, 1)
        bsdf_destination.inputs['Roughness'].default_value = 1.0
        bsdf_destination.inputs['Metallic'].default_value = 0.0
        destination.data.materials.append(material_destination)
        return destination

    def create_lights(self, sun_rx_radian, sun_ry_radian, sun_density, bachground_color, bachground_density):
        # ---------------------- 添加光源 ----------------------
        # 1. 强阳光（关键光）
        bpy.ops.object.light_add(type='SUN', location=(1000, -1000, 1000))
        sun = bpy.context.object
        sun.data.energy = sun_density
        sun.data.angle = 0
        sun.rotation_euler = (sun_rx_radian, sun_ry_radian, 0)

        # 2. 环境光（世界背景）
        world = self.scene.world
        world.use_nodes = True
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (*bachground_color, 1)
        bg.inputs[1].default_value = bachground_density  # 强度降低，避免过曝

        # 3. 开启 Cycles 自带的接触阴影（让立方体贴地部分更黑，超级真实）
        self.scene.cycles.use_fast_gi = True
        self.scene.render.film_transparent = False

    def set_rendering(self, resolution):
        self.scene.render.engine = 'CYCLES'

        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        cprefs.refresh_devices()
        cprefs.compute_device_type = 'OPTIX'
        self.scene.cycles.device = 'GPU'

        self.scene.cycles.samples = 64
        self.scene.cycles.denoiser = 'OPTIX'
        self.scene.cycles.use_denoising = True
        self.scene.view_layers["ViewLayer"].cycles.use_denoising = True
        self.scene.cycles.denoiser = 'OPTIX'
        self.scene.cycles.denoising_use_gpu = True

        self.scene.render.resolution_x = resolution
        self.scene.render.resolution_y = resolution
        self.scene.render.image_settings.file_format = 'PNG'

    @staticmethod
    def create_camera(camera_name, camera_position, camera_lookat):
        # ---------------------- 创建相机 ----------------------
        bpy.ops.object.camera_add(location=camera_position)
        camera = bpy.context.active_object
        camera.name = camera_name
        set_camera_lookat(camera, Vector(camera_lookat))
        return camera

    def shot(self, output_path):
        self.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True, animation=False)

    def shot_front_view(self, output_path):
        self.scene.camera = self.front_view_camera
        self.shot(output_path)
        self.scene.camera = None

    def shot_side_view(self, output_path):
        self.scene.camera = self.side_view_camera
        self.shot(output_path)
        self.scene.camera = None

    def shot_top_view(self, output_path):
        self.scene.camera = self.top_view_camera
        self.shot(output_path)
        self.scene.camera = None

    def move_gripper(self, dx, dy, dz):
        self.gripper.location[0] += dx
        self.gripper.location[1] += dy
        self.gripper.location[2] += dz

    def set_gripper(self, x, y, z):
        self.gripper.location[0] = x
        self.gripper.location[1] = y
        self.gripper.location[2] = z

    def move_object(self, dx, dy, dz):
        self.object.location[0] += dx
        self.object.location[1] += dy
        self.object.location[2] += dz

    def gripper_to_pick_location(self):
        target_position = self.object.location.copy()
        target_position[2] = OBJECT_SIZE + 0.5 * GRIPPER_HEIGHT
        return target_position - self.gripper.location

    def gripper_to_place_location(self):
        target_position = self.bin_place.location.copy()
        target_position[2] = OBJECT_SIZE + 0.5 * GRIPPER_HEIGHT
        return target_position - self.gripper.location
