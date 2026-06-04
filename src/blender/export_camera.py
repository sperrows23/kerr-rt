import bpy
import json
import math

scene = bpy.context.scene
cam_obj = scene.camera

if cam_obj is None:
    raise RuntimeError("No active camera in the scene.")

cam = cam_obj.data
render = scene.render
frame_start = scene.frame_start
frame_end = scene.frame_end

# Effective render aspect (pixels × pixel-aspect). Blender's ``cam.angle`` is the
# FOV along the *sensor-fit* axis (horizontal for a landscape sensor under the
# default AUTO fit), but the Taichi renderer consumes "fov" as the VERTICAL field
# of view — it reconstructs the horizontal half-angle from this aspect
# (tan_half_x = tan_half_y · width/height). So derive the vertical FOV explicitly
# here, regardless of sensor fit, instead of exporting the raw larger-axis angle. (A1)
res_x = render.resolution_x * render.pixel_aspect_x
res_y = render.resolution_y * render.pixel_aspect_y
if cam.sensor_fit == "VERTICAL":
    fit_vertical = True
elif cam.sensor_fit == "HORIZONTAL":
    fit_vertical = False
else:  # 'AUTO': the longer pixel axis is the fit axis
    fit_vertical = res_y > res_x


def vertical_fov():
    """cam.angle is along the fit axis; convert to the vertical FOV the renderer wants."""
    if fit_vertical:
        return cam.angle
    return 2.0 * math.atan(math.tan(0.5 * cam.angle) * res_y / res_x)


frames = []
for frame in range(frame_start, frame_end + 1):
    scene.frame_set(frame)

    mat = cam_obj.matrix_world
    rot = mat.to_3x3()

    pos = list(mat.translation)
    right = list(rot.col[0])   # local X → world right
    up = list(rot.col[1])      # local Y → world up
    fwd = [-rot.col[2][0], -rot.col[2][1], -rot.col[2][2]]  # local -Z → forward

    frames.append({
        "frame": frame,
        "pos": pos,
        "fwd": fwd,
        "up": up,
        "right": right,
        "fov": vertical_fov(),  # vertical FOV in radians (derived above; see A1)
    })

filepath = bpy.path.abspath("//camera_matrix.json")
with open(filepath, "w") as f:
    json.dump(frames, f, indent=2)

print(f"Camera data successfully saved: {filepath}")
