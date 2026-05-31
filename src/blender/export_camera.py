import bpy
import json
import math

scene = bpy.context.scene
cam_obj = scene.camera

if cam_obj is None:
    raise RuntimeError("No active camera in the scene.")

cam = cam_obj.data
frame_start = scene.frame_start
frame_end = scene.frame_end

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
        "fov": cam.angle,      # vertical FOV in radians
    })

filepath = bpy.path.abspath("//camera_matrix.json")
with open(filepath, "w") as f:
    json.dump(frames, f, indent=2)

print(f"Camera data successfully saved: {filepath}")
