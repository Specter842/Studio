"""Animated 3D text on a transparent background. Runs inside Blender.

    blender --background --factory-startup --python text_reveal.py -- args.json

This file is executed by Blender's own bundled Python, not the project venv:
`bpy` exists only in there, and nothing in this file may import from the rest
of the project. It is a standalone script that happens to live in the repo.

Renders an RGBA PNG sequence. Not video: Blender's video writers drop the alpha
channel in most container/codec combinations, and the compositor wants alpha.

Fades are deliberately *not* done here. Animating material alpha needs blend
mode changes that differ across Blender versions and render engines, whereas
ffmpeg's fade filter is one line and behaves identically everywhere. Blender
does the motion; the compositor does the fade.
"""

import json
import sys
from math import radians

import bpy

# EEVEE Next in 4.2+, plain EEVEE before that, Cycles as a last resort. Trying
# in order keeps one script working across the Blender versions people have.
ENGINES = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES")

STYLES = {
    #          metallic, roughness, emission strength
    "metal":   (0.85, 0.22, 0.0),
    "matte":   (0.00, 0.65, 0.0),
    # Emission is kept modest: anything much above this clips to flat white
    # once it is composited over footage, losing the bevel and the 3D read.
    "neon":    (0.00, 0.35, 2.2),
}


def read_arguments():
    """Everything after `--` is ours; Blender consumes the rest."""
    if "--" not in sys.argv:
        raise SystemExit("text_reveal.py: expected `-- <args.json>`")
    args_path = sys.argv[sys.argv.index("--") + 1]
    with open(args_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_world(scene):
    """Give the scene an environment to reflect.

    `read_factory_settings(use_empty=True)` leaves no world at all, and a
    metallic surface has no diffuse component — it is *only* reflection. With
    nothing to reflect it renders pure black, which looks exactly like a
    material that failed to apply. `film_transparent` still keeps the
    background out of the render, so this lights the text without appearing.
    """
    world = bpy.data.worlds.new("ElementWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.62, 0.64, 0.68, 1.0)
    background.inputs["Strength"].default_value = 1.0


def set_engine(scene):
    for engine in ENGINES:
        try:
            scene.render.engine = engine
            return engine
        except TypeError:
            continue
    raise SystemExit(f"None of {ENGINES} are available in this Blender build.")


def build_text(body, color, style):
    bpy.ops.object.text_add(location=(0.0, 0.0, 0.0))
    text_object = bpy.context.object
    data = text_object.data

    data.body = body
    data.align_x = "CENTER"
    data.align_y = "CENTER"
    # Extrude and bevel are what make it read as 3D rather than flat type.
    data.extrude = 0.05
    data.bevel_depth = 0.008
    data.bevel_resolution = 2

    metallic, roughness, emission = STYLES.get(style, STYLES["metal"])
    material = bpy.data.materials.new("Element")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if emission:
        # Input names moved between 3.x and 4.x; set whichever exists.
        for name in ("Emission Color", "Emission"):
            if name in shader.inputs:
                shader.inputs[name].default_value = (*color, 1.0)
                break
        if "Emission Strength" in shader.inputs:
            shader.inputs["Emission Strength"].default_value = emission
    data.materials.append(material)

    return text_object


def frame_camera(scene, text_object):
    """An orthographic camera scaled to whatever the text turned out to be.

    Blender builds text lying in the XY plane facing +Z, so the camera has to
    look straight down -Z to see the face of it. Put the camera on -Y instead
    and you photograph the letters edge-on: the render comes out as a row of
    flat bars, which is the extrusion depth and nothing else.

    Sizing the camera to the measured object means the caller never has to pick
    a font size, and long titles cannot run off the sides.
    """
    bpy.context.view_layer.update()
    width = max(text_object.dimensions.x, 0.1)
    height = max(text_object.dimensions.y, 0.1)

    # Default camera orientation already looks along -Z.
    bpy.ops.object.camera_add(location=(0.0, 0.0, 6.0), rotation=(0.0, 0.0, 0.0))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    # 1.35 leaves breathing room so the bevel and lighting falloff are not clipped.
    camera.data.ortho_scale = max(width * 1.35, height * 1.35 * aspect, 2.0)
    scene.camera = camera


def add_lights():
    """Both lights live on the camera side (+Z), or the face stays unlit."""
    bpy.ops.object.light_add(type="AREA", location=(3.0, 2.0, 5.0))
    key = bpy.context.object
    key.data.energy = 900.0
    key.data.size = 6.0

    bpy.ops.object.light_add(type="AREA", location=(-4.0, -2.0, 4.0))
    fill = bpy.context.object
    fill.data.energy = 300.0
    fill.data.size = 8.0


def animate(text_object, frames):
    """Rise into place, settling out of a slight backward tilt.

    Movement is in Y because that is vertical on screen for a camera looking
    down -Z; sliding in Z would push the text toward the lens instead.
    """
    reveal = max(2, int(frames * 0.35))

    text_object.location = (0.0, -0.45, 0.0)
    text_object.scale = (0.82, 0.82, 0.82)
    text_object.rotation_euler = (radians(-25.0), 0.0, 0.0)
    text_object.keyframe_insert("location", frame=1)
    text_object.keyframe_insert("scale", frame=1)
    text_object.keyframe_insert("rotation_euler", frame=1)

    text_object.location = (0.0, 0.0, 0.0)
    text_object.scale = (1.0, 1.0, 1.0)
    text_object.rotation_euler = (0.0, 0.0, 0.0)
    text_object.keyframe_insert("location", frame=reveal)
    text_object.keyframe_insert("scale", frame=reveal)
    text_object.keyframe_insert("rotation_euler", frame=reveal)

    # Ease out, so it decelerates into place instead of arriving linearly.
    for curve in text_object.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "CUBIC"
            point.easing = "EASE_OUT"


def configure_render(scene, settings, frames):
    render = scene.render
    render.resolution_x = int(settings["width"])
    render.resolution_y = int(settings["height"])
    render.resolution_percentage = 100
    render.fps = max(1, int(round(settings["fps"])))

    # The whole point: everything the text does not cover must be transparent.
    render.film_transparent = True
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.compression = 15

    scene.frame_start = 1
    scene.frame_end = frames
    render.filepath = settings["output_prefix"]

    # Modest sampling: these are short overlays, not hero renders, and EEVEE at
    # 16 samples is visually indistinguishable here at a fraction of the time.
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16


def main():
    settings = read_arguments()
    frames = max(1, round(float(settings["seconds"]) * float(settings["fps"])))

    reset_scene()
    scene = bpy.context.scene
    engine = set_engine(scene)
    add_world(scene)

    configure_render(scene, settings, frames)
    text_object = build_text(
        settings["text"],
        tuple(settings.get("color", [1.0, 1.0, 1.0])),
        settings.get("style", "metal"),
    )
    add_lights()
    frame_camera(scene, text_object)
    animate(text_object, frames)

    print(f"[text_reveal] engine={engine} frames={frames} "
          f"size={scene.render.resolution_x}x{scene.render.resolution_y}")
    bpy.ops.render.render(animation=True)
    print("[text_reveal] done")


if __name__ == "__main__":
    main()
