# SPDX-FileCopyrightText: 2026 varomix
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Regression test for the Bricker node: builds input meshes, runs Bricker, and compares the result
with the expected output in `bricker_golden.json` (topology exactly, positions and attributes
within a small tolerance, so it holds across compilers and platforms).

Run with the fork build (exit code 1 on a mismatch):
  blender -b --factory-startup --python bricker_golden.py
After an intended change of Bricker's output, regenerate the expected output:
  blender -b --factory-startup --python bricker_golden.py -- --update
"""
import json
import math
import sys
from pathlib import Path

import bpy

GOLDEN = Path(__file__).resolve().with_name("bricker_golden.json")
TOLERANCE = 1e-5


def add_attributes(mesh):
    """A point float, a face float and the UV map (corners), to check interpolation."""
    w = mesh.attributes.new("w", 'FLOAT', 'POINT')
    w.data.foreach_set("value", [v.co.x + 2.0 * v.co.y - 0.5 * v.co.z for v in mesh.vertices])
    pv = mesh.attributes.new("pv", 'FLOAT', 'FACE')
    pv.data.foreach_set("value", [float(i) * 0.25 for i in range(len(mesh.polygons))])
    if not mesh.uv_layers:
        uv = mesh.uv_layers.new(name="UVMap")
        for loop in mesh.loops:
            co = mesh.vertices[loop.vertex_index].co
            uv.data[loop.index].uv = (co.x * 0.5 + 0.5, co.y * 0.5 + 0.5)


def mesh_from_data(name, verts, faces):
    """A mesh with every edge given explicitly, in first-appearance order. Bricker's output order
    follows the input's edge order, and Blender's own edge calculation (and the primitive
    operators) don't guarantee a stable order between runs."""
    edges, seen = [], set()
    for face in faces:
        for k in range(len(face)):
            e = (face[k], face[(k + 1) % len(face)])
            if tuple(sorted(e)) not in seen:
                seen.add(tuple(sorted(e)))
                edges.append(e)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def grid(nx, ny, size=2.0):
    verts = [(size * (i / nx - 0.5), size * (j / ny - 0.5), 0.0)
             for j in range(ny + 1) for i in range(nx + 1)]  # fmt: skip
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1,
              (j + 1) * (nx + 1) + i) for j in range(ny) for i in range(nx)]  # fmt: skip
    return lambda: mesh_from_data("grid", verts, faces)


def cube():
    verts = [(x, y, z) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return lambda: mesh_from_data("cube", verts, faces)


def circle(n):
    verts = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n), 0.0) for k in range(n)]
    return lambda: mesh_from_data("circle", verts, [tuple(range(n))])


def uv_sphere(segments, rings):
    verts = [(0.0, 0.0, 1.0)]
    for r in range(1, rings):
        theta = math.pi * r / rings
        for s in range(segments):
            phi = 2 * math.pi * s / segments
            verts.append((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                          math.cos(theta)))  # fmt: skip
    verts.append((0.0, 0.0, -1.0))
    bottom = len(verts) - 1
    ring = lambda r, s: 1 + (r - 1) * segments + s % segments  # noqa: E731
    faces = [(0, ring(1, s), ring(1, s + 1)) for s in range(segments)]
    for r in range(1, rings - 1):
        faces += [(ring(r, s), ring(r + 1, s), ring(r + 1, s + 1), ring(r, s + 1))
                  for s in range(segments)]  # fmt: skip
    faces += [(ring(rings - 1, s + 1), ring(rings - 1, s), bottom) for s in range(segments)]
    return lambda: mesh_from_data("sphere", verts, faces)


def select_faces_x_positive(obj):
    mesh = obj.data
    sel = mesh.attributes.new("sel", 'BOOLEAN', 'FACE')
    sel.data.foreach_set("value", [p.center.x > 0.0 for p in mesh.polygons])
    return "sel"


CASES = [
    # name, builder, size, offset, angle (degrees), fix shared edges, selection
    ("grid_unit", grid(2, 2),
     (0.5, 0.5, 0.5), (0, 0, 0), (0, 0, 0), True, None),
    ("grid_offset", grid(5, 5),
     (0.3, 0.45, 1.0), (0.11, -0.07, 0.0), (0, 0, 0), True, None),
    ("grid_angle", grid(2, 2),
     (0.4, 0.4, 0.4), (0, 0, 0), (20, 35, 0), True, None),
    ("cube", cube(),
     (0.6, 0.6, 0.6), (0.05, 0.1, 0.15), (0, 0, 0), True, None),
    ("circle_ngon", circle(12),
     (0.35, 0.35, 0.35), (0, 0, 0), (0, 0, 0), True, None),
    ("lshape_concave", lambda: mesh_from_data(
        "lshape", [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)],
        [(0, 1, 2, 3, 4, 5)]),
     (0.45, 0.45, 0.45), (0.1, 0.1, 0), (0, 0, 0), True, None),
    ("tilted_quad", lambda: mesh_from_data(
        "tilted", [(0, 0, 0), (1.5, 0.2, 0.4), (1.3, 1.4, 1.0), (-0.2, 1.1, 0.6)],
        [(0, 1, 2, 3)]),
     (0.3, 0.3, 0.3), (0, 0, 0), (15, 25, 40), True, None),
    ("sphere_closed", uv_sphere(12, 6),
     (0.5, 0.5, 0.5), (0.02, 0.03, 0.04), (0, 0, 0), True, None),
    ("selection_fix", grid(3, 3),
     (0.3, 0.3, 0.3), (0, 0, 0), (0, 0, 0), True, select_faces_x_positive),
    ("selection_nofix", grid(3, 3),
     (0.3, 0.3, 0.3), (0, 0, 0), (0, 0, 0), False, select_faces_x_positive),
]  # fmt: skip


def run_case(name, builder, size, offset, angle, fix, selection):
    obj = builder()
    obj.name = name
    add_attributes(obj.data)
    group = selection(obj) if selection else None

    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nin, nout = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    node = tree.nodes.new("GeometryNodeBricker")
    node.inputs["Size"].default_value = size
    node.inputs["Offset"].default_value = offset
    node.inputs["Angle"].default_value = [math.radians(a) for a in angle]
    node.inputs["Fix Shared Edges"].default_value = fix
    tree.links.new(nin.outputs[0], node.inputs["Mesh"])
    tree.links.new(node.outputs["Mesh"], nout.inputs[0])
    if group:
        named = tree.nodes.new("GeometryNodeInputNamedAttribute")
        named.data_type = 'BOOLEAN'
        named.inputs["Name"].default_value = group
        tree.links.new(named.outputs["Attribute"], node.inputs["Selection"])
    obj.modifiers.new("bricker", 'NODES').node_group = tree

    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    result = {
        "positions": [list(v.co) for v in mesh.vertices],
        "faces": [list(p.vertices) for p in mesh.polygons],
        "edges": len(mesh.edges),
        "w": [d.value for d in mesh.attributes["w"].data],
        "pv": [d.value for d in mesh.attributes["pv"].data],
        "uv": [list(d.uv) for d in mesh.uv_layers.active.data],
        "valid": not mesh.validate(verbose=False),
    }
    return result


def max_difference(a, b):
    worst = 0.0
    for x, y in zip(a, b):
        if isinstance(x, list):
            worst = max(worst, max_difference(x, y))
        else:
            worst = max(worst, abs(x - y))
    return worst


def compare(name, got, want):
    problems = []
    for key in ("faces", "edges", "valid"):
        if got[key] != want[key]:
            problems.append(f"{key} differ")
    for key in ("positions", "w", "pv", "uv"):
        if len(got[key]) != len(want[key]):
            problems.append(f"{key}: {len(got[key])} values, expected {len(want[key])}")
        elif (diff := max_difference(got[key], want[key])) > TOLERANCE:
            problems.append(f"{key} differ by up to {diff:.3g}")
    status = "ok" if not problems else "FAIL: " + "; ".join(problems)
    print(f"  {name:16} {len(got['positions']):5} points {len(got['faces']):5} faces  {status}")
    return not problems


def main():
    update = "--update" in sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else False
    if not hasattr(bpy.types, "GeometryNodeBricker"):
        print("Bricker is not loaded (is toast_plugin_bricker built?)")
        sys.exit(1)
    results = {case[0]: run_case(*case) for case in CASES}
    if update:
        GOLDEN.write_text(json.dumps(results, separators=(",", ":")) + "\n")
        print(f"Wrote {GOLDEN} ({len(results)} cases)")
        return
    golden = json.loads(GOLDEN.read_text())
    print(f"Bricker golden test, {len(golden)} cases:")
    ok = set(golden) == set(results)
    if not ok:
        print(f"  cases differ: {sorted(set(golden) ^ set(results))}")
    for name, want in golden.items():
        if name in results:
            ok &= compare(name, results[name], want)
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


main()
