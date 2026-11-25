# SPDX-FileCopyrightText: 2026 varomix
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Compare the "Bricker" node against reference output: JSON files with input meshes and the
expected divided meshes.

Run with the fork build:
  blender -b --factory-startup --python verify_bricker.py -- <ref_dir> <result.json>
"""
import json
import math
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
REF_DIR, OUT_PATH = argv[0], argv[1]
ND = 4


def make_mesh(name, data):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in data["P"]], [], [list(f) for f in data["F"]])
    attrs = data.get("attrs", {})
    if "vertex:uv" in attrs:
        layer = mesh.attributes.new("uv", 'FLOAT2', 'CORNER')
        flat = [uv[:2] for face in attrs["vertex:uv"] for uv in face]
        layer.data.foreach_set("vector", [c for uv in flat for c in uv])
    if "point:Cd" in attrs:
        layer = mesh.attributes.new("Cd", 'FLOAT_VECTOR', 'POINT')
        layer.data.foreach_set("vector", [c for v in attrs["point:Cd"] for c in v])
    if "point:w" in attrs:
        layer = mesh.attributes.new("w", 'FLOAT', 'POINT')
        layer.data.foreach_set("value", attrs["point:w"])
    if "prim:pv" in attrs:
        layer = mesh.attributes.new("pv", 'FLOAT', 'FACE')
        layer.data.foreach_set("value", attrs["prim:pv"])
    groups = data.get("groups", {})
    if groups:
        for gname, prims in groups.items():
            layer = mesh.attributes.new(gname, 'BOOLEAN', 'FACE')
            values = [False] * len(data["F"])
            for i in prims:
                values[i] = True
            layer.data.foreach_set("value", values)
    mesh.update()
    return mesh


def make_tree(size, offset, angle, fix, group):
    tree = bpy.data.node_groups.new("bricker_test", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nin = tree.nodes.new("NodeGroupInput")
    nout = tree.nodes.new("NodeGroupOutput")
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
    return tree, node


def evaluate(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    return ev.to_mesh()


def rkey(v):
    return tuple(round(c + 0.0, ND) + 0.0 for c in v)


def canon_faces(P, F):
    return sorted(tuple(sorted(rkey(P[v]) for v in f)) for f in F)


def run_case(path):
    d = json.load(open(path))
    p = d["parms"]
    size = (p["sizex"], p["sizey"], p["sizez"])
    off = (p["offsetx"], p["offsety"], p["offsetz"])
    ang = (p["anglex"], p["angley"], p["anglez"])
    mesh = make_mesh(d["name"], d["input"])
    obj = bpy.data.objects.new(d["name"], mesh)
    bpy.context.scene.collection.objects.link(obj)
    tree, node = make_tree(size, off, ang, bool(p["fixsharededges"]), p.get("group") or None)
    mod = obj.modifiers.new("bricker", 'NODES')
    mod.node_group = tree
    res = evaluate(obj)
    P = [tuple(v.co) for v in res.vertices]
    F = [list(poly.vertices) for poly in res.polygons]
    hP = d["output"]["P"]
    hF = d["output"]["F"]
    used = set(v for f in F for v in f) | set(range(len(d["input"]["P"])))
    out = {
        "points": [len(used), len(hP)],
        "faces": [len(F), len(hF)],
        "geometry_match": canon_faces(P, F) == canon_faces(hP, hF),
    }
    # Attribute comparison (positions as keys).
    hattrs = d["output"]["attrs"]
    errs = {}
    if "point:Cd" in hattrs:
        h = {rkey(hP[i]): hattrs["point:Cd"][i] for i in range(len(hP))}
        b = res.attributes["Cd"].data
        errs["Cd"] = max(max(abs(a - c) for a, c in zip(b[i].vector, h[rkey(P[i])]))
                         for i in range(len(P)) if rkey(P[i]) in h)
    if "point:w" in hattrs:
        h = {rkey(hP[i]): hattrs["point:w"][i] for i in range(len(hP))}
        b = res.attributes["w"].data
        errs["w"] = max(abs(b[i].value - h[rkey(P[i])]) for i in range(len(P)) if rkey(P[i]) in h)
    if "prim:pv" in hattrs:
        h = {}
        for i, f in enumerate(hF):
            h[tuple(sorted(rkey(hP[v]) for v in f))] = hattrs["prim:pv"][i]
        b = res.attributes["pv"].data
        errs["pv"] = max(abs(b[i].value - h[tuple(sorted(rkey(P[v]) for v in F[i]))])
                         for i in range(len(F)))
    if "vertex:uv" in hattrs:
        h = {}
        for i, f in enumerate(hF):
            fk = tuple(sorted(rkey(hP[v]) for v in f))
            for j, v in enumerate(f):
                h[(fk, rkey(hP[v]))] = hattrs["vertex:uv"][i][j][:2]
        b = res.attributes["uv"].data
        e = 0.0
        for poly in res.polygons:
            fk = tuple(sorted(rkey(P[v]) for v in poly.vertices))
            for li, v in zip(poly.loop_indices, poly.vertices):
                ref = h.get((fk, rkey(P[v])))
                if ref is not None:
                    e = max(e, abs(b[li].vector[0] - ref[0]), abs(b[li].vector[1] - ref[1]))
        errs["uv"] = e
    out["attr_max_error"] = errs
    out["valid"] = not res.validate(verbose=False)
    return out


def benchmark():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=1.0)
    obj = bpy.context.active_object
    tree, node = make_tree((0.02, 0.02, 0.02), (0, 0, 0), (0, 0, 0), True, None)
    mod = obj.modifiers.new("bricker", 'NODES')
    mod.node_group = tree
    times = []
    for i in range(4):
        node.inputs["Offset"].default_value = (0.001 * i, 0, 0)
        t0 = time.perf_counter()
        res = evaluate(obj)
        times.append(time.perf_counter() - t0)
    return {
        "input_faces": len(obj.data.polygons),
        "output_faces": len(res.polygons),
        "output_verts": len(res.vertices),
        "seconds": times,
    }


results = {"version": bpy.app.version_string, "cases": {}}
for name in sorted(os.listdir(REF_DIR)):
    if not name.endswith(".json") or "probe" in name:
        continue
    try:
        results["cases"][name[:-5]] = run_case(os.path.join(REF_DIR, name))
    except Exception as ex:
        import traceback
        results["cases"][name[:-5]] = {"error": traceback.format_exc()}
try:
    results["benchmark"] = benchmark()
except Exception:
    import traceback
    results["benchmark"] = {"error": traceback.format_exc()}
json.dump(results, open(OUT_PATH, "w"), indent=1)
print("WROTE", OUT_PATH)
