"""Close a shell left open by the T-junction of a FixMissingSeam seam vertex.

The M4 heal ``_heal_approximated_face`` lets ``ShapeFix_Face.FixMissingSeam`` add the chart seam of a periodic face
bounded by stored fits. Where the seam meets a boundary loop inside a source edge, the healed face carries that edge
as two pieces, while the neighbouring (partner) face keeps the shared source edge whole. Sewing does not cut that
T-junction and the shell stays open (``geometry.shell_open``; medium-block 4B70 face $8, edge $65, face $33).

The seam of a cylinder or cone is a ruling at a fixed chart parameter, so it cannot in general be moved to meet every
boundary loop at an existing vertex. Instead the partner edge is split at the seam vertex, on the partner face only:
both pieces keep the edge's own curve, pcurves, parameterisation (sub-ranges of a same-parameter edge) and tolerance;
the new vertex lies on the curve and carries that same edge tolerance. No tolerance is raised, nothing is re-fitted,
and no ShapeFix / SameParameter call runs.

Split-point bound (R206): a seam vertex is split into a partner edge only when its distance to that edge's curve is at
most the seam vertex's OWN tolerance (``BRep_Tool.Tolerance`` of that vertex; 9.44e-5 mm on 4B70 against a gap of
1.008e-6 mm), never the partner edge's (possibly far larger) tolerance. An edge the vertex reaches only through the
edge's tolerance refuses the whole attempt: "seam split refused: gap G mm > vertex tolerance T mm".

Where the final gate reads it (no new API): ``converter.seam_junction_splits`` (one dict per split: shell, face,
partner_face, point, parameter, gap_mm, vertex_tolerance_mm, partner_edge_tolerance_mm, method) and
``converter.seam_junction_refusals`` (one dict per refused attempt: shell, reason, plus face, partner_face, gap_mm,
vertex_tolerance_mm, bound when the vertex-tolerance bound refused). Both are public lists like
``approximated_face_repairs``; the census reads them from the converter after ``convert()`` (also after a refused
conversion, where the refusal is the record of why the shell stayed open).

Fallback only: ``_shell`` calls this solely after the existing path refused the sewn shell with
``geometry.shell_open``, so every shell the existing path closes is never touched. Fail closed: the re-sewn shell has
to pass the existing shell checks (validity, closure, orientation) and have a positive volume; on anything else, or
any OCC error, the converter's ``shell_closure_checks`` are cut back to their state at the original refusal (the only
converter state the attempt writes besides the two records), a refusal is recorded in ``seam_junction_refusals`` and
the caller re-raises today's error unchanged.
"""

from __future__ import annotations

import math

METHOD = "partner edge split at the FixMissingSeam seam vertex"
HEAL_METHOD = "ShapeFix_Face.FixMissingSeam"
# A consistency check of the split partner face, not a modelling tolerance: the same surface and boundary curves must
# enclose the same area, both integrated adaptively to AREA_EPS (relative error estimate).
AREA_EPS = 1e-9
AREA_AGREEMENT = 1e-6


class _Refused(Exception):
    """A qualification that does not hold; the reason (and any structured facts) is recorded."""

    def __init__(self, reason, **facts):
        super().__init__(reason)
        self.facts = facts


def split_seam_junctions(converter, shell, faces, built, checks):
    """Return the re-sewn, checked components of ``shell`` or None (refusal recorded, state restored).

    ``faces`` are the shell's face entities and ``built`` the converter's faces in the same order (as sewn by the
    existing path); ``checks`` is ``len(converter.shell_closure_checks)`` before the existing path's shell checks.
    """
    failed = len(converter.shell_closure_checks)
    try:
        components, splits = _attempt(converter, shell, faces, built)
    except Exception as error:  # _Refused, CadQueryConversionError, OCC Standard_Failure: all refuse
        del converter.shell_closure_checks[failed:]
        reason = str(error) if isinstance(error, _Refused) else f"{type(error).__name__}: {error}"
        facts = error.facts if isinstance(error, _Refused) else {}
        converter.seam_junction_refusals.append({"shell": shell.index, "reason": reason, **facts})
        return None
    accepted = converter.shell_closure_checks[failed:]
    del converter.shell_closure_checks[checks:]  # the refused attempt's records go; the accepted shell's stay
    converter.shell_closure_checks.extend(accepted)
    for split in splits:
        converter.seam_junction_splits.append({"shell": shell.index, **split, "method": METHOD})
    return components


def _attempt(converter, shell, faces, built):
    healed = {repair.get("face") for repair in converter.approximated_face_repairs
              if str(repair.get("method", "")).startswith(HEAL_METHOD)}
    positions = [k for k, face in enumerate(faces) if face.index in healed]
    if not positions:
        raise _Refused("no face of the shell carries a FixMissingSeam heal")
    lump = converter.model.resolve(shell.lump)
    if getattr(getattr(lump, "shell", None), "index", None) != shell.index:
        raise _Refused("the shell is not the outer shell of its lump")
    if any(not face.wrapped.Location().IsIdentity() for face in built):
        raise _Refused("a face of the shell is located")

    from OCP.BRep import BRep_Tool
    splits = {}
    records = []
    for k in positions:
        for vertex in _seam_vertices(built[k].wrapped):
            hits = _partner_hits(built, k, vertex)
            tolerance = BRep_Tool.Tolerance_s(vertex)
            for j, edge, parameter, gap in hits:
                if gap > tolerance:  # reached only through the partner edge's tolerance (R206): refuse
                    raise _Refused(f"seam split refused: gap {gap:.6g} mm > vertex tolerance {tolerance:.6g} mm",
                                   face=faces[k].index, partner_face=faces[j].index, gap_mm=gap,
                                   vertex_tolerance_mm=tolerance, bound="seam vertex tolerance")
            if len(hits) > 1:
                raise _Refused(f"ambiguous partner edges for a seam vertex of face ${faces[k].index}")
            if not hits:
                continue  # the seam meets this loop at an existing vertex
            j, edge, parameter, gap = hits[0]
            if any(edge.IsSame(other) for other, _ in splits.get(j, [])):
                raise _Refused(f"two seam vertices inside one edge of face ${faces[j].index}")
            splits.setdefault(j, []).append((edge, parameter))
            point = BRep_Tool.Pnt_s(vertex)
            records.append({"face": faces[k].index, "partner_face": faces[j].index,
                            "point": (point.X(), point.Y(), point.Z()), "parameter": parameter, "gap_mm": gap,
                            "vertex_tolerance_mm": tolerance, "partner_edge_tolerance_mm": BRep_Tool.Tolerance_s(edge)})
    if not splits:
        raise _Refused("no seam vertex lies inside a partner edge")
    rebuilt = list(built)
    for j, items in splits.items():
        rebuilt[j] = _split_face(converter, built[j], items)
    result = converter.cq.Shell.makeShell(rebuilt)
    components = converter._shell_components(result, shell, faces)  # the existing path's checks; raises on failure
    for index, component in enumerate(components):
        volume = _volume(component)
        if not math.isfinite(volume) or volume <= 0:
            raise _Refused(f"component {index} volume {volume!r} is not positive")
    return components, records


def _edges(shape):
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape
    found = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, found)
    return [TopoDS.Edge_s(found.FindKey(i)) for i in range(1, found.Extent() + 1)]


def _seam_vertices(face):
    """The vertices of the healed face's chart seam (edges closed on the face)."""
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp
    out = []
    for edge in _edges(face):
        if not BRep_Tool.IsClosed_s(edge, face):
            continue
        for vertex in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)):
            if not vertex.IsNull() and not any(vertex.IsSame(known) for known in out):
                out.append(vertex)
    return out


def _partner_hits(built, k, vertex):
    """(face position, edge, curve parameter, gap) for every edge of another face holding ``vertex`` strictly inside
    within the wider of the vertex / edge tolerances (the candidates; the caller applies the vertex-tolerance bound)."""
    from OCP.BRep import BRep_Tool
    hits = []
    for j, face in enumerate(built):
        if j == k:
            continue
        for edge in _edges(face.wrapped):
            if BRep_Tool.Degenerated_s(edge) or BRep_Tool.IsClosed_s(edge, face.wrapped):
                continue
            found = _interior_parameter(edge, vertex)
            if found is not None:
                hits.append((j, edge, *found))
    return hits


def _interior_parameter(edge, vertex):
    """(curve parameter, distance of ``vertex`` to the curve) when ``vertex`` lies on ``edge`` within the wider of the
    existing vertex / edge tolerances and clear of both end vertices; else None. The split-point bound itself (the
    vertex's own tolerance) is applied by ``_attempt``, so that a refusal by it can be recorded."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCP.TopExp import TopExp
    adaptor = BRepAdaptor_Curve(edge)
    if not adaptor.Is3DCurve():
        return None
    point = BRep_Tool.Pnt_s(vertex)
    limit = max(BRep_Tool.Tolerance_s(vertex), BRep_Tool.Tolerance_s(edge))
    for end in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)):
        if end.IsNull() or point.Distance(BRep_Tool.Pnt_s(end)) <= max(limit, BRep_Tool.Tolerance_s(end)):
            return None
    local = point.Transformed(adaptor.Trsf().Inverted())
    first, last = adaptor.FirstParameter(), adaptor.LastParameter()
    projection = GeomAPI_ProjectPointOnCurve(local, adaptor.Curve().Curve(), first, last)
    if projection.NbPoints() == 0 or projection.LowerDistance() > limit:
        return None
    parameter = projection.LowerDistanceParameter()
    if not first < parameter < last:
        return None
    return parameter, projection.LowerDistance()


def _split_face(converter, face, items):
    """``face`` with each (edge, parameter) split into two edges on sub-ranges of the edge's own representations."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_ReShape
    from OCP.ShapeBuild import ShapeBuild_Edge
    from OCP.TopAbs import TopAbs_FORWARD
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS, TopoDS_Vertex, TopoDS_Wire
    builder = BRep_Builder()
    reshape = BRepTools_ReShape()
    copier = ShapeBuild_Edge()
    for edge, parameter in items:
        if not edge.Location().IsIdentity():
            raise _Refused("the partner edge is located")
        if not (BRep_Tool.SameParameter_s(edge) and BRep_Tool.SameRange_s(edge)):
            raise _Refused("the partner edge is not same-parameter / same-range")
        forward = TopoDS.Edge_s(edge.Oriented(TopAbs_FORWARD))
        adaptor = BRepAdaptor_Curve(forward)
        first, last = adaptor.FirstParameter(), adaptor.LastParameter()
        tolerance = BRep_Tool.Tolerance_s(forward)
        middle = TopoDS_Vertex()
        builder.MakeVertex(middle, adaptor.Value(parameter), tolerance)
        lower = copier.CopyReplaceVertices(forward, TopExp.FirstVertex_s(forward), middle)
        builder.Range(lower, first, parameter)
        upper = copier.CopyReplaceVertices(forward, middle, TopExp.LastVertex_s(forward))
        builder.Range(upper, parameter, last)
        wire = TopoDS_Wire()
        builder.MakeWire(wire)
        builder.Add(wire, lower)
        builder.Add(wire, upper)
        reshape.Replace(forward, wire)
    result = converter.cq.Face(TopoDS.Face_s(reshape.Apply(face.wrapped)))
    if len(_edges(result.wrapped)) != len(_edges(face.wrapped)) + len(items):
        raise _Refused("the split partner face does not carry the split edges")
    if not result.isValid():
        raise _Refused("the split partner face is invalid")
    before, after = _area(face.wrapped), _area(result.wrapped)
    if not (math.isfinite(after) and after > 0 and abs(after - before) <= AREA_AGREEMENT * abs(before)):
        raise _Refused(f"the split partner face area {after!r} differs from {before!r}")
    return result


def _area(face):
    """Adaptive surface integration (the fixed-order default depends on how the boundary is cut into edges)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, properties, AREA_EPS)
    return properties.Mass()


def _volume(component):
    """Signed volume enclosed by the shell as oriented (no orientation fix)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS_Solid
    builder = BRep_Builder()
    solid = TopoDS_Solid()
    builder.MakeSolid(solid)
    builder.Add(solid, component.wrapped)
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, properties)
    return properties.Mass()
