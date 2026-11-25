"""dwg2glb (M4b F5, rulings R200 / R202 / R203): split a shell at non-manifold source edges into body-consistent solids.

simple-stairs 22B8 (5 definitions, 320 occurrences) is one ACIS shell holding two bodies, a rod and a sleeve, that
meet only along one circle: the source edge ``$692``, whose ring has 4 coedges (rod side, rod top, sleeve bore,
sleeve bottom). Sewing all faces at once pairs the 4 coincident edges across the bodies; orienting that shell inverts
the rod, and OCCT certifies a wrong solid (-314,963 mm3).

Here, before today's sewing and only for a shell with a source edge of more than 2 coedges:

* **Partition** (``split_bodies``, combinatorial, no OCC): bodies are the connected face sets over the manifold
  (2-coedge) source edges plus one pairing of every non-manifold ring. A pairing joins the ring's coedges two by two,
  each pair of opposite senses (a manifold pair). It is *body-consistent* when the pairs of every ring end up in
  different bodies. Required: every non-manifold ring has exactly 4 coedges (the parent admits only 4-triangle edges,
  Task 6b); exactly one pairing is body-consistent (two or more = ambiguous); at least two bodies.
* **Per body**: the body's faces are sewn on their own, with the same ``BRepBuilderAPI_Sewing()`` call as
  ``makeShell`` (kept here to read ``Modified``). Every body must sew into one valid shell that keeps the as-built
  orientation of every face.
* **Shared loop**: per ring, the edges the two faces of each body share there form one closed loop in each body.
  Where the bodies cut that loop at different vertices (22B8: the rod side's chart seam halves the ring, the sleeve's
  faces keep it whole), a body's loop edge is cut at the other body's loop vertex, as the seam split of
  ``_seam_junctions`` cuts a partner edge: the pieces keep the edge's own curve, pcurves and tolerance on sub-ranges
  of a same-parameter edge, and the vertex must lie on the curve within its OWN tolerance (R206; never the edge's).
  Each cut is recorded (``loop_edge_splits``: point, parameter, gap, vertex and edge tolerance). The loop edges must
  then match one to one: the same 3D points in the same direction over equal parameter spans, up to a constant
  parameter shift (22B8: the rod's loop circle starts at -x, the sleeve's at +x, an offset of pi). Each pair becomes
  one edge: a copy of the first body's edge (``ShapeBuild_Edge.Copy``: its curve, pcurves, range and tolerance) that
  also carries the second body's pcurves, re-expressed exactly for the shift where there is one (a 2D line moved along
  itself, a 2D circle's frame turned; any other pcurve refuses; each shift is recorded in ``parameter_shifts``); both
  bodies then use it (``BRepTools_ReShape``), and the second body's loop vertices are the first body's. Nothing is
  re-fitted or approximated, no tolerance is raised (``UpdateEdge`` with the edge's own
  tolerance; checked afterwards), no ShapeFix / SameParameter call runs. BRepMesh therefore discretises the loop once
  and the parent welds it into 4-triangle edges (Task 6b).
* **Accepted** only if every final shell is valid, closed and oriented (the existing ``_check_shell_closure``), keeps
  the as-built orientation of every face, encloses a finite positive volume as oriented, and ``Solid.makeSolid`` and
  the compound of the solids (what ``_lump`` builds) are valid with the same positive volumes.

Fail closed: any other outcome, or any exception (OCC ``Standard_Failure`` included), refuses the whole split; the
attempt's closure records are cut back, the reason is recorded in ``converter.nonmanifold_split_refusals`` and today's
path (``makeShell`` of all faces, ending in today's result or refusal) runs. An accepted split is recorded in
``converter.nonmanifold_splits`` (``shell``, ``edges``, ``bodies``, ``volumes``, ``shared_loop_edges``,
``loop_edge_splits``, ``parameter_shifts``, ``method``);
no dwg2glb schema change. A shell without a > 2-coedge source edge never reaches the split (``nonmanifold_rings``
returns {}), so every shell M4 converts takes exactly today's path (census: 0 of 2,037 emitted definitions has such an
edge). Only the outer shell of a lump without cavity shells is split (``_lump``: "cavities in a split non-manifold
shell are not supported" stays).
"""

from __future__ import annotations

import math
from itertools import product

METHOD = "non-manifold source edge split into body-consistent solids"
MAX_PAIRINGS = 64  # pairing combinations tried; 22B8 has 2
SAMPLES = 17  # points compared along a loop edge
# Agreement of two loop edges' parameter ranges, relative to the range; a consistency check, not a modelling tolerance.
RANGE_AGREEMENT = 1e-12
VOLUME_AGREEMENT = 1e-9


class _Refused(Exception):
    """A qualification that does not hold; its text is the recorded reason."""


# ------------------------------------------------------------------ source topology and partition (no OCC)

def coedge_rings(converter, faces) -> dict:
    """{source edge index: [(coedge index, face index, coedge reversed), ...]} over the shell's faces."""
    from .model import LoopEntity
    rings = {}
    for face in faces:
        if face.loop.is_null:
            continue
        loops = converter._linked_entities(face.loop, LoopEntity, "next_loop", context=f"face ${face.index} loops")
        for loop in loops:
            for coedge in converter._coedges(loop):
                rings.setdefault(coedge.edge.index, []).append((coedge.index, face.index, bool(coedge.reversed)))
    return rings


def nonmanifold_rings(converter, faces) -> dict:
    """The shell's coedge rings when some source edge has more than 2 coedges, else {} (also when the topology
    cannot be read: today's path then runs and reports as today)."""
    try:
        rings = coedge_rings(converter, faces)
    except Exception:
        return {}
    return rings if any(len(ring) > 2 for ring in rings.values()) else {}


def _pairings(ring):
    """Every split of the ring's coedges into pairs of opposite senses."""
    if not ring:
        return [[]]
    first, rest = ring[0], ring[1:]
    out = []
    for k, other in enumerate(rest):
        if other[2] != first[2]:
            for tail in _pairings(rest[:k] + rest[k + 1:]):
                out.append([(first, other)] + tail)
    return out


def split_bodies(faces, coedge_rings, reasons=None):
    """The bodies (lists of face entities, in shell order) of the unique body-consistent partition, else None.

    ``coedge_rings`` as returned by :func:`coedge_rings`; a refusal reason is appended to ``reasons`` if given."""
    def refuse(reason):
        if reasons is not None:
            reasons.append(reason)
        return None

    position = {}
    for k, face in enumerate(faces):
        if face.index in position:
            return refuse(f"face ${face.index} appears twice in the shell")
        position[face.index] = k
    big = {edge: ring for edge, ring in coedge_rings.items() if len(ring) > 2}
    if not big:
        return refuse("no source edge has more than 2 coedges")
    for edge, ring in sorted(big.items()):
        if len(ring) != 4:
            return refuse(f"edge ${edge} has {len(ring)} coedges; only 4-coedge edges are split")
    for ring in coedge_rings.values():
        if any(face not in position for _, face, _ in ring):
            return refuse("a coedge lies on a face outside the shell")
    manifold = [(position[ring[0][1]], position[ring[1][1]]) for ring in coedge_rings.values() if len(ring) == 2]
    options = []
    for edge, ring in sorted(big.items()):
        pairings = _pairings(list(ring))
        if not pairings:
            return refuse(f"edge ${edge}: no pairing of its coedges into opposite senses")
        options.append(pairings)
    if math.prod(len(p) for p in options) > MAX_PAIRINGS:
        return refuse(f"more than {MAX_PAIRINGS} pairings to try")
    valid = []
    for choice in product(*options):
        parent = list(range(len(faces)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for a, b in manifold:
            union(a, b)
        for pairing in choice:
            for one, two in pairing:
                union(position[one[1]], position[two[1]])
        consistent = all(len({find(position[one[1]]) for one, _ in pairing}) == len(pairing) for pairing in choice)
        if consistent:
            groups = {}
            for k in range(len(faces)):
                groups.setdefault(find(k), []).append(k)
            valid.append(sorted(groups.values(), key=min))
    if len(valid) != 1:
        return refuse("no body-consistent pairing" if not valid else f"ambiguous partition: {len(valid)} pairings")
    groups = valid[0]
    if len(groups) < 2 or sum(len(g) for g in groups) != len(faces):
        return refuse("the partition has fewer than two bodies")
    return [[faces[k] for k in group] for group in groups]


# ------------------------------------------------------------------ the converter's split (OCC)

def split_nonmanifold_shell(converter, shell, faces, built, rings):
    """Return the checked shells of the split bodies (one per body, sharing their loop edges) or None (refusal
    recorded, converter state restored). ``built`` are the converter's faces in ``faces`` order."""
    checks = len(converter.shell_closure_checks)
    edges = sorted(edge for edge, ring in rings.items() if len(ring) > 2)
    try:
        reasons = []
        bodies = split_bodies(faces, rings, reasons)
        if bodies is None:
            raise _Refused(reasons[0] if reasons else "no partition")
        shells, record = _attempt(converter, shell, faces, built, rings, bodies)
    except Exception as error:  # _Refused, CadQueryConversionError, OCC Standard_Failure: all refuse
        del converter.shell_closure_checks[checks:]
        reason = str(error) if isinstance(error, _Refused) else f"{type(error).__name__}: {error}"
        converter.nonmanifold_split_refusals.append({"shell": shell.index, "edges": edges, "reason": reason})
        return None
    converter.nonmanifold_splits.append({"shell": shell.index, "edges": edges, **record, "method": METHOD})
    return shells


def _attempt(converter, shell, faces, built, rings, bodies):
    lump = converter.model.resolve(shell.lump)
    if getattr(getattr(lump, "shell", None), "index", None) != shell.index:
        raise _Refused("the shell is not the outer shell of its lump")
    if not shell.next_shell.is_null:
        raise _Refused("the lump has a cavity shell")
    if any(not face.wrapped.Location().IsIdentity() for face in built):
        raise _Refused("a face of the shell is located")
    start = len(converter.shell_closure_checks)
    position = {face.index: k for k, face in enumerate(faces)}
    body_of = {position[f.index]: b for b, body in enumerate(bodies) for f in body}
    members = [[position[f.index] for f in body] for body in bodies]
    shells, current = [], {}
    for b, body in enumerate(members):
        sewn, mapping = _sew_body([built[k].wrapped for k in body])
        _one_valid_shell(converter, sewn, shell, b)
        shells.append(sewn)
        current.update(zip(body, mapping))
        _orientation_kept(sewn, [(built[k].wrapped, current[k]) for k in body])
    before = [_max_tolerance(s) for s in shells]
    shared, cuts, shifts = 0, [], []
    for edge in sorted(e for e, ring in rings.items() if len(ring) > 2):
        by_body = {}
        for _, face, _ in rings[edge]:
            by_body.setdefault(body_of[position[face]], []).append(position[face])
        (x, xs), (y, ys) = sorted(by_body.items())
        if len(set(xs)) != 2 or len(set(ys)) != 2:
            raise _Refused(f"edge ${edge}: a body meets it twice in one face")
        count, shells[x], shells[y], current, edge_cuts, edge_shifts = _share_loop(
            edge, shells[x], shells[y], xs, ys, current)
        shared += count
        cuts.extend({**cut, "body": x if cut["body"] == "x" else y} for cut in edge_cuts)
        shifts.extend({**shift, "body": y} for shift in edge_shifts)
    del converter.shell_closure_checks[start:]  # the closure records describe the final shells only
    volumes, wrapped = [], []
    for b, body in enumerate(members):
        if _max_tolerance(shells[b]) > before[b]:
            raise _Refused(f"body {b}: sharing the loop raised a tolerance")
        _one_valid_shell(converter, shells[b], shell, b)
        _orientation_kept(shells[b], [(built[k].wrapped, current[k]) for k in body])
        volume = _volume(shells[b])
        if not math.isfinite(volume) or volume <= 0:
            raise _Refused(f"body {b}: volume {volume!r} is not positive")
        volumes.append(volume)
        wrapped.append(converter.cq.Shell(shells[b]))
    solids = [converter.cq.Solid.makeSolid(component) for component in wrapped]  # as _lump builds them
    for b, (solid, volume) in enumerate(zip(solids, volumes)):
        solid_volume = solid.Volume()
        if not (solid.isValid() and math.isfinite(solid_volume)
                and abs(solid_volume - volume) <= VOLUME_AGREEMENT * volume):
            raise _Refused(f"body {b}: the solid is invalid or its volume {solid_volume!r} differs from {volume!r}")
    if not converter.cq.Compound.makeCompound(solids).isValid():
        raise _Refused("the compound of the split solids is invalid")
    record = {"bodies": [[f.index for f in body] for body in bodies], "volumes": volumes,
              "shared_loop_edges": shared, "loop_edge_splits": cuts, "parameter_shifts": shifts}
    return tuple(wrapped), record


def _sew_body(faces):
    """(sewn shell, the sewn face of every input face): ``makeShell``'s sewing, with its ``Modified`` map."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    sewing = BRepBuilderAPI_Sewing()
    for face in faces:
        sewing.Add(face)
    sewing.Perform()
    result = sewing.SewedShape()
    return result, [sewing.Modified(face) for face in faces]


def _one_valid_shell(converter, sewn, shell, body):
    """The existing checks of a sewn shell on one body: exactly one valid, closed, oriented shell."""
    from OCP.TopAbs import TopAbs_SHELL
    if sewn.ShapeType() != TopAbs_SHELL:
        raise _Refused(f"body {body} did not sew into one shell")
    component = converter.cq.Shell(sewn)
    if not component.isValid():
        raise _Refused(f"body {body}: the sewn shell is invalid")
    converter._check_shell_closure(component, shell.index, body)  # raises shell_open / shell_orientation


def _shell_faces(sewn):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    out = []
    explorer = TopExp_Explorer(sewn, TopAbs_FACE)
    while explorer.More():
        out.append(explorer.Current())
        explorer.Next()
    return out


def _orientation_kept(sewn, pairs):
    """Every face appears once in the shell, with the orientation it was built with."""
    found = _shell_faces(sewn)
    if len(found) != len(pairs):
        raise _Refused(f"the body's shell has {len(found)} faces for {len(pairs)} built faces")
    for built, mapped in pairs:
        hits = [face for face in found if face.IsSame(mapped)]
        if len(hits) != 1:
            raise _Refused("a built face is not in the body's shell exactly once")
        if hits[0].Orientation() != built.Orientation():
            raise _Refused("sewing changed the as-built orientation of a face")


def _edges_between(sewn, one, two):
    """The edges of ``sewn`` whose only two faces are ``one`` and ``two``."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    data = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(sewn, TopAbs_EDGE, TopAbs_FACE, data)
    out = []
    for k in range(1, data.Extent() + 1):
        faces = list(data.FindFromIndex(k))
        unique = []
        for face in faces:
            if not any(face.IsSame(u) for u in unique):
                unique.append(face)
        if len(unique) == 2 and {_which(unique[0], one, two), _which(unique[1], one, two)} == {1, 2}:
            out.append(TopoDS.Edge_s(data.FindKey(k)))
    return out


def _which(face, one, two):
    return 1 if face.IsSame(one) else 2 if face.IsSame(two) else 0


def _forward(edge):
    from OCP.TopAbs import TopAbs_FORWARD
    from OCP.TopoDS import TopoDS
    return TopoDS.Edge_s(edge.Oriented(TopAbs_FORWARD))


def _same_edge(a, b):
    """The parameter shift ``d`` with b(t + d) == a(t) when ``a`` and ``b`` (forward) are the same curve piece in the
    same direction: equal parameter spans, every sampled point within the smaller edge tolerance. Else None."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    if BRep_Tool.Degenerated_s(a) or BRep_Tool.Degenerated_s(b):
        return None
    if not (a.Location().IsIdentity() and b.Location().IsIdentity()):
        return None
    ca, cb = BRepAdaptor_Curve(a), BRepAdaptor_Curve(b)
    fa, la, fb, lb = ca.FirstParameter(), ca.LastParameter(), cb.FirstParameter(), cb.LastParameter()
    span = max(abs(la - fa), abs(lb - fb), 1e-300)
    if abs((la - fa) - (lb - fb)) > RANGE_AGREEMENT * span:
        return None
    shift = fb - fa
    limit = min(BRep_Tool.Tolerance_s(a), BRep_Tool.Tolerance_s(b))
    for k in range(SAMPLES):
        t = fa + (la - fa) * k / (SAMPLES - 1)
        if ca.Value(t).Distance(cb.Value(t + shift)) > limit:
            return None
    return 0.0 if abs(shift) <= RANGE_AGREEMENT * span else shift


def _shifted(pcurve, shift):
    """``pcurve`` re-expressed exactly for the parameter t - ``shift``: c'(t) = c(t + shift). Lines and circles only
    (a line moved along itself, a circle's frame turned); anything else refuses (no fit)."""
    from OCP.Geom2d import Geom2d_Circle, Geom2d_Line, Geom2d_TrimmedCurve
    from OCP.gp import gp_Ax22d, gp_Ax2d, gp_Circ2d, gp_Dir2d
    if shift == 0.0:
        return pcurve
    basis = pcurve.BasisCurve() if isinstance(pcurve, Geom2d_TrimmedCurve) else pcurve
    if isinstance(basis, Geom2d_Line):
        line = basis.Lin2d()
        return Geom2d_Line(gp_Ax2d(line.Location().Translated(_scaled(line.Direction(), shift)), line.Direction()))
    if isinstance(basis, Geom2d_Circle):
        circle = basis.Circ2d()
        axis = circle.Axis()
        x, y = axis.XDirection(), axis.YDirection()
        c, s_ = math.cos(shift), math.sin(shift)
        nx = gp_Dir2d(c * x.X() + s_ * y.X(), c * x.Y() + s_ * y.Y())
        ny = gp_Dir2d(-s_ * x.X() + c * y.X(), -s_ * x.Y() + c * y.Y())
        return Geom2d_Circle(gp_Circ2d(gp_Ax22d(axis.Location(), nx, ny), circle.Radius()))
    raise _Refused(f"a loop pcurve ({type(basis).__name__}) cannot be re-expressed exactly for the parameter shift")


def _scaled(direction, length):
    from OCP.gp import gp_Vec2d
    return gp_Vec2d(direction.X() * length, direction.Y() * length)


def _closed_loop(edges):
    """Every vertex of ``edges`` ends exactly two of them (a closed edge counts its vertex twice)."""
    from OCP.TopExp import TopExp
    ends = []
    for edge in edges:
        for vertex in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)):
            if vertex.IsNull():
                return False
            ends.append(vertex)
    for vertex in ends:
        if sum(1 for other in ends if other.IsSame(vertex)) != 2:
            return False
    return bool(edges)


def _share_loop(edge_index, sx, sy, xs, ys, current):
    """Make the shells ``sx`` and ``sy`` use the same edges on the loop of source edge ``edge_index``.

    ``xs`` / ``ys`` are the positions of the two faces of each body at that edge; ``current`` maps a built face
    position to its face in the current shells. Returns (shared edges, new sx, new sy, updated current)."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_ReShape
    from OCP.ShapeBuild import ShapeBuild_Edge
    from OCP.TopAbs import TopAbs_FORWARD
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    loop_x = [_forward(e) for e in _edges_between(sx, current[xs[0]], current[xs[1]])]
    loop_y = [_forward(e) for e in _edges_between(sy, current[ys[0]], current[ys[1]])]
    if not (_closed_loop(loop_x) and _closed_loop(loop_y)):
        raise _Refused(f"edge ${edge_index}: the loop edges of a body do not form a closed loop")
    # The bodies' faces may cut the loop at different vertices (22B8: the rod side's chart seam halves the ring, the
    # sleeve's faces keep it whole): cut each body's loop edges at the other body's loop vertices first.
    cuts = []
    for target in ("y", "x"):
        mine, theirs = (loop_y, loop_x) if target == "y" else (loop_x, loop_y)
        pieces = _cuts_for(edge_index, mine, theirs)
        if not pieces:
            continue
        shell_ = sy if target == "y" else sx
        new_shell, reshape, records = _cut_edges(edge_index, shell_, pieces)
        for k, face in list(current.items()):
            value = reshape.Value(face)
            if not value.IsNull() and not value.IsSame(face):
                current[k] = value
        if target == "y":
            sy = new_shell
            loop_y = [_forward(e) for e in _edges_between(sy, current[ys[0]], current[ys[1]])]
        else:
            sx = new_shell
            loop_x = [_forward(e) for e in _edges_between(sx, current[xs[0]], current[xs[1]])]
        cuts.extend({"body": target, **record} for record in records)
    if not loop_x or len(loop_x) != len(loop_y):
        raise _Refused(f"edge ${edge_index}: the bodies carry {len(loop_x)} and {len(loop_y)} loop edges")
    if not (_closed_loop(loop_x) and _closed_loop(loop_y)):
        raise _Refused(f"edge ${edge_index}: the cut loop edges of a body do not form a closed loop")
    matches = []
    for ex in loop_x:
        hits = [(ey, shift) for ey in loop_y for shift in [_same_edge(ex, ey)] if shift is not None]
        if len(hits) != 1:
            raise _Refused(f"edge ${edge_index}: a loop edge of the first body matches {len(hits)} edges of the "
                           "second (the loop cannot be shared without a re-fit or a tolerance change)")
        matches.append((ex, *hits[0]))
    if any(sum(1 for _, other, _ in matches if other.IsSame(ey)) != 1 for _, ey, _ in matches):
        raise _Refused(f"edge ${edge_index}: the loop edges do not match one to one")
    builder = BRep_Builder()
    copier = ShapeBuild_Edge()
    reshape_x, reshape_y = BRepTools_ReShape(), BRepTools_ReShape()
    replaced = []
    faces_y = [TopoDS.Face_s(current[k]) for k in ys]
    shifts = []
    for ex, ey, shift in matches:
        tolerance = BRep_Tool.Tolerance_s(ex)
        curve = BRepAdaptor_Curve(ex)
        first, last = curve.FirstParameter(), curve.LastParameter()
        shared = copier.Copy(ex, True)
        if shift:
            point = curve.Value(first)
            shifts.append({"edge": edge_index, "shift": shift, "point": (point.X(), point.Y(), point.Z())})
        for face in faces_y:
            pcurve = BRep_Tool.CurveOnSurface_s(ey, face, first, last)
            if pcurve is None:
                raise _Refused(f"edge ${edge_index}: a loop edge has no pcurve on the second body's face")
            pcurve = _shifted(pcurve, shift)
            surface = BRep_Tool.Surface_s(face)
            for k in range(SAMPLES):
                t = first + (last - first) * k / (SAMPLES - 1)
                uv = pcurve.Value(t)
                if surface.Value(uv.X(), uv.Y()).Distance(curve.Value(t)) > tolerance:
                    raise _Refused(f"edge ${edge_index}: the second body's pcurve leaves the shared edge's "
                                   "tolerance")
            builder.UpdateEdge(shared, pcurve, face, tolerance)
        if BRep_Tool.Tolerance_s(shared) != tolerance:
            raise _Refused(f"edge ${edge_index}: the shared edge's tolerance changed")
        reshape_x.Replace(ex, shared)
        reshape_y.Replace(ey, shared)
        for vx, vy in ((TopExp.FirstVertex_s(ex), TopExp.FirstVertex_s(ey)),
                       (TopExp.LastVertex_s(ex), TopExp.LastVertex_s(ey))):
            if vy.IsSame(vx) or any(vy.IsSame(done) for done in replaced):
                continue
            if BRep_Tool.Pnt_s(vx).Distance(BRep_Tool.Pnt_s(vy)) > BRep_Tool.Tolerance_s(vx):
                raise _Refused(f"edge ${edge_index}: a loop vertex of the second body lies outside the first "
                               "body's vertex tolerance")
            reshape_y.Replace(vy.Oriented(TopAbs_FORWARD), vx.Oriented(TopAbs_FORWARD))
            replaced.append(vy)
    new_x, new_y = reshape_x.Apply(sx), reshape_y.Apply(sy)
    updated = dict(current)
    for k, face in current.items():
        for reshape in (reshape_x, reshape_y):
            value = reshape.Value(face)
            if not value.IsNull() and not value.IsSame(face):
                updated[k] = value
    loop_x2 = _edges_between(new_x, updated[xs[0]], updated[xs[1]])
    loop_y2 = _edges_between(new_y, updated[ys[0]], updated[ys[1]])
    if len(loop_x2) != len(matches) or not all(any(e.IsSame(o) for o in loop_y2) for e in loop_x2):
        raise _Refused(f"edge ${edge_index}: the shells do not share the loop edges after the replacement")
    return len(matches), new_x, new_y, updated, cuts, shifts


def _vertices(edges):
    from OCP.TopExp import TopExp
    out = []
    for edge in edges:
        for vertex in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)):
            if not any(vertex.IsSame(known) for known in out):
                out.append(vertex)
    return out


def _cuts_for(edge_index, mine, theirs):
    """{position in ``mine``: [(parameter, vertex, gap)]} for every vertex of the other body's loop ``theirs`` that
    is no vertex of ``mine``: it must lie strictly inside exactly one edge of ``mine``, on that edge's curve within
    the vertex's OWN tolerance (the R206 bound of the seam split; never the edge's tolerance)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
    known = _vertices(mine)
    out = {}
    for vertex in _vertices(theirs):
        point = BRep_Tool.Pnt_s(vertex)
        tolerance = BRep_Tool.Tolerance_s(vertex)
        if any(point.Distance(BRep_Tool.Pnt_s(v)) <= max(tolerance, BRep_Tool.Tolerance_s(v)) for v in known):
            continue
        hits = []
        for k, edge in enumerate(mine):
            curve = BRepAdaptor_Curve(edge)
            first, last = curve.FirstParameter(), curve.LastParameter()
            projection = GeomAPI_ProjectPointOnCurve(point, curve.Curve().Curve(), first, last)
            if projection.NbPoints() == 0:
                continue
            gap, parameter = projection.LowerDistance(), projection.LowerDistanceParameter()
            if first < parameter < last and gap <= max(tolerance, BRep_Tool.Tolerance_s(edge)):
                hits.append((k, parameter, gap))
        if len(hits) != 1:
            raise _Refused(f"edge ${edge_index}: a loop vertex of one body lies inside {len(hits)} loop edges of "
                           "the other")
        k, parameter, gap = hits[0]
        if gap > tolerance:
            raise _Refused(f"edge ${edge_index}: loop cut refused: gap {gap:.6g} mm > vertex tolerance "
                           f"{tolerance:.6g} mm")
        out.setdefault(k, []).append((parameter, vertex, gap))
    return {k: (mine[k], sorted(items, key=lambda item: item[0])) for k, items in out.items()}


def _cut_edges(edge_index, sewn, pieces):
    """``sewn`` with every listed loop edge cut at its vertices: the pieces keep the edge's own curve, pcurves (on
    both faces) and tolerance on sub-ranges of a same-parameter edge; the cut vertex is the other body's vertex."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_ReShape
    from OCP.ShapeBuild import ShapeBuild_Edge
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS_Wire
    builder = BRep_Builder()
    copier = ShapeBuild_Edge()
    reshape = BRepTools_ReShape()
    records = []
    for edge, items in pieces.values():
        if not edge.Location().IsIdentity():
            raise _Refused(f"edge ${edge_index}: a loop edge is located")
        if not (BRep_Tool.SameParameter_s(edge) and BRep_Tool.SameRange_s(edge)):
            raise _Refused(f"edge ${edge_index}: a loop edge to cut is not same-parameter / same-range")
        curve = BRepAdaptor_Curve(edge)
        first, last = curve.FirstParameter(), curve.LastParameter()
        bounds = [(first, TopExp.FirstVertex_s(edge))] + [(t, v) for t, v, _ in items] + \
            [(last, TopExp.LastVertex_s(edge))]
        wire = TopoDS_Wire()
        builder.MakeWire(wire)
        for (t0, v0), (t1, v1) in zip(bounds, bounds[1:]):
            if not t0 < t1:
                raise _Refused(f"edge ${edge_index}: two loop cuts at one parameter")
            piece = copier.CopyReplaceVertices(edge, v0, v1)
            builder.Range(piece, t0, t1)
            builder.Add(wire, piece)
        reshape.Replace(edge, wire)
        for t, vertex, gap in items:
            point = BRep_Tool.Pnt_s(vertex)
            records.append({"edge": edge_index, "point": (point.X(), point.Y(), point.Z()), "parameter": t,
                            "gap_mm": gap, "vertex_tolerance_mm": BRep_Tool.Tolerance_s(vertex),
                            "edge_tolerance_mm": BRep_Tool.Tolerance_s(edge)})
    return reshape.Apply(sewn), reshape, records


def _max_tolerance(shape):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    worst = 0.0
    for kind, cast in ((TopAbs_EDGE, TopoDS.Edge_s), (TopAbs_VERTEX, TopoDS.Vertex_s)):
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            worst = max(worst, BRep_Tool.Tolerance_s(cast(explorer.Current())))
            explorer.Next()
    return worst


def _volume(sewn):
    """Signed volume enclosed by the shell as oriented (no orientation fix)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS, TopoDS_Solid
    builder = BRep_Builder()
    solid = TopoDS_Solid()
    builder.MakeSolid(solid)
    builder.Add(solid, TopoDS.Shell_s(sewn))
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, properties)
    return properties.Mass()
