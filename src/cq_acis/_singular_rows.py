"""dwg2glb (M4b F7): close a stored-fit face whose fit collapses a band of end pole rows.

ASM saves some `rb_blend_spl_sur` fits whose end pole rows collapse to the apex of a three-sided face; the fit's
singularity code (1 = low, 2 = high) names that end. The two apex coedges meet the collapsed row at different free
parameters (u = 0 and u = 1): a UV gap of the whole free range at one 3D point. ``_singular_segment`` closes such a
gap only when the surface opens fast (the iso 5 % into it is longer than 10 x the join limit); these fits open too
slowly, so the face stays open in UV and is invalid (``UnorientableShape``).

This module is a fallback only. ``CadQueryConverter._face`` calls :func:`close_collapsed_rows` after the face is
invalid and the existing heal did not help, just before it raises ``geometry.face_invalid``. The face is rebuilt
once through ``_bspline_face`` with one degenerated edge on the collapsed (knot-bound) row per qualified join. Every
qualification is required; anything else records a refusal in ``singular_row_refusals`` and returns the invalid
face, so the caller raises ``geometry.face_invalid`` exactly as before. A face the old path builds never gets here.

Qualification:

* the fit's ACIS singularity code, read in Python from the surface record by a mirror of the Rust stored-fit reader
  (``approximation.rs`` ``surface_at``), cross-checked field by field against the Rust-decoded fit and closure; a
  parse failure or any disagreement refuses;
* the poles: a band of k >= 1 consecutive collapsed rows (each row's poles within the model tolerance of one point)
  at exactly one end, the end the code names, and a later row spreading by more than 10 x the tolerance;
* each candidate join (one per loop, where ``_singular_segment`` refused): both UV ends on the row parameter within
  the parameter tolerance, 3D gap within the join limit, the shared source vertex within tolerance + fit allowance
  of the band point, free-parameter gap above 10 x the parameter tolerance.

Acceptance of the rebuilt face: ``isValid``; BRepCheck face, wire-in-face and edge-on-face NoError; area finite and
positive; exactly one degenerated edge per qualified join. Any OCC exception refuses.
"""
from __future__ import annotations

import math

from .model import EdgeEntity, RawEntity, Vec3

ENDS = ("u_low", "u_high", "v_low", "v_high")
METHOD = "collapsed_row_of_stored_fit"
_CODES = {0: 0, 1: 1, 2: 2, 3: 3, "none": 0, "low": 1, "high": 2, "both": 3}
_CLOSURES = {0: 0, 1: 1, 2: 2, "open": 0, "closed": 1, "periodic": 2}
_RATIONAL = {"nubs": False, "nurbs": True}
_RATIONAL_DIRECTIONS = ("both", "u", "v", "none")


class _Refusal(Exception):
    """A qualification or acceptance condition does not hold."""


class _Unreadable(ValueError):
    """The mirror cannot read the stored fit."""


def _tag(value, expected: int) -> bool:
    return isinstance(value, bytes) and value == bytes([expected])


def _word(value):
    return value if isinstance(value, str) else None


class _Reader:
    """Mirror of ``asm_nurbs::Reader`` for the fields a stored surface fit uses."""

    def __init__(self, values, pos):
        self.values, self.pos = values, pos

    def take(self):
        if self.pos >= len(self.values):
            raise _Unreadable("truncated stored fit")
        value = self.values[self.pos]
        self.pos += 1
        return value

    def integer(self) -> int:
        value = self.take()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _Unreadable("expected an unsigned integer")
        return value

    def number(self) -> float:
        value = self.take()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _Unreadable("expected a number")
        value = float(value)
        if not math.isfinite(value):
            raise _Unreadable("non-finite number")
        return value

    def code(self, table, what):
        value = self.take()
        if isinstance(value, bool) or not isinstance(value, (int, str)) or value not in table:
            raise _Unreadable(f"unqualified stored spline {what}")
        return table[value]

    def knots(self, degree: int, count: int):
        if not 1 <= degree <= 16 or not 2 <= count <= 4096:
            raise _Unreadable("knot/degree limit")
        knots, multiplicities = [], []
        for _ in range(count):
            knots.append(self.number())
            multiplicities.append(self.integer())
        if multiplicities[0] != degree or multiplicities[-1] != degree:
            raise _Unreadable("unqualified endpoint convention")
        full = [multiplicities[0] + 1, *multiplicities[1:-1], multiplicities[-1] + 1]
        poles = sum(full) - (degree + 1)
        if poles < 1:
            raise _Unreadable("invalid pole count")
        return knots, full, poles

    def bound(self):
        value = self.take()
        if _tag(value, 0x0B):
            return None
        if _tag(value, 0x0A):
            return self.number()
        raise _Unreadable("invalid bound")

    def range(self):
        return self.bound(), self.bound()


def _scope_index(model):
    """Mirror of ``ApproximationIndex::new``: every non-reference subtype scope in opening order."""
    scopes, broken_at = [], None
    for entity in model.entities:
        raw = entity if isinstance(entity, RawEntity) else entity.raw
        values, stack = raw.values, []
        for position, value in enumerate(values):
            if _tag(value, 15):
                word = _word(values[position + 1]) if position + 1 < len(values) else None
                if word == "ref":
                    stack.append(None)
                elif word is not None:
                    scopes.append([word, raw.index, position, 0])
                    stack.append(len(scopes) - 1)
                else:
                    broken_at = raw.index if broken_at is None else broken_at
                    stack.append(None)
            elif _tag(value, 16):
                if not stack:
                    broken_at = raw.index if broken_at is None else broken_at
                else:
                    slot = stack.pop()
                    if slot is not None:
                        scopes[slot][3] = position + 1
        if stack:
            broken_at = raw.index if broken_at is None else broken_at
    return scopes, broken_at


def _expanded(model, raw, index_cache):
    """The record's values with a ``ref`` envelope expanded, as ``approximation::resolve`` does."""
    values = tuple(raw.values)
    if len(values) < 4 or not _tag(values[2], 15) or not (_tag(values[1], 10) or _tag(values[1], 11)):
        raise _Unreadable("not a subtype envelope")
    if _word(values[3]) != "ref":
        return values
    if len(values) < 6 or isinstance(values[4], bool) or not isinstance(values[4], int) or not _tag(values[5], 16):
        raise _Unreadable("invalid subtype reference")
    if "scopes" not in index_cache:
        index_cache["scopes"] = _scope_index(model)
    scopes, broken_at = index_cache["scopes"]
    if broken_at is not None and raw.index >= broken_at:
        raise _Unreadable("subtype numbering is unqualified after a malformed scope")
    target = values[4]
    if not 0 <= target < len(scopes) or scopes[target][3] == 0 or scopes[target][1] > raw.index:
        raise _Unreadable("unresolved subtype reference")
    _, owner_index, start, end = scopes[target]
    owner = model.entities[owner_index]
    owner_values = tuple((owner if isinstance(owner, RawEntity) else owner.raw).values)
    return values[:2] + owner_values[start:end] + values[6:]


def _read_fit(model, raw, index_cache):
    """Mirror of ``approximation.rs`` ``surface`` / ``surface_at``: the last stored surface fit directly inside the
    subtype scope that reads completely. Returns its header, knots, poles, fit tolerance and singularity codes."""
    values = _expanded(model, raw, index_cache)
    depth, close = 0, None
    for position in range(2, len(values)):
        if _tag(values[position], 15):
            depth += 1
        elif _tag(values[position], 16):
            depth -= 1
            if depth < 0:
                raise _Unreadable("unmatched subtype close")
            if depth == 0:
                close = position
                break
    if close is None:
        raise _Unreadable("unclosed subtype")
    candidates, depth = [], 0
    for position in range(2, close):
        value = values[position]
        if _tag(value, 15):
            depth += 1
        elif _tag(value, 16):
            depth = max(depth - 1, 0)
        elif depth == 1 and _word(value) in _RATIONAL:
            candidates.append(position)
    failure = None
    for start in reversed(candidates):
        try:
            return _surface_at(values, start, close)
        except _Unreadable as error:
            failure = failure or error
    raise failure or _Unreadable("no stored surface fit")


def _surface_at(values, start, close):
    rational = _RATIONAL[values[start]]
    r = _Reader(values, start + 1)
    u_degree, v_degree = r.integer(), r.integer()
    if r.pos < len(values) and _word(values[r.pos]) in _RATIONAL_DIRECTIONS:
        r.pos += 1
    closure = (r.code(_CLOSURES, "closure"), r.code(_CLOSURES, "closure"))
    codes = (r.code(_CODES, "singularity"), r.code(_CODES, "singularity"))
    u_count_knots, v_count_knots = r.integer(), r.integer()
    u_knots, u_multiplicities, u_count = r.knots(u_degree, u_count_knots)
    v_knots, v_multiplicities, v_count = r.knots(v_degree, v_count_knots)
    count = u_count * v_count
    width = 4 if rational else 3
    if count > 1_000_000 or count > (len(values) - r.pos) // width:
        raise _Unreadable("truncated poles")
    poles, weights = [], []
    for _ in range(count):
        poles.append((r.number(), r.number(), r.number()))
        weights.append(r.number() if rational else 1.0)
    fit = r.number()
    if fit < 0 or r.pos > close:
        raise _Unreadable("invalid stored surface fit")
    tail = _Reader(values, close + 1)
    tail.range()
    tail.range()
    if tail.pos != len(values):
        raise _Unreadable("unconsumed fields after the stored surface")
    return {"codes": codes, "closure": closure, "u_degree": u_degree, "v_degree": v_degree, "u_knots": u_knots,
            "v_knots": v_knots, "u_multiplicities": u_multiplicities, "v_multiplicities": v_multiplicities,
            "u_count": u_count, "v_count": v_count, "poles": poles, "weights": weights, "fit": fit}


def stored_fit_singularity(converter, surface) -> tuple[int, int]:
    """The (u, v) singularity codes of ``surface``'s stored fit, read by the Python mirror and cross-checked against
    the Rust-decoded fit and closure. Raises ``_Refusal`` (reason starting with "mirror") otherwise."""
    raw = getattr(surface, "raw", None)
    closure = converter._approximation_closures.get(getattr(surface, "index", None))
    if not isinstance(raw, RawEntity) or raw.type_name != "spline-surface" or closure is None:
        raise _Refusal("mirror: the surface is not a stored spline-surface fit")
    cache = converter.__dict__.setdefault("_fit_scope_cache", {})
    try:
        fit = _read_fit(converter.model, raw, cache)
    except _Unreadable as error:
        raise _Refusal(f"mirror: the stored fit is unreadable ({error})") from None
    checks = {
        "closure": fit["closure"] == tuple(closure),
        "degrees": (fit["u_degree"], fit["v_degree"]) == (surface.u_degree, surface.v_degree),
        "knots": (tuple(fit["u_knots"]), tuple(fit["v_knots"])) == (tuple(surface.u_knots), tuple(surface.v_knots)),
        "multiplicities": (tuple(fit["u_multiplicities"]), tuple(fit["v_multiplicities"]))
        == (tuple(surface.u_multiplicities), tuple(surface.v_multiplicities)),
        "counts": (fit["u_count"], fit["v_count"]) == (surface.u_count, surface.v_count),
        "poles": len(fit["poles"]) == len(surface.poles)
        and all(p == (q.x, q.y, q.z) for p, q in zip(fit["poles"], surface.poles)),
        "weights": tuple(fit["weights"]) == tuple(surface.weights),
        "fit": fit["fit"] == surface.fit_tolerance,
    }
    wrong = [name for name, ok in checks.items() if not ok]
    if wrong:
        raise _Refusal(f"mirror: the stored fit read in Python disagrees with the Rust reader ({', '.join(wrong)})")
    return fit["codes"]


def _row_poles(surface, end):
    """Pole rows of ``surface`` ordered from ``end`` (a v row runs along u; a u row is a column along v)."""
    nu, nv = surface.NbUPoles(), surface.NbVPoles()
    if end.startswith("v"):
        rows = [[surface.Pole(i, j) for i in range(1, nu + 1)] for j in range(1, nv + 1)]
    else:
        rows = [[surface.Pole(i, j) for j in range(1, nv + 1)] for i in range(1, nu + 1)]
    return rows if end.endswith("low") else rows[::-1]


def _spread(row) -> float:
    return max(row[0].Distance(p) for p in row)


def _band(surface, singularity, tolerance):
    if singularity is None:
        return None, "no readable singularity code"
    if not hasattr(surface, "NbUPoles") or not hasattr(surface, "Pole"):
        return None, "the fit is not a B-spline surface"
    flagged = []
    for axis, code in zip("uv", singularity):
        if code == 3:
            return None, f"singularity code {tuple(singularity)} names both {axis} ends"
        if code in (1, 2):
            flagged.append(f"{axis}_{'low' if code == 1 else 'high'}")
    if len(flagged) != 1:
        return None, f"singularity code {tuple(singularity)} does not name exactly one end"
    (end,) = flagged
    bands = {}
    for name in ENDS:
        rows = _row_poles(surface, name)
        k = 0
        while k < len(rows) and _spread(rows[k]) <= tolerance:
            k += 1
        if k:
            bands[name] = (k, rows)
    if not bands:
        return None, f"no collapsed pole row at {end}, which the singularity code names"
    if list(bands) != [end]:
        return None, f"poles collapse at {sorted(bands)}, the singularity code names {end}"
    k, rows = bands[end]
    if not any(_spread(row) > 10 * tolerance for row in rows[k:]):
        return None, f"every pole row at {end} collapses: the fit is degenerate as a whole"
    bounds = surface.Bounds()
    row = bounds[ENDS.index(end)]
    return (end, k, row, rows[0][0]), None


def collapsed_band(fit_surface, singularity, tolerance):
    """``(end, band rows k, row parameter, band point)`` when ``fit_surface`` (an OCCT B-spline surface) collapses
    k >= 1 consecutive pole rows at exactly the one end the singularity code names, and a later row spreads by more
    than 10 x ``tolerance``; otherwise None."""
    return _band(fit_surface, singularity, tolerance)[0]


def row_segment(previous, join, axis, row):
    """The degenerated edge's UV ends: the as-built apex UVs with the row coordinate set exactly to the row
    parameter (the knot bound; an as-built end may overshoot it by an ulp or two)."""
    start, end = list(previous), list(join)
    start[axis] = row
    end[axis] = row
    return tuple(start), tuple(end)


def _qualify(converter, surface, placement, candidates):
    if not candidates:
        raise _Refusal("no candidate join on a collapsed row")
    codes = stored_fit_singularity(converter, surface)
    geometry = converter._bspline_geometry(surface, placement)
    band = collapsed_band(geometry, codes, converter.tolerance)
    if band is None:
        raise _Refusal(_band(geometry, codes, converter.tolerance)[1])
    end, band_rows, row, point = band
    axis = 0 if end.startswith("u") else 1
    free = 1 - axis
    parameter_tolerance = converter.tolerance  # _attach_surface_pcurves: tolerance / max(radius 1, 1)
    allowance = converter._approximation_tolerance(surface)
    apex = Vec3(point.X(), point.Y(), point.Z())
    by_loop = {}
    for candidate in candidates:
        by_loop.setdefault(candidate["loop"], []).append(candidate)
    plan = {}
    for loop, items in by_loop.items():
        if len(items) != 1:
            raise _Refusal(f"loop ${loop}: {len(items)} UV gaps the singular-point path refused (one qualifies)")
        (candidate,) = items
        previous, join = candidate["previous"], candidate["join"]
        if any(not math.isfinite(uv[axis]) or abs(uv[axis] - row) > parameter_tolerance for uv in (previous, join)):
            raise _Refusal(f"loop ${loop}: the apex UV ends are not on the collapsed row {end}")
        gap = geometry.Value(*previous).Distance(geometry.Value(*join))
        if not math.isfinite(gap) or gap > candidate["limit"]:
            raise _Refusal(f"loop ${loop}: 3D gap {gap!r} mm exceeds the join limit")
        before = converter._require(candidate["previous_coedge"].edge, EdgeEntity, context="collapsed row join")
        vertex = before.start_vertex if candidate["previous_coedge"].reversed else before.end_vertex
        location = placement.point_vector(converter._vertex_location(vertex, context="collapsed row apex"))
        distance = (location - apex).magnitude
        if not math.isfinite(distance) or distance > converter.tolerance + allowance:
            raise _Refusal(f"loop ${loop}: the apex vertex ${vertex.index} is {distance:.3g} mm off the band point "
                           f"(allowed {converter.tolerance + allowance:.3g} mm)")
        if not abs(previous[free] - join[free]) > 10 * parameter_tolerance:
            raise _Refusal(f"loop ${loop}: the free-parameter gap is not a row gap")
        plan[loop] = (candidate["position"], axis, row, point, candidate["limit"], end, band_rows)
    return plan


def row_insertion(converter, loop, edges, surface, fresh, singular):
    """Insert the planned degenerated edge into ``loop``'s rebuilt edges (``fresh``: the candidates this rebuild
    recorded for the loop). Called by ``_wire`` only during :func:`close_collapsed_rows`."""
    from .cadquery import CadQueryConversionError
    position, axis, row, point, limit, end, band_rows = converter._row_insertions[loop.index]
    if singular:
        raise CadQueryConversionError(f"loop ${loop.index}: singular point and collapsed row in one loop",
                                      code="geometry.face_invalid")
    matching = [c for c in fresh if c["position"] == position]
    if len(fresh) != 1 or len(matching) != 1:
        raise CadQueryConversionError(f"loop ${loop.index}: the rebuild's UV gaps differ from the qualified join",
                                      code="geometry.face_invalid")
    start, stop = row_segment(matching[0]["previous"], matching[0]["join"], axis, row)
    edges = list(edges)
    edges.insert(position, converter._degenerated_edge(surface, start, stop, point, limit))
    converter._row_insertion_records.append({"loop": loop.index, "position": position, "uv": (start, stop),
                                             "method": METHOD, "row": end, "band_rows": band_rows})
    return tuple(edges)


def _status_names(statuses):
    return [str(status).split(".")[-1] for status in statuses]


def _accept(face, joins):
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer, BRepCheck_Edge, BRepCheck_Wire
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    if not face.isValid():
        raise _Refusal("the rebuilt face is invalid")
    shape = TopoDS.Face_s(face.wrapped)
    bad = [name for name in _status_names(BRepCheck_Analyzer(shape).Result(shape).Status())
           if name != "BRepCheck_NoError"]
    explorer = TopExp_Explorer(shape, TopAbs_WIRE)
    while explorer.More():
        check = BRepCheck_Wire(TopoDS.Wire_s(explorer.Current()))
        check.Minimum()
        check.InContext(shape)
        bad += [f"wire {name}" for name in _status_names(check.StatusOnShape(shape)) if name != "BRepCheck_NoError"]
        explorer.Next()
    degenerated = 0
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        degenerated += bool(BRep_Tool.Degenerated_s(edge))
        check = BRepCheck_Edge(edge)
        check.InContext(shape)
        bad += [f"edge {name}" for name in _status_names(check.StatusOnShape(shape)) if name != "BRepCheck_NoError"]
        explorer.Next()
    if bad:
        raise _Refusal(f"BRepCheck on the rebuilt face: {sorted(set(bad))}")
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    area = props.Mass()
    if not math.isfinite(area) or area <= 0:
        raise _Refusal(f"the rebuilt face has area {area!r}")
    if degenerated != joins:
        raise _Refusal(f"{degenerated} degenerated edges for {joins} qualified joins")
    return area


def _snapshot(converter):
    lists = {name: len(value) for name, value in vars(converter).items() if isinstance(value, list)}
    return lists, converter.pcurve_count, converter.pcurve_max_deviation


def _restore(converter, snapshot, face_index, had_deferred):
    lists, count, deviation = snapshot
    for name, length in lists.items():
        del getattr(converter, name)[length:]
    converter.pcurve_count, converter.pcurve_max_deviation = count, deviation
    if not had_deferred:
        converter._deferred_periodic_seams.pop(face_index, None)


def close_collapsed_rows(converter, face, surface, loops, placement, result):
    """The fallback (see the module docstring): the rebuilt valid face, or ``result`` unchanged with a refusal
    recorded. The rebuild's own evidence (pcurve checks, joins, ...) duplicates the first build's and is dropped;
    only the degenerated-edge records are added."""
    candidates = list(converter._row_join_candidates)
    snapshot = _snapshot(converter)
    had_deferred = face.index in converter._deferred_periodic_seams
    records, reason = [], None
    try:
        plan = _qualify(converter, surface, placement, candidates)
        converter._row_insertions, converter._row_insertion_records = plan, records
        try:
            rebuilt = converter._bspline_face(face, surface, loops, placement)
        finally:
            converter._row_insertions, converter._row_insertion_records = None, []
        area = _accept(rebuilt, len(plan))
    except _Refusal as refusal:
        reason = str(refusal)
    except Exception as error:  # CadQueryConversionError from the rebuild, or any OCC exception
        reason = f"{type(error).__name__}: {error}"
    _restore(converter, snapshot, face.index, had_deferred)
    if reason is not None:
        converter.singular_row_refusals.append({"face": face.index, "reason": reason})
        return result
    converter.degenerate_edges.extend(records)
    converter.singular_row_faces.append({"face": face.index, "joins": len(records), "area_mm2": area,
                                         "row": records[0]["row"], "band_rows": records[0]["band_rows"]})
    return rebuilt
