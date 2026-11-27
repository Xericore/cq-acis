"""Qualified chart seams for cylindrical faces with two winding boundaries.

``periodic_seam_face`` extends this to circular cylinder and cone faces with any loop count and any
boundary curve (two opposite rim windings plus non-winding holes), with a seam parameter clear of every source
vertex and hole; a face it does not qualify stays ``geometry.face_invalid``.
"""
from __future__ import annotations

import math


def cylinder_seam_face(converter, face, loops, wires, geometry, original, placement, *, elliptic_sections=False):
    """Add a chart seam, retaining every oriented source curve and trim interval.

    Ordinary circular/linear boundaries use generated linear UV curves. Two
    complete elliptic plane sections additionally require an analytic support
    and separation proof. Both paths require opposite single windings; saved
    UV curves and general face healing are outside this operation.
    """
    from OCP.BRep import BRep_Tool
    from OCP.Geom import Geom_Circle, Geom_Line, Geom_Ellipse, Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_Line
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopAbs import TopAbs_REVERSED
    from .cadquery import CadQueryConversionError
    from .model import EdgeEntity, VertexEntity

    if not elliptic_sections and (len(loops) != 2 or len(wires) != 2
                                  or not isinstance(geometry.surface, Geom_CylindricalSurface)):
        return periodic_seam_face(converter, face, loops, wires, geometry, placement)
    if len(loops) != 2 or len(wires) != 2:
        return None
    source_surface = converter._resolve_geometry(face.surface)
    saved_ranges = (source_surface.u_range, source_surface.v_range)
    if elliptic_sections:
        saved_ranges += (source_surface.profile_range,)
    if any(r is not None and (r.lower is not None or r.upper is not None)
           for r in saved_ranges):
        return None
    coedges = [co for loop in loops for co in converter._coedges(loop)]
    if any(co.raw.type_name != 'coedge' or not co.pcurve.is_null for co in coedges):
        return None
    source_edges = [converter.model.resolve(co.edge) for co in coedges]
    if any(not isinstance(e, EdgeEntity) or e.raw.type_name != 'edge' for e in source_edges):
        return None
    if elliptic_sections:
        _check_elliptic_sections(converter, loops, geometry, placement)
    for source_edge in source_edges:
        vertices = [converter.model.resolve(v) for v in (source_edge.start_vertex, source_edge.end_vertex)]
        if any(not isinstance(v, VertexEntity) or v.raw.type_name != 'vertex' for v in vertices):
            return None
    boundary_points = [v.toTuple() for wire in wires for v in wire.Vertices()]
    for source_edge in source_edges:
        for vertex in (source_edge.start_vertex, source_edge.end_vertex):
            point = placement.point(converter._vertex_location(vertex, context='cylinder seam source vertex'))
            if not any(math.dist(point, actual) <= converter.tolerance for actual in boundary_points):
                raise CadQueryConversionError('cylinder trim does not retain a saved source vertex',
                                              code='geometry.analytic_trim_mismatch')
    boundaries = []
    winding = []
    for wire in wires:
        delta = [0., 0.]
        for edge in wire.Edges():
            curve = BRep_Tool.Curve_s(edge.wrapped, 0., 0.)
            pc = BRep_Tool.CurveOnSurface_s(edge.wrapped, original.wrapped, 0., 0.)
            if (not isinstance(curve, (Geom_Circle, Geom_Ellipse) if elliptic_sections else (Geom_Circle, Geom_Line))
                    or pc is None or (not elliptic_sections and not isinstance(pc, Geom2d_Line))):
                return None if elliptic_sections else periodic_seam_face(
                    converter, face, loops, wires, geometry, placement)
            first, last = BRep_Tool.Range_s(edge.wrapped)
            start, end = pc.Value(first), pc.Value(last)
            sign = -1 if edge.wrapped.Orientation() == TopAbs_REVERSED else 1
            delta[0] += sign * (end.X() - start.X())
            delta[1] += sign * (end.Y() - start.Y())
            boundaries.append((edge, curve, first, last))
        if abs(abs(delta[0]) - 2 * math.pi) > 1e-10 or abs(delta[1]) > converter.tolerance:
            return None
        winding.append(delta[0])
    if abs(sum(winding)) > 1e-10:
        raise CadQueryConversionError('cylinder boundary windings have inconsistent senses',
                                      code='geometry.boundary_orientation')
    fixer = ShapeFix_Face(original.wrapped)
    fixer.SetPrecision(converter.tolerance)
    fixer.SetMaxTolerance(converter.tolerance)
    if not fixer.FixMissingSeam():
        return None
    result = converter.cq.Face(fixer.Face())
    # FixMissingSeam may reverse every wire to give the chart its natural
    # orientation. Restore the source's whole-face sense before checking each
    # physical use; a mixture of changed senses is never accepted.
    restored = False
    if placement.orientation_sign > 0:
        senses = []
        for edge in result.Edges():
            if BRep_Tool.IsClosed_s(edge.wrapped, result.wrapped):
                continue
            curve = BRep_Tool.Curve_s(edge.wrapped, 0., 0.)
            matches = [source for source, saved, _, _ in boundaries if curve == saved]
            if len(matches) == 1:
                senses.append(edge.wrapped.Orientation() != matches[0].wrapped.Orientation())
        if senses and all(senses):
            result = converter._reverse_face(result)
            restored = True
    try:
        proof = _check_seam_boundaries(converter, result, geometry.surface, boundaries)
    except CadQueryConversionError:
        raise
    except Exception as error:
        raise CadQueryConversionError(f'face ${face.index}: cylinder seam verification failed: {error}',
                                      code='geometry.analytic_trim_mismatch') from error
    converter.periodic_seam_faces.append(dict(face=face.index,
        loops=[loop.index for loop in loops], edges=[co.edge.index for co in coedges],
        method='cylinder chart seam; unchanged oriented 3D curves and complete trim intervals',
        elliptic_sections=elliptic_sections, source_sense_restored=restored, **proof))
    return result


def _check_elliptic_sections(converter, loops, geometry, placement):
    """Prove two disjoint plane sections of the same circular cylinder.

    The projected major/minor axes must form a radius-R circle. Each section
    is then a single-valued height c + a*cos(U) + b*sin(U); its full separation
    from the other section has an analytic minimum, without sample fitting.
    """
    from .cadquery import CadQueryConversionError
    from .model import EdgeEntity, EllipseCurveEntity
    sections = []
    tol, radius = converter.tolerance, geometry.radius
    for loop in loops:
        coedges = converter._coedges(loop)
        if len(coedges) != 1:
            raise CadQueryConversionError('elliptic cylinder section must be a complete rim')
        edge = converter._require(coedges[0].edge, EdgeEntity,
                                  context='elliptic cylinder section')
        curve = converter._resolve_geometry(edge.curve)
        if not isinstance(curve, EllipseCurveEntity) or edge.start_vertex != edge.end_vertex:
            raise CadQueryConversionError('elliptic cylinder section must be a closed ellipse')
        if (edge.start_parameter is None or edge.end_parameter is None
                or not math.isfinite(edge.start_parameter + edge.end_parameter)
                or not math.isclose(abs(edge.end_parameter-edge.start_parameter), 2*math.pi,
                                    rel_tol=0., abs_tol=1e-10)):
            raise CadQueryConversionError('elliptic cylinder section does not retain a complete saved interval',
                                          code='geometry.analytic_trim_mismatch')
        delta = placement.point_vector(curve.center) - geometry.center
        major = placement.vector(curve.major_axis)
        normal = placement.oriented_direction(curve.normal).normalized()
        minor = normal.cross(major) * abs(curve.ratio)
        axial = normal.dot(geometry.axis)
        radial_center = delta - geometry.axis * delta.dot(geometry.axis)
        projected = [v - geometry.axis * v.dot(geometry.axis) for v in (major, minor)]
        if (radial_center.magnitude > tol or abs(axial) < 1e-8 or abs(normal.dot(major)) > tol
                or any(abs(v.magnitude-radius) > tol for v in projected)
                or abs(projected[0].dot(projected[1])) > tol*radius):
            raise CadQueryConversionError('ellipse is not a section of the source cylinder',
                                          code='geometry.analytic_trim_mismatch')
        sections.append((delta.dot(geometry.axis), -radius*normal.dot(geometry.x_direction)/axial,
                         -radius*normal.dot(geometry.y_direction)/axial))
    dc, da, db = (b-a for a, b in zip(*sections))
    if abs(dc) - math.hypot(da, db) <= tol:
        raise CadQueryConversionError('cylinder section boundaries intersect or touch',
                                      code='geometry.analytic_trim_mismatch')


def _check_seam_boundaries(converter, result, surface, boundaries):
    from OCP.BRep import BRep_Tool
    from OCP.BRepLib import BRepLib
    from OCP.Geom import Geom_CylindricalSurface, Geom_RectangularTrimmedSurface
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from ._supported_curves import deviation

    support = BRep_Tool.Surface_s(result.wrapped)
    if isinstance(support, Geom_RectangularTrimmedSurface):
        support = support.BasisSurface()
    def frame(s):
        return (s.Radius(), s.Location().Coord(), s.Position().XDirection().Coord(),
                s.Position().YDirection().Coord(), s.Axis().Direction().Coord())
    if not isinstance(support, Geom_CylindricalSurface) or frame(support) != frame(surface):
        raise ValueError('cylinder support was changed')
    # The seam tool creates a new edge at its default tolerance and may split
    # circles with unset parameter flags. Recompute generated UV clocks and
    # their tolerances, then independently qualify every 3D curve below.
    # This path has no saved pcurves to refit.
    BRepLib.SameParameter_s(result.wrapped, converter.tolerance, True)
    spans = [[] for _ in boundaries]
    maximum = 0.
    seams = []
    # OCCT may round the scaled source precision (e.g. 1e-6 * 10) by one ulp.
    tolerance_limit = math.nextafter(converter.tolerance, math.inf)
    for edge in result.Edges():
        curve = BRep_Tool.Curve_s(edge.wrapped, 0., 0.)
        first, last = BRep_Tool.Range_s(edge.wrapped)
        seam = BRep_Tool.IsClosed_s(edge.wrapped, result.wrapped)
        if seam:
            seams.append(edge)
        else:
            matches = [i for i, (source, saved, _, _) in enumerate(boundaries)
                       if curve == saved and edge.wrapped.Orientation() == source.wrapped.Orientation()]
            if len(matches) != 1:
                raise ValueError('changed, ambiguous or reversed physical boundary')
            spans[matches[0]].append((first, last))
        uses = (edge.wrapped, TopoDS.Edge_s(edge.wrapped.Reversed())) if seam else (edge.wrapped,)
        for use in uses:
            pc = BRep_Tool.CurveOnSurface_s(use, result.wrapped, 0., 0.)
            if pc is None or BRep_Tool.Range_s(use, result.wrapped) != (first, last):
                raise ValueError('missing pcurve or inconsistent trim clock')
            distance = deviation(curve, pc, surface, first, last)
            if not math.isfinite(distance) or distance > converter.tolerance:
                raise ValueError('seam pcurve exceeds source precision')
            maximum = max(maximum, distance)
        if BRep_Tool.Tolerance_s(edge.wrapped) > tolerance_limit:
            raise ValueError('edge tolerance was enlarged')
    if len(seams) != 1 or len(result.Wires()) != 1:
        raise ValueError('expected one chart seam in one wire')
    explorer = TopExp_Explorer(result.wrapped, TopAbs_EDGE)
    senses = []
    while explorer.More():
        use = explorer.Current()
        if use.IsSame(seams[0].wrapped):
            senses.append(use.Orientation())
        explorer.Next()
    if len(senses) != 2 or senses[0] == senses[1]:
        raise ValueError('chart seam must have two opposite uses')
    for (_, curve, first, last), intervals in zip(boundaries, spans):
        epsilon = max(1e-12, 64 * math.ulp(max(1., abs(first), abs(last))))
        if curve.IsPeriodic():
            period = curve.Period()
            normalized = []
            for start, end in intervals:
                length = end - start
                if length <= 0 or length > period + epsilon:
                    raise ValueError('invalid periodic trim length')
                start = first + (start - first) % period
                if first + period - start <= epsilon:
                    start = first
                end = start + length
                if end > first + period + epsilon:
                    normalized.extend(((start, first + period), (first, end - period)))
                else:
                    normalized.append((start, min(end, first + period)))
            intervals = normalized
        cursor = first
        for start, end in sorted(intervals):
            if abs(start - cursor) > epsilon or end <= start:
                raise ValueError('source trim was omitted, overlapped or extended')
            cursor = end
        if not intervals or abs(cursor - last) > epsilon:
            raise ValueError('incomplete source trim coverage')
    vertices = result.Vertices()
    for source, _, _, _ in boundaries:
        for vertex in source.Vertices():
            if not any(math.dist(vertex.toTuple(), other.toTuple()) <= converter.tolerance for other in vertices):
                raise ValueError('source vertex was removed or moved')
    if any(BRep_Tool.Tolerance_s(v.wrapped) > tolerance_limit for v in vertices):
        raise ValueError('vertex tolerance was enlarged')
    if not result.isValid() or not math.isfinite(result.Area()) or result.Area() <= 0:
        raise ValueError('cylinder seam did not produce a valid bounded face')
    return dict(source_edges=len(boundaries), physical_edges=sum(map(len, spans)),
                seam_edges=len(seams), max_deviation_mm=maximum, tolerance_mm=converter.tolerance)


# ---------------------------------------------------------------- periodic faces without a chart seam

# The seam keeps this multiple of the face's largest edge/vertex tolerance away from every source vertex and
# every hole, so that no split point of the seam can be merged with, or mistaken for, a source vertex.
_SEAM_CLEARANCE_FACTOR = 100.0
# Exact-inverse UV samples per edge; the count doubles until the UV curve meets the edge's own tolerance.
_UV_SAMPLES = (33, 65, 129, 257, 513, 1025, 2049)


class _Refusal(Exception):
    """A periodic face that the qualified chart seam does not cover: the face stays geometry.face_invalid."""


def periodic_seam_face(converter, face, loops, wires, geometry, placement, *, after_heal=False):
    """Add a chart seam to a circular cylinder/cone face whose boundary has none.

    Qualifies only two loops winding once around the axis in opposite senses plus any number of non-winding
    holes in the band's sense. The UV curves come from the support's exact inverse of each unchanged 3D
    boundary curve (never from a projection or a fit) and are checked against it within the edge's own
    tolerance. The seam lies at a parameter clear of every source vertex and hole; OCCT's ComposeShell inserts
    it on a one-patch grid of the unchanged support. The result keeps every source curve and trim interval,
    all in their source orientation or (a reversed-sense source) all turned into the chart's own sense, as an
    unseamed face leaves BRepLib_MakeFace; the caller applies the face sense. Anything else returns None (the
    caller's geometry.face_invalid) and is recorded in ``converter.periodic_seam_refusals``.

    A face bounded by stored fits is first left to ``_heal_approximated_face`` exactly as before: copies of its
    wires wait in ``converter._deferred_periodic_seams`` until ``_face`` calls ``periodic_seam_after_heal``.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.Geom import Geom_ConicalSurface, Geom_CylindricalSurface
    from OCP.TopoDS import TopoDS

    def refuse(reason):
        converter.periodic_seam_refusals.append(dict(face=face.index, reason=reason, after_heal=after_heal))
        return None

    def copied(sources):
        # The seam is built on copies: the caller's wires are never touched, whatever the outcome.
        return tuple(converter.cq.Wire(TopoDS.Wire_s(BRepBuilderAPI_Copy(wire.wrapped, False, False).Shape()))
                     for wire in sources)

    if not isinstance(geometry.surface, (Geom_CylindricalSurface, Geom_ConicalSurface)):
        return refuse('support is not a circular cylinder or cone')
    # Fail closed: an OCCT failure or a conversion error anywhere in the seam code is a refusal of the seam
    # path, so the face stays geometry.face_invalid (never another code, never a later 'could not sew').
    try:
        source_surface = converter._resolve_geometry(face.surface)
        if not after_heal and converter._face_uses_approximation(source_surface, loops):
            converter._deferred_periodic_seams[face.index] = dict(
                loops=loops, wires=copied(wires), geometry=geometry, placement=placement,
                flip=face.reversed != (source_surface.cos_half_angle < 0))
            return None
        if placement.orientation_sign <= 0:
            return refuse('mirrored placement')
        reason = _unqualified_source(converter, face, loops, wires, placement)
        if reason is not None:
            return refuse(reason)
        # A deferred face already holds the copies taken before the heal ran (and used nowhere else).
        result, record = _compose_periodic_seam(converter, wires if after_heal else copied(wires), geometry)
        converter.periodic_seam_faces.append(dict(face=face.index, loops=[loop.index for loop in loops],
            edges=[co.edge.index for loop in loops for co in converter._coedges(loop)],
            method=('periodic chart seam clear of every source vertex and hole; exact-inverse UV curves; '
                    'unchanged oriented 3D curves and complete trim intervals'),
            after_heal=after_heal, **record))
        return result
    except _Refusal as refusal:
        return refuse(str(refusal))
    except Exception as error:  # OCP Standard_Failure, CadQueryConversionError from a deviation check, ...
        return refuse(f'{type(error).__name__}: {error}')


def periodic_seam_after_heal(converter, face, deferred, healed):
    """The deferred seam of a stored-fit face that ``_heal_approximated_face`` left invalid, in the face sense."""
    result = periodic_seam_face(converter, face, deferred['loops'], deferred['wires'], deferred['geometry'],
                                deferred['placement'], after_heal=True)
    if result is None:
        return healed
    return converter._reverse_face(result) if deferred['flip'] else result


def _unqualified_source(converter, face, loops, wires, placement):
    """``cylinder_seam_face``'s source conditions; the reason one fails, else None."""
    from .model import EdgeEntity, VertexEntity
    source_surface = converter._resolve_geometry(face.surface)
    if any(r is not None and (r.lower is not None or r.upper is not None)
           for r in (source_surface.u_range, source_surface.v_range)):
        return 'saved chart range'
    coedges = [co for loop in loops for co in converter._coedges(loop)]
    if any(co.raw.type_name != 'coedge' or not co.pcurve.is_null for co in coedges):
        return 'saved UV curve or tolerant coedge'
    source_edges = [converter.model.resolve(co.edge) for co in coedges]
    if any(not isinstance(e, EdgeEntity) or e.raw.type_name != 'edge' for e in source_edges):
        return 'tolerant or unresolved edge record'
    for source_edge in source_edges:
        vertices = [converter.model.resolve(v) for v in (source_edge.start_vertex, source_edge.end_vertex)]
        if any(not isinstance(v, VertexEntity) or v.raw.type_name != 'vertex' for v in vertices):
            return 'tolerant or unresolved vertex record'
    boundary_points = [v.toTuple() for wire in wires for v in wire.Vertices()]
    for source_edge in source_edges:
        for vertex in (source_edge.start_vertex, source_edge.end_vertex):
            point = placement.point(converter._vertex_location(vertex, context='periodic seam source vertex'))
            if not any(math.dist(point, actual) <= converter.tolerance for actual in boundary_points):
                return 'the built boundary does not retain a saved source vertex'
    return None


class _Chart:
    """The exact inverse of an OCCT circular cylinder/cone chart (u = atan2 in the frame, v along the axis)."""

    def __init__(self, surface):
        from OCP.Geom import Geom_ConicalSurface
        position = surface.Position()
        o, x, y, z = position.Location(), position.XDirection(), position.YDirection(), position.Direction()
        self.origin = (o.X(), o.Y(), o.Z())
        self.axes = tuple((d.X(), d.Y(), d.Z()) for d in (x, y, z))
        self.cone = isinstance(surface, Geom_ConicalSurface)
        angle = surface.SemiAngle() if self.cone else 0.
        self.reference = surface.RefRadius() if self.cone else surface.Radius()
        self.sin, self.cos = math.sin(angle), math.cos(angle)

    def inverse(self, point):
        d = (point.X() - self.origin[0], point.Y() - self.origin[1], point.Z() - self.origin[2])
        x, y, z = (sum(a * b for a, b in zip(d, axis)) for axis in self.axes)
        v = z / self.cos
        if self.radius(v) <= 0:
            raise _Refusal('boundary point at or beyond the cone apex')
        return math.atan2(y, x), v

    def radius(self, v):
        return self.reference + v * self.sin


def _uv_curve(curve, first, last, chart, surface, limit):
    """The UV curve of one edge: exact-inverse points at the edge's own 3D parameters, exact when linear,
    otherwise interpolated; accepted only within ``limit`` of the 3D curve. Returns (pcurve, samples, deviation,
    kind) with samples ``(t, u, v)``, u continuous."""
    from OCP.Geom2d import Geom2d_BSplineCurve
    from OCP.Geom2dAPI import Geom2dAPI_Interpolate
    from OCP.TColgp import TColgp_Array1OfPnt2d, TColgp_HArray1OfPnt2d
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_HArray1OfReal
    from OCP.gp import gp_Pnt2d
    from .cadquery import CadQueryConversionError
    from ._supported_curves import deviation
    for count in _UV_SAMPLES:
        samples = []
        for i in range(count):
            t = first + (last - first) * i / (count - 1)
            u, v = chart.inverse(curve.Value(t))
            if samples:
                u += round((samples[-1][1] - u) / (2 * math.pi)) * 2 * math.pi
            samples.append((t, u, v))
        (t0, u0, v0), (t1, u1, v1) = samples[0], samples[-1]
        scale = max(1., abs(u0), abs(u1), abs(v0), abs(v1))
        linear = all(abs(u - u0 - (u1 - u0) * (t - t0) / (t1 - t0)) <= 1e-12 * scale
                     and abs(v - v0 - (v1 - v0) * (t - t0) / (t1 - t0)) <= 1e-12 * scale for t, u, v in samples)
        if linear:
            poles = TColgp_Array1OfPnt2d(1, 2)
            poles.SetValue(1, gp_Pnt2d(u0, v0))
            poles.SetValue(2, gp_Pnt2d(u1, v1))
            knots = TColStd_Array1OfReal(1, 2)
            knots.SetValue(1, first)
            knots.SetValue(2, last)
            multiplicities = TColStd_Array1OfInteger(1, 2)
            multiplicities.SetValue(1, 2)
            multiplicities.SetValue(2, 2)
            pcurve = Geom2d_BSplineCurve(poles, knots, multiplicities, 1)
        else:
            points = TColgp_HArray1OfPnt2d(1, count)
            parameters = TColStd_HArray1OfReal(1, count)
            for i, (t, u, v) in enumerate(samples, 1):
                points.SetValue(i, gp_Pnt2d(u, v))
                parameters.SetValue(i, t)
            interpolation = Geom2dAPI_Interpolate(points, parameters, False, 1e-12)
            interpolation.Perform()
            if not interpolation.IsDone():
                continue
            pcurve = interpolation.Curve()
        try:
            distance = deviation(curve, pcurve, surface, first, last)
        except CadQueryConversionError:
            continue
        if distance <= limit:
            return pcurve, samples, distance, 'linear' if linear else f'interpolated {count}'
        if linear:
            break
    raise _Refusal('exact-inverse UV curve exceeds the edge tolerance')


def _free_parameter(intervals):
    """The middle of the largest gap of [0, 2 pi) left by ``intervals`` (any real bounds), or None."""
    two_pi = 2 * math.pi
    pieces = []
    for low, high in intervals:
        if high - low >= two_pi:
            return None
        start = low % two_pi
        end = start + (high - low)
        if end > two_pi:
            pieces.extend(((start, two_pi), (0., end - two_pi)))
        else:
            pieces.append((start, end))
    if not pieces:
        return None
    pieces.sort()
    merged = [list(pieces[0])]
    for low, high in pieces[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    gaps = [(merged[i + 1][0] - merged[i][1], merged[i][1]) for i in range(len(merged) - 1)]
    gaps.append((merged[0][0] + two_pi - merged[-1][1], merged[-1][1]))
    width, start = max(gaps)
    if width <= 0:
        return None
    return (start + width / 2) % two_pi


def _compose_periodic_seam(converter, wires, geometry):
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.Geom import Geom_RectangularTrimmedSurface
    from OCP.Precision import Precision
    from OCP.ShapeBuild import ShapeBuild_ReShape
    from OCP.ShapeExtend import ShapeExtend_CompositeSurface
    from OCP.ShapeFix import ShapeFix_ComposeShell
    from OCP.TColGeom import TColGeom_HArray2OfSurface
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Face
    from OCP.gp import gp_Vec2d

    two_pi = 2 * math.pi
    surface = geometry.surface
    tolerance = converter.tolerance
    chart = _Chart(surface)
    location = TopLoc_Location()
    builder = BRep_Builder()

    def explore(shape, kind):
        found = []
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            found.append(explorer.Current())
            explorer.Next()
        return found

    loops, sources, vertex_tolerances, uv_kinds = [], [], [], []
    for wire in wires:
        uses = []
        explorer = BRepTools_WireExplorer(wire.wrapped)
        while explorer.More():
            uses.append(TopoDS.Edge_s(explorer.Current()))
            explorer.Next()
        if not uses or len(uses) != len(explore(wire.wrapped, TopAbs_EDGE)):
            raise _Refusal('a boundary wire is not one connected chain')
        loop = dict(uses=[], vertices=[], samples=[])
        previous = None
        for edge in uses:
            if BRep_Tool.Degenerated_s(edge):
                raise _Refusal('degenerated boundary edge')
            curve = BRep_Tool.Curve_s(edge, 0., 0.)
            if curve is None:
                raise _Refusal('boundary edge without a 3D curve')
            first, last = BRep_Tool.Range_s(edge)
            limit = max(tolerance, BRep_Tool.Tolerance_s(edge))
            pcurve, samples, distance, kind = _uv_curve(curve, first, last, chart, surface, limit)
            uv_kinds.append(kind)
            traversal = samples[::-1] if edge.Orientation() == TopAbs_REVERSED else samples
            shift = 0.
            if previous is not None:
                shift = round((previous[0][0] - traversal[0][1]) / two_pi) * two_pi
                gap = surface.Value(*previous[0]).Distance(surface.Value(traversal[0][1] + shift, traversal[0][2]))
                if gap > previous[1] + limit + tolerance:
                    raise _Refusal('boundary UV chain is not continuous')
            pcurve.Translate(gp_Vec2d(shift, 0.))
            points = [(u + shift, v) for _, u, v in traversal]
            loop['uses'].append(dict(edge=edge, pcurve=pcurve, first=first, last=last, limit=limit))
            loop['vertices'].append(points[0])
            loop['samples'].extend(points)
            sources.append(dict(edge=edge, curve=curve, first=first, last=last, limit=limit,
                                orientation=edge.Orientation(), distance=distance))
            previous = (points[-1], limit)
        start, end = loop['vertices'][0], previous[0]
        closing = surface.Value(*start).Distance(surface.Value(*end))
        turns = round((end[0] - start[0]) / two_pi)
        if (closing > previous[1] + loop['uses'][0]['limit'] + tolerance or turns not in (-1, 0, 1)
                or abs(end[0] - start[0] - turns * two_pi) > math.pi / 2):
            raise _Refusal('a boundary loop does not close in the chart')
        loop['winding'] = turns
        us = [u for u, _ in loop['samples']]
        vs = [v for _, v in loop['samples']]
        loop['span'] = (min(us), max(us))
        loop['mean_v'] = sum(vs) / len(vs)
        polygon = loop['samples']
        loop['area'] = sum(polygon[i - 1][0] * polygon[i][1] - polygon[i][0] * polygon[i - 1][1]
                           for i in range(len(polygon))) / 2
        loops.append(loop)
        vertex_tolerances.extend(BRep_Tool.Tolerance_s(TopoDS.Vertex_s(v))
                                 for v in explore(wire.wrapped, TopAbs_VERTEX))

    rims = [loop for loop in loops if loop['winding']]
    holes = [loop for loop in loops if not loop['winding']]
    if len(rims) != 2:
        raise _Refusal(f'{len(rims)} loops wind around the axis; a band needs two')
    if rims[0]['winding'] == rims[1]['winding']:
        raise _Refusal('both rim windings have the same sense (winding mismatch)')
    lower = min(rims, key=lambda loop: loop['mean_v'])
    natural = lower['winding'] > 0  # OCCT's chart orientation: the lower rim runs in +u
    for hole in holes:
        if hole['area'] == 0 or (hole['area'] > 0) == natural:
            raise _Refusal('a hole is oriented against the band sense')

    widest = max([tolerance, *vertex_tolerances, *(s['limit'] for s in sources)])
    clearance = _SEAM_CLEARANCE_FACTOR * widest
    forbidden = []
    for loop in loops:
        for u, v in loop['vertices']:
            half = clearance / chart.radius(v)
            forbidden.append((u - half, u + half))
    for hole in holes:
        half = clearance / min(chart.radius(v) for _, v in hole['samples'])
        forbidden.append((hole['span'][0] - half, hole['span'][1] + half))
    theta = _free_parameter(forbidden)
    if theta is None:
        raise _Refusal('no seam parameter clear of every source vertex and hole')

    for loop in loops:
        anchor = loop['span'][0] if not loop['winding'] else loop['vertices'][0][0]
        shift = theta + (anchor - theta) % two_pi - anchor
        if not loop['winding'] and loop['span'][1] + shift >= theta + two_pi:
            raise _Refusal('a hole crosses the seam parameter')
        for use in loop['uses']:
            use['pcurve'].Translate(gp_Vec2d(shift, 0.))
            edge = use['edge']
            builder.UpdateEdge(edge, use['pcurve'], surface, location, BRep_Tool.Tolerance_s(edge))
            builder.Range(edge, surface, location, use['first'], use['last'])
            builder.SameRange(edge, True)
            builder.SameParameter(edge, True)

    vs = [v for loop in loops for _, v in loop['samples']]
    low, high = min(vs), max(vs)
    margin = .01 * (high - low) + 10 * widest
    if chart.sin:
        apex = -chart.reference / chart.sin
        margin = min(margin, (low - apex) / 2 if apex < low else (apex - high) / 2)
    face = TopoDS_Face()
    builder.MakeFace(face, surface, location, tolerance)
    for wire in wires:
        builder.Add(face, wire.wrapped if natural else wire.wrapped.Reversed())
    grid = TColGeom_HArray2OfSurface(1, 1, 1, 1)
    grid.SetValue(1, 1, Geom_RectangularTrimmedSurface(surface, theta, theta + two_pi, low - margin, high + margin))
    compose = ShapeFix_ComposeShell()
    compose.Init(ShapeExtend_CompositeSurface(grid), location, face, Precision.Confusion_s())
    try:
        compose.ClosedMode = True
    except Exception as error:  # OCP binding without a settable flag
        raise _Refusal(f'ComposeShell closed mode unavailable: {type(error).__name__}') from error
    if compose.ClosedMode is not True:
        raise _Refusal('ComposeShell closed mode unavailable')
    compose.SetContext(ShapeBuild_ReShape())
    compose.SetMaxTolerance(widest)
    compose.Perform()
    faces = explore(compose.Result(), TopAbs_FACE)
    if len(faces) != 1:
        raise _Refusal(f'the seam composition gave {len(faces)} faces')
    # The face keeps the chart's own sense, as BRepLib_MakeFace's inside check leaves an unseamed face: the
    # caller applies the face sense (``_cone_face``: face.reversed against the sign of cos).
    result = converter.cq.Face(TopoDS.Face_s(faces[0]))
    try:
        _settle_seam(converter, result, surface, theta)
        proof = _check_periodic_seam(converter, result, surface, sources, natural, len(holes), clearance,
                                     widest, [v for wire in wires for v in wire.Vertices()])
    except ValueError as error:
        raise _Refusal(f'seam verification: {error}') from error
    proof.update(surface='cone' if chart.cone else 'cylinder', holes=len(holes), seam_parameter=theta,
                 seam_clearance_mm=clearance, loops_reversed_to_chart_sense=not natural,
                 uv_curves=sorted(set(uv_kinds)))
    return result, proof


def _settle_seam(converter, result, surface, theta):
    """Give the generated seam its exact 3D iso-line and the source precision, and align split trim clocks.

    ComposeShell builds the seam's 3D curve by approximation at OCCT's default 1e-5, computes the split
    parameters of a piece separately in 3D and 2D and leaves the pieces' same-range flag unset. The seam's two
    UV uses must be the iso-line u = theta; its 3D curve becomes that exact surface line (same parameter as the
    UV line) and its tolerance the source precision. A piece whose 3D clock restarted one period away takes
    the equivalent values of its UV clock, a piece whose UV range differs from its 3D range by rounding only
    takes the 3D range, and a piece with one range is flagged same-range and same-parameter. Every value is
    then checked independently by ``_check_periodic_seam`` and BRepCheck.
    """
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.Geom import Geom_Line
    from OCP.Geom2d import Geom2d_Line
    from OCP.ShapeFix import ShapeFix_ShapeTolerance
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Vec

    builder = BRep_Builder()
    face = result.wrapped
    seams, pieces = [], []
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        (seams if BRep_Tool.IsClosed_s(edge, face) else pieces).append(edge)
        explorer.Next()
    for edge in pieces:
        first, last = BRep_Tool.Range_s(edge)
        low, high = BRep_Tool.Range_s(edge, face)
        epsilon = max(1e-12, 64 * math.ulp(max(1., abs(first), abs(last), abs(low), abs(high))))
        curve = BRep_Tool.Curve_s(edge, 0., 0.)
        if curve is not None and curve.IsPeriodic():
            # A split periodic 3D curve may restart its clock one period away from the UV clock; the UV
            # curve keeps the source edge's parameters, so the 3D range takes the same (equivalent) values.
            turns = round((low - first) / curve.Period())
            shift = turns * curve.Period()
            if turns and all(abs(a - b - shift) <= max(epsilon, 1e-9 * max(1., abs(a)))
                             for a, b in ((low, first), (high, last))):
                first, last = first + shift, last + shift
                builder.Range(edge, first, last, True)
        if (low, high) != (first, last) and curve is not None and all(
                abs(a - b) <= max(epsilon, 1e-9 * max(1., abs(a)))
                and curve.Value(a).Distance(curve.Value(b)) <= 1e-3 * converter.tolerance
                for a, b in ((low, first), (high, last))):
            # The split computed the UV and 3D clocks of the same point separately (rounding only, the 3D
            # points within 1e-3 of the source precision): the UV curve takes the 3D clock.
            builder.Range(edge, face, first, last)
        if BRep_Tool.Range_s(edge, face) == (first, last):
            # One clock again: the flags say so, and the deviation is checked afterwards (here and by BRepCheck).
            if not BRep_Tool.SameRange_s(edge):
                builder.SameRange(edge, True)
            if not BRep_Tool.SameParameter_s(edge):
                builder.SameParameter(edge, True)
    if len(seams) != 2 or not seams[0].IsSame(seams[1]):
        return
    seam = seams[0]
    first, last = BRep_Tool.Range_s(seam, face)
    lines = [BRep_Tool.CurveOnSurface_s(use, face, 0., 0.) for use in seams]
    if not all(isinstance(line, Geom2d_Line) for line in lines):
        raise ValueError('the generated seam is not an iso-line of the support')
    for line in lines:
        origin, direction = line.Location(), line.Direction()
        if (abs(direction.X()) > 1e-12 or abs(abs(direction.Y()) - 1) > 1e-12
                or abs((origin.X() - theta + math.pi) % (2 * math.pi) - math.pi) > 1e-9):
            raise ValueError('the generated seam is not the iso-line of the seam parameter')
    line = lines[0]
    origin, sense = line.Location(), line.Direction().Y()
    start = surface.Value(origin.X(), origin.Y())
    along = surface.Value(origin.X(), origin.Y() + sense)
    axis = gp_Ax1(gp_Pnt(start.X(), start.Y(), start.Z()),
                  gp_Dir(gp_Vec(start, along)))
    builder.UpdateEdge(seam, Geom_Line(axis), TopLoc_Location(), converter.tolerance)
    builder.Range(seam, first, last, True)
    ShapeFix_ShapeTolerance().SetTolerance(seam, converter.tolerance, TopAbs_EDGE)


def _check_periodic_seam(converter, result, surface, sources, natural, holes, clearance, widest, source_vertices):
    from OCP.BRep import BRep_Tool
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepLib import BRepLib
    from OCP.GProp import GProp_GProps
    from OCP.Geom import Geom_ConicalSurface, Geom_RectangularTrimmedSurface
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from ._supported_curves import deviation

    tolerance = converter.tolerance
    support = BRep_Tool.Surface_s(result.wrapped)
    if isinstance(support, Geom_RectangularTrimmedSurface):
        support = support.BasisSurface()

    def frame(s):
        p = s.Position()
        shape = (s.SemiAngle(), s.RefRadius()) if isinstance(s, Geom_ConicalSurface) else (s.Radius(),)
        return (type(s).__name__, shape) + tuple((d.X(), d.Y(), d.Z()) for d in (
            p.Location(), p.XDirection(), p.YDirection(), p.Direction()))
    if frame(support) != frame(surface):
        raise ValueError('support was changed')
    spans = [[] for _ in sources]
    seams = []
    maximum = 0.
    explorer = TopExp_Explorer(result.wrapped, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        explorer.Next()
        if BRep_Tool.IsClosed_s(edge, result.wrapped):
            seams.append(edge)
            continue
        curve = BRep_Tool.Curve_s(edge, 0., 0.)
        # Every physical use keeps its source orientation, or all of them are turned (a reversed-sense source
        # in the chart's own sense); a mixture never matches.
        matches = [i for i, s in enumerate(sources)
                   if curve == s['curve'] and (edge.Orientation() == s['orientation']) == natural]
        if len(matches) != 1:
            raise ValueError('changed, ambiguous or reversed physical boundary')
        source = sources[matches[0]]
        first, last = BRep_Tool.Range_s(edge)
        spans[matches[0]].append((first, last))
        pc = BRep_Tool.CurveOnSurface_s(edge, result.wrapped, 0., 0.)
        if pc is None or BRep_Tool.Range_s(edge, result.wrapped) != (first, last):
            raise ValueError(f'missing UV curve or inconsistent trim clock ({pc is None}, '
                             f'{BRep_Tool.Range_s(edge, result.wrapped)} != {(first, last)})')
        distance = deviation(curve, pc, surface, first, last)
        if not math.isfinite(distance) or distance > source['limit']:
            raise ValueError('UV curve exceeds the source edge tolerance')
        maximum = max(maximum, distance)
        if BRep_Tool.Tolerance_s(edge) > math.nextafter(source['limit'], math.inf):
            raise ValueError('edge tolerance was enlarged')
    if len(seams) != 2 or not seams[0].IsSame(seams[1]) or seams[0].Orientation() == seams[1].Orientation():
        raise ValueError('expected one chart seam with two opposite uses')
    seam = seams[0]
    if BRep_Tool.Curve_s(seam, 0., 0.) is None:
        BRepLib.BuildCurve3d_s(seam, tolerance)
    seam_curve = BRep_Tool.Curve_s(seam, 0., 0.)
    first, last = BRep_Tool.Range_s(seam)
    for use in seams:
        pc = BRep_Tool.CurveOnSurface_s(use, result.wrapped, 0., 0.)
        if seam_curve is None or pc is None or BRep_Tool.Range_s(use, result.wrapped) != (first, last):
            raise ValueError('seam without a 3D curve, a UV curve or a consistent trim clock')
        distance = deviation(seam_curve, pc, surface, first, last)
        if not math.isfinite(distance) or distance > tolerance:
            raise ValueError('seam UV curve exceeds source precision')
        maximum = max(maximum, distance)
    if BRep_Tool.Tolerance_s(seam) > math.nextafter(tolerance, math.inf):
        raise ValueError(f'seam tolerance {BRep_Tool.Tolerance_s(seam)!r} exceeds source precision '
                         f'(curve {seam_curve.DynamicType().Name()}, max deviation {maximum!r})')
    wires = TopExp_Explorer(result.wrapped, TopAbs_WIRE)
    count = 0
    while wires.More():
        count += 1
        wires.Next()
    if count != 1 + holes:
        raise ValueError(f'{count} wires for two rims and {holes} holes: a hole was merged or split')
    for source, intervals in zip(sources, spans):
        _check_complete_trim(source['curve'], source['first'], source['last'], intervals)
    vertices = result.Vertices()
    nearest = math.inf
    for vertex in source_vertices:
        if not any(math.dist(vertex.toTuple(), other.toTuple()) <= tolerance for other in vertices):
            raise ValueError('source vertex was removed or moved')
        distance = BRepExtrema_DistShapeShape(vertex.wrapped, seam)
        if not distance.IsDone() or distance.Value() < clearance:
            raise ValueError('the seam is not clear of a source vertex')
        nearest = min(nearest, distance.Value())
    if any(BRep_Tool.Tolerance_s(v.wrapped) > math.nextafter(widest, math.inf) for v in vertices):
        raise ValueError('vertex tolerance was enlarged')
    # Adaptive integration: the fixed-order area of a face bounded by many-span spline UV curves can be far off.
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(result.wrapped, properties, 1e-9)
    area = properties.Mass()
    valid = result.isValid()
    if not valid or not math.isfinite(area) or area <= 0:
        raise ValueError(f'the seamed face is not a valid bounded face (valid {valid}, area {area!r}; '
                         f'{_check_statuses(result)})')
    return dict(source_edges=len(sources), physical_edges=sum(map(len, spans)), seam_edges=1,
                max_deviation_mm=maximum, tolerance_mm=tolerance, min_vertex_seam_distance_mm=nearest,
                area_mm2=area)


def _check_statuses(result):
    """BRepCheck statuses other than NoError of a face and its sub-shapes (for the refusal reason)."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    analyzer = BRepCheck_Analyzer(result.wrapped)
    found = []
    for kind, label in ((None, 'face'), (TopAbs_WIRE, 'wire'), (TopAbs_EDGE, 'edge'), (TopAbs_VERTEX, 'vertex')):
        shapes = [result.wrapped]
        if kind is not None:
            shapes = []
            explorer = TopExp_Explorer(result.wrapped, kind)
            while explorer.More():
                shapes.append(explorer.Current())
                explorer.Next()
        for shape in shapes:
            try:
                checked = analyzer.Result(shape)
                statuses = [str(s) for s in checked.Status()]
                if kind is not None:  # the statuses of the sub-shape in its contexts (face, wire, edge)
                    checked.InitContextIterator()
                    while checked.MoreShapeInContext():
                        context = checked.ContextualShape().ShapeType()
                        statuses.extend(f'{s} in {str(context).split("_")[-1]}' for s in checked.StatusOnShape())
                        checked.NextShapeInContext()
            except Exception as error:  # a binding detail must not hide the refusal
                statuses = [f'unreadable {type(error).__name__}: {error}']
            bad = [s for s in statuses if 'NoError' not in s]
            if bad and kind == TopAbs_EDGE:
                bad = [f'{s} [{_edge_facts(result, shape)}]' for s in bad]
            found.extend(f'{label} {s}' for s in bad)
    return ', '.join(sorted(set(found))) or 'no BRepCheck status'


def _edge_facts(result, shape):
    """Curve kinds, ranges, tolerance and same-parameter deviation of one edge on the face (refusal detail)."""
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    from ._supported_curves import deviation
    try:
        edge = TopoDS.Edge_s(shape)
        curve = BRep_Tool.Curve_s(edge, 0., 0.)
        pc = BRep_Tool.CurveOnSurface_s(edge, result.wrapped, 0., 0.)
        first, last = BRep_Tool.Range_s(edge)
        support = BRep_Tool.Surface_s(result.wrapped)
        distance = deviation(curve, pc, support, first, last) if curve is not None and pc is not None else None
        return (f'{curve.DynamicType().Name() if curve is not None else None}/'
                f'{pc.DynamicType().Name() if pc is not None else None} 3d {(first, last)} '
                f'2d {BRep_Tool.Range_s(edge, result.wrapped)} tol {BRep_Tool.Tolerance_s(edge)!r} '
                f'deviation {distance!r} sp {BRep_Tool.SameParameter_s(edge)} sr {BRep_Tool.SameRange_s(edge)}')
    except Exception as error:  # detail only
        return f'unreadable {type(error).__name__}: {error}'


def _check_complete_trim(curve, first, last, intervals):
    """Every piece of one source edge together covers its trim interval exactly once (periodic curves too)."""
    epsilon = max(1e-12, 64 * math.ulp(max(1., abs(first), abs(last))))
    if curve.IsPeriodic():
        period = curve.Period()
        normalized = []
        for start, end in intervals:
            length = end - start
            if length <= 0 or length > period + epsilon:
                raise ValueError('invalid periodic trim length')
            start = first + (start - first) % period
            if first + period - start <= epsilon:
                start = first
            end = start + length
            if end > first + period + epsilon:
                normalized.extend(((start, first + period), (first, end - period)))
            else:
                normalized.append((start, min(end, first + period)))
        intervals = normalized
    cursor = first
    for start, end in sorted(intervals):
        if abs(start - cursor) > epsilon or end <= start:
            raise ValueError('source trim was omitted, overlapped or extended')
        cursor = end
    if not intervals or abs(cursor - last) > epsilon:
        raise ValueError('incomplete source trim coverage')
