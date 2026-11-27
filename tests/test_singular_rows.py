"""A stored-fit face whose fit collapses a band of end pole rows at one end.

ASM saves `rb_blend_spl_sur` fits whose end pole rows collapse to the apex of a three-sided face, flagged by the
fit's singularity code (1 = low, 2 = high end). The face's two side pcurves meet the apex row at u = 0 and u = 1, a
UV gap of the whole u range. The existing singular-point path (`_singular_segment`) refuses it because the surface
opens too slowly: the iso 5 % into the surface is shorter than 10 x the join limit. These tests pin the fallback
that closes such a face with one degenerated edge on the collapsed (knot-bound) row, and everything it must refuse.

The fixture mirrors medium-block 4B70 face $1081 (gate 0): a degree-3, 5 x 16 stored fit whose pole rows 0 and 1
each collapse (to different points), the next rows spreading by a few micrometres, and a three-coedge loop: the far
row plus the u = 0 and u = 1 iso sides meeting at the apex. Every edge is a stored `int_int_cur` fit of the exact
iso curve; faces are built by the converter from its own edges.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
import unittest
from unittest import mock

from cq_acis import (
    NULL_REF, AcisMetadata, AcisModel, CadQueryConversionError, CadQueryConverter, CoedgeEntity, EdgeEntity,
    EntityRef, FaceEntity, LoopEntity, PointEntity, RawEntity, Vec3, VertexEntity,
)
from cq_acis.cadquery import _Placement

OCP_AVAILABLE = importlib.util.find_spec("OCP") is not None

DEGREE = 3
U_COUNT, V_COUNT = 5, 16
U_KNOTS, U_MULTS = (0.0, 0.5, 1.0), (3, 1, 3)  # ASM form: end multiplicity = degree
V_KNOTS = tuple(i / 13 for i in range(14))
V_MULTS = (3,) + (1,) * 12 + (3,)
LENGTH = 4.4
FIT_1081 = 1.99e-4  # $1081 / $1224 surface fit (mm)
FIT_365 = 6.28e-3   # $365 surface fit (mm)
PLACEMENT = _Placement(None, 1.0)


def raw(index, name):
    return RawEntity(index, name, NULL_REF, None)


def widths(band, scale, power, overrides=None):
    """Half-row width per row, counted from the apex end: ``band`` collapsed rows, then opening slowly."""
    out = [0.0 if d < band else scale * ((d - band + 1) / (V_COUNT - band)) ** power for d in range(V_COUNT)]
    for d, value in (overrides or {}).items():
        out[d] = value
    return out


def pole_rows(end, row_widths):
    """``rows[j][i]``: pole (u index i, v index j); ``row_widths`` is indexed by the distance from the apex end."""
    rows = []
    for j in range(V_COUNT):
        d = j if end == "v_low" else V_COUNT - 1 - j
        y = LENGTH * j / (V_COUNT - 1)
        row = []
        for i in range(U_COUNT):
            x = row_widths[d] * (i / (U_COUNT - 1) - 0.5)
            row.append((x, y, 0.3 * (y / LENGTH) ** 2 + 1.5 * x * x))
        rows.append(row)
    return rows


def surface_record(index, rows, fit, codes, sense=b"\x0b"):
    """A `spline-surface` record whose `rb_blend_spl_sur` scope ends with the stored nubs fit."""
    values = [NULL_REF, sense, b"\x0f", "rb_blend_spl_sur", 22220, "nubs", DEGREE, DEGREE, 0, 0, codes[0], codes[1],
              len(U_KNOTS), len(V_KNOTS)]
    for knot, multiplicity in zip(U_KNOTS + V_KNOTS, U_MULTS + V_MULTS):
        values += [knot, multiplicity]
    for row in rows:
        for pole in row:
            values += list(pole)
    values += [fit, b"\x10", b"\x0b", b"\x0b", b"\x0b", b"\x0b"]
    return RawEntity(index, "spline-surface", NULL_REF, None, tuple(values))


def curve_record(index, poles, knots, multiplicities, fit):
    """An `intcurve-curve` record carrying a stored cubic fit (here: the exact iso curve)."""
    values = [NULL_REF, b"\x0b", b"\x0f", "int_int_cur", 22220, 0, "nubs", DEGREE, 0, len(knots)]
    for knot, multiplicity in zip(knots, multiplicities):
        values += [knot, multiplicity]
    for pole in poles:
        values += list(pole)
    values += [fit, "null_surface", "null_surface", "nullbs", "nullbs", b"\x10", b"\x0b", b"\x0b"]
    return RawEntity(index, "intcurve-curve", NULL_REF, None, tuple(values))


def patch_model(*, end="v_low", band=2, scale=0.35, power=2.0, surface_fit=FIT_1081, curve_fit=FIT_1081,
                codes=None, face_reversed=False, overrides=None, apex_offset=(0.0, 0.0, 0.0), four_sided=False):
    """One face on a stored fit; returns (model, pole rows).

    Three-sided (default): the far row and the two iso sides meet at the apex on the collapsed ``end`` row; the
    loop runs counter-clockwise in UV, or clockwise with reversed coedges when ``face_reversed``. ``four_sided``:
    the whole patch boundary (no apex).
    """
    rows = pole_rows(end, widths(band, scale, power, overrides))
    if codes is None:
        codes = (0, 1 if end == "v_low" else 2)
    column = lambda i: [rows[j][i] for j in range(V_COUNT)]
    first, last = column(0), column(U_COUNT - 1)
    if four_sided:
        points = {"P00": rows[0][0], "P10": rows[0][-1], "P11": rows[-1][-1], "P01": rows[-1][0]}
        sides = [(rows[0], U_KNOTS, U_MULTS, "P00", "P10"), (last, V_KNOTS, V_MULTS, "P10", "P11"),
                 (rows[-1][::-1], U_KNOTS, U_MULTS, "P11", "P01"), (first[::-1], V_KNOTS, V_MULTS, "P01", "P00")]
    elif end == "v_low":
        apex = rows[0][0]
        points = {"A": tuple(a + b for a, b in zip(apex, apex_offset)), "B": rows[-1][0], "C": rows[-1][-1]}
        sides = [(rows[-1][::-1], U_KNOTS, U_MULTS, "C", "B"), (first[::-1], V_KNOTS, V_MULTS, "B", "A"),
                 (last, V_KNOTS, V_MULTS, "A", "C")]
    else:
        apex = rows[-1][0]
        points = {"A": tuple(a + b for a, b in zip(apex, apex_offset)), "B": rows[0][0], "C": rows[0][-1]}
        sides = [(rows[0], U_KNOTS, U_MULTS, "B", "C"), (last, V_KNOTS, V_MULTS, "C", "A"),
                 (first[::-1], V_KNOTS, V_MULTS, "A", "B")]
    n = len(sides)
    names = list(points)
    vertex_index = {name: 3 + 3 * n + 2 * k for k, name in enumerate(names)}
    order = list(range(n))[::-1] if face_reversed else list(range(n))
    coedge_of = {side: 3 + 3 * side for side in range(n)}
    entities = [
        FaceEntity(raw(0, "face"), NULL_REF, NULL_REF, EntityRef(2), NULL_REF, NULL_REF, EntityRef(1),
                   face_reversed, False, None),
        surface_record(1, rows, surface_fit, codes),
        LoopEntity(raw(2, "loop"), NULL_REF, NULL_REF, EntityRef(coedge_of[order[0]]), EntityRef(0)),
    ]
    for side, (poles, knots, multiplicities, start, stop) in enumerate(sides):
        k = order.index(side)
        base = coedge_of[side]
        entities += [
            CoedgeEntity(raw(base, "coedge"), NULL_REF, EntityRef(coedge_of[order[(k + 1) % n]]),
                         EntityRef(coedge_of[order[(k - 1) % n]]), NULL_REF, EntityRef(base + 1), face_reversed,
                         EntityRef(2), NULL_REF),
            EdgeEntity(raw(base + 1, "edge"), NULL_REF, EntityRef(vertex_index[start]), 0.0,
                       EntityRef(vertex_index[stop]), 1.0, EntityRef(base), EntityRef(base + 2), False, None),
            curve_record(base + 2, poles, knots, multiplicities, curve_fit),
        ]
    for name in names:
        index = vertex_index[name]
        edge = next(coedge_of[s] + 1 for s, side in enumerate(sides) if name in side[3:])
        entities += [VertexEntity(raw(index, "vertex"), NULL_REF, EntityRef(edge), EntityRef(index + 1)),
                     PointEntity(raw(index + 1, "point"), NULL_REF, Vec3(*points[name]))]
    return AcisModel(AcisMetadata(units_mm=1.0, resabs=1e-6), tuple(entities)), rows


def reference_surface(rows):
    """The fit as an independent OCCT surface (full multiplicities)."""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
    from OCP.gp import gp_Pnt

    def array(values, cls):
        result = cls(1, len(values))
        for i, value in enumerate(values, 1):
            result.SetValue(i, value)
        return result

    poles = TColgp_Array2OfPnt(1, U_COUNT, 1, V_COUNT)
    for j, row in enumerate(rows, 1):
        for i, pole in enumerate(row, 1):
            poles.SetValue(i, j, gp_Pnt(*pole))
    full = lambda m: (m[0] + 1,) + tuple(m[1:-1]) + (m[-1] + 1,)
    return Geom_BSplineSurface(poles, array(U_KNOTS, TColStd_Array1OfReal), array(V_KNOTS, TColStd_Array1OfReal),
                               array(full(U_MULTS), TColStd_Array1OfInteger),
                               array(full(V_MULTS), TColStd_Array1OfInteger), DEGREE, DEGREE)


def gauss_legendre(n):
    nodes, weights = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            step = p1 / dp
            x -= step
            if abs(step) < 1e-16:
                break
        nodes.append(x)
        weights.append(2 / ((1 - x * x) * dp * dp))
    return nodes, weights


def patch_area(surface):
    """Area of the whole patch by Gauss-Legendre quadrature (10 x 10 points per knot span)."""
    from OCP.gp import gp_Pnt, gp_Vec
    nodes, weights = gauss_legendre(10)
    total = 0.0
    for u0, u1 in zip(U_KNOTS, U_KNOTS[1:]):
        for v0, v1 in zip(V_KNOTS, V_KNOTS[1:]):
            for a, wa in zip(nodes, weights):
                u = (u0 + u1) / 2 + (u1 - u0) / 2 * a
                for b, wb in zip(nodes, weights):
                    v = (v0 + v1) / 2 + (v1 - v0) / 2 * b
                    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
                    surface.D1(u, v, p, du, dv)
                    total += wa * wb * du.Crossed(dv).Magnitude() * (u1 - u0) * (v1 - v0) / 4
    return total


def face_area(face):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face.wrapped, props, 1e-9)
    return props.Mass()


def degenerated_edges(face):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        if BRep_Tool.Degenerated_s(edge):
            out.append(edge)
        explorer.Next()
    return out


def degenerated_uv(face, edge):
    """UV ends of a degenerated edge's pcurve on the face's surface."""
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    surface = BRep_Tool.Surface_s(face.wrapped)
    first, last = BRep_Tool.Range_s(edge)
    pcurve = BRep_Tool.CurveOnSurface_s(edge, surface, TopLoc_Location(), 0.0, 0.0)
    a, b = pcurve.Value(first), pcurve.Value(last)
    return (a.X(), a.Y()), (b.X(), b.Y())


def normal_agrees(face, reference, u=0.5, v=0.5, flip=False):
    """Built face normal (face orientation applied) against the fit's own Su x Sv, negated when ``flip``."""
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.gp import gp_Pnt, gp_Vec
    built_point, built = gp_Pnt(), gp_Vec()
    BRepGProp_Face(face.wrapped).Normal(u, v, built_point, built)
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    reference.D1(u, v, p, du, dv)
    expected = du.Crossed(dv)
    if flip:
        expected.Reverse()
    return built.Normalized().Dot(expected.Normalized())


def convert_face(model, converter=None):
    converter = converter or CadQueryConverter(model)
    return converter, converter._face(model.entities[0], PLACEMENT)


@unittest.skipUnless(OCP_AVAILABLE, "needs OCP")
class CollapsedRowFaces(unittest.TestCase):
    def assert_closed_on_row(self, converter, face, rows, *, end, band, face_reversed=False):
        self.assertTrue(face.isValid())
        reference = reference_surface(rows)
        bound = 0.0 if end == "v_low" else 1.0
        (edge,) = degenerated_edges(face)
        start, stop = degenerated_uv(face, edge)
        self.assertEqual((start[1], stop[1]), (bound, bound))  # exactly on the knot-bound row
        self.assertAlmostEqual(abs(stop[0] - start[0]), 1.0, places=9)
        self.assertEqual([(e["method"], e["row"], e["band_rows"]) for e in converter.degenerate_edges],
                         [("collapsed_row_of_stored_fit", end, band)])
        expected = patch_area(reference)
        self.assertLess(abs(face_area(face) - expected), 1e-6 * expected)
        self.assertGreater(normal_agrees(face, reference, flip=face_reversed), 0.999999)
        self.assertEqual(converter.singular_row_refusals, [])
        self.assertEqual(len(converter.pcurve_checks), 3)  # one build's evidence, no duplicates

    def assert_refused(self, model, converter=None):
        converter = converter or CadQueryConverter(model)
        with self.assertRaises(CadQueryConversionError) as caught:
            convert_face(model, converter)
        self.assertEqual(caught.exception.code, "geometry.face_invalid")
        self.assertEqual(len(converter.singular_row_refusals), 1)
        self.assertEqual(converter.degenerate_edges, [])
        self.assertEqual(len(converter.pcurve_checks), 3)
        return converter.singular_row_refusals[0]

    def test_two_row_band_low_closes_with_degenerated_edge(self):
        model, rows = patch_model()
        reference = reference_surface(rows)
        # The fixture is the $1081 case: the 5 % iso is far shorter than 10 x the join limit.
        limit = 1e-6 + 2 * (FIT_1081 + FIT_1081)
        self.assertFalse(CadQueryConverter._singular_segment(reference, (0.0, 0.0), (1.0, 0.0), limit))
        converter, face = convert_face(model)
        self.assert_closed_on_row(converter, face, rows, end="v_low", band=2)

    def test_one_row_band_high_closes(self):
        for face_reversed in (False, True):
            with self.subTest(face_reversed=face_reversed):
                model, rows = patch_model(end="v_high", band=1, power=1.2, surface_fit=FIT_365, curve_fit=1e-4,
                                          face_reversed=face_reversed)
                limit = 1e-6 + 2 * (FIT_365 + 1e-4)
                self.assertFalse(CadQueryConverter._singular_segment(
                    reference_surface(rows), (1.0, 1.0), (0.0, 1.0), limit))
                converter, face = convert_face(model)
                self.assert_closed_on_row(converter, face, rows, end="v_high", band=1, face_reversed=face_reversed)

    def test_degenerated_edge_on_knot_bound(self):
        from cq_acis._singular_rows import row_segment
        overshoot = 1.0 + 4.4e-16
        self.assertEqual(row_segment([2.8e-11, overshoot], [1.0, overshoot], 1, 1.0),
                         ((2.8e-11, 1.0), (1.0, 1.0)))
        self.assertEqual(row_segment([0.0, 0.25], [-3.9e-16, 1.0], 0, 0.0), ((0.0, 0.25), (0.0, 1.0)))
        # End to end: the as-built apex ends overshoot the bound; the degenerated edge still lies on it.
        model, rows = patch_model(end="v_high", band=1, power=1.2, surface_fit=FIT_365, curve_fit=1e-4)
        converter = CadQueryConverter(model)
        original = converter._attach_surface_pcurves

        def overshooting(*args, **kwargs):
            count = len(converter._row_join_candidates)
            result = original(*args, **kwargs)
            for candidate in converter._row_join_candidates[count:]:
                candidate["previous"][1] = overshoot
                candidate["join"][1] = overshoot
            return result

        converter._attach_surface_pcurves = overshooting
        _, face = convert_face(model, converter)
        self.assert_closed_on_row(converter, face, rows, end="v_high", band=1)

    def test_flag_without_collapsed_row_refused(self):
        import cq_acis._singular_rows as rows_module
        # Flag (0, 1) but pole row 0 spreads 50 um. The 50 um row gap lies inside the joined vertex tolerance, so
        # OCCT builds this face on the old path (native exploration, 47ceae4): the fallback is never entered.
        model, _ = patch_model(band=0, scale=0.05, overrides={0: 5e-5, 1: 1e-4})
        with mock.patch.object(rows_module, "close_collapsed_rows", side_effect=AssertionError("entered")):
            converter, face = convert_face(model)
        self.assertTrue(face.isValid())
        self.assertEqual(converter.degenerate_edges, [])
        self.assertEqual(len(converter._row_join_candidates), 1)
        # Offered to the fallback anyway, the flagged end without collapsed poles is refused, the face returned.
        entity = model.entities[0]
        surface = converter._resolve_geometry(entity.surface)
        sentinel = object()
        returned = rows_module.close_collapsed_rows(converter, entity, surface, (model.entities[2],), PLACEMENT,
                                                    sentinel)
        self.assertIs(returned, sentinel)
        (refusal,) = converter.singular_row_refusals
        self.assertIn("collapsed", refusal["reason"])
        self.assertEqual(converter.degenerate_edges, [])
        self.assertEqual(len(converter.pcurve_checks), 3)

    def test_collapsed_row_without_flag_refused(self):
        for codes in ((0, 0), (0, 2), (1, 0), (0, 3)):
            with self.subTest(codes=codes):
                model, _ = patch_model(codes=codes)
                reason = self.assert_refused(model)
                self.assertIn("singularity", reason["reason"])

    def test_flag_parse_mismatch_refused(self):
        with self.subTest("the saved closure disagrees with the Rust reader"):
            model, _ = patch_model()
            converter = CadQueryConverter(model)
            converter._resolve_geometry(model.entities[0].surface)
            converter._approximation_closures[1] = (1, 0)
            reason = self.assert_refused(model, converter)
            self.assertIn("mirror", reason["reason"])
        with self.subTest("the record cannot be read by the mirror"):
            model, _ = patch_model()
            converter = CadQueryConverter(model)
            surface = converter._resolve_geometry(model.entities[0].surface)
            converter._geometry_cache[1] = replace(surface, raw=replace(surface.raw, values=surface.raw.values[:-2]))
            reason = self.assert_refused(model, converter)
            self.assertIn("mirror", reason["reason"])

    def test_apex_vertex_off_pole_refused(self):
        # 60 um off the pole: inside the curve fit (100 um, so edges and the tolerant join pass) but outside
        # tolerance + surface fit (21 um) of the band point.
        model, _ = patch_model(surface_fit=2e-5, curve_fit=1e-4, apex_offset=(0.0, 0.0, 6e-5))
        reason = self.assert_refused(model)
        self.assertIn("apex", reason["reason"])

    def test_unflagged_fit_takes_the_old_path(self):
        import cq_acis._singular_rows as rows_module
        model, rows = patch_model(band=0, scale=1.0, power=1.0, overrides={0: 0.5}, codes=(0, 0), four_sided=True)
        with mock.patch.object(rows_module, "close_collapsed_rows", side_effect=AssertionError("entered")), \
                mock.patch.object(rows_module, "collapsed_band", side_effect=AssertionError("entered")):
            converter, face = convert_face(model)
        self.assertTrue(face.isValid())
        self.assertEqual(converter.degenerate_edges, [])
        self.assertEqual(converter.singular_row_refusals, [])
        expected = patch_area(reference_surface(rows))
        self.assertLess(abs(face_area(face) - expected), 1e-6 * expected)

    def fast_opening(self, codes):
        # One collapsed row and a wide fan: the 5 % iso is long, so the existing singular-point path applies.
        return patch_model(band=1, scale=3.0, power=1.0, surface_fit=2e-4, curve_fit=1e-4, codes=codes)

    def test_existing_singular_point_path_unchanged(self):
        # Characterisation (passes at 47ceae4): the existing path closes the apex of an unflagged fit.
        model, rows = self.fast_opening((0, 0))
        converter, face = convert_face(model)
        self.assertTrue(face.isValid())
        self.assertEqual([e["method"] for e in converter.degenerate_edges], ["singular_point_of_stored_fit"])
        self.assertEqual(len(degenerated_edges(face)), 1)
        expected = patch_area(reference_surface(rows))
        self.assertLess(abs(face_area(face) - expected), 1e-6 * expected)

    def test_flagged_face_that_builds_today_is_untouched(self):
        import cq_acis._singular_rows as rows_module
        unflagged, _ = self.fast_opening((0, 0))
        before, before_face = convert_face(unflagged)
        flagged, _ = self.fast_opening((0, 1))
        with mock.patch.object(rows_module, "close_collapsed_rows", side_effect=AssertionError("entered")), \
                mock.patch.object(rows_module, "collapsed_band", side_effect=AssertionError("entered")):
            converter, face = convert_face(flagged)
        self.assertTrue(face.isValid())
        self.assertEqual(converter.degenerate_edges, before.degenerate_edges)
        self.assertEqual(converter.singular_row_refusals, [])
        self.assertEqual(face_area(face), face_area(before_face))

    def test_occ_error_fails_closed(self):
        from OCP.Standard import Standard_Failure
        import cq_acis._singular_rows as rows_module
        with self.subTest("degenerated edge"):
            model, _ = patch_model()
            converter = CadQueryConverter(model)
            converter._degenerated_edge = mock.Mock(side_effect=Standard_Failure("injected"))
            reason = self.assert_refused(model, converter)
            self.assertIn("Standard_Failure", reason["reason"])
        with self.subTest("band qualification"):
            model, _ = patch_model()
            with mock.patch.object(rows_module, "collapsed_band", side_effect=Standard_Failure("injected")):
                reason = self.assert_refused(model)
            self.assertIn("Standard_Failure", reason["reason"])

    def test_refusal_rolls_back_every_public_record(self):
        """R206 (review M3): lists, the reported dicts and the pcurve scalars come back; only private caches stay."""
        import cq_acis._singular_rows as rows_module
        model, _ = patch_model()
        converter = CadQueryConverter(model)
        converter.subtype_failures[1] = "kept"
        converter.approximation_failures[2] = "kept"
        snapshot = rows_module._snapshot(converter)
        converter.subtype_failures[1] = "changed"
        converter.subtype_failures[3] = "added"
        converter.approximation_failures.clear()
        converter.pcurve_checks.append({"added": True})
        converter.pcurve_count += 5
        converter._geometry_cache["private"] = "stays"
        rows_module._restore(converter, snapshot, 7, True)
        self.assertEqual(converter.subtype_failures, {1: "kept"})
        self.assertEqual(converter.approximation_failures, {2: "kept"})
        self.assertEqual(converter.pcurve_checks, [])
        self.assertEqual(converter.pcurve_count, 0)
        self.assertEqual(converter._geometry_cache["private"], "stays")


@unittest.skipUnless(OCP_AVAILABLE, "needs OCP")
class CollapsedBand(unittest.TestCase):
    def test_band_rows_end_parameter_and_point(self):
        from cq_acis._singular_rows import collapsed_band
        _, rows = patch_model()
        end, band_rows, row, point = collapsed_band(reference_surface(rows), (0, 1), 1e-6)
        self.assertEqual((end, band_rows, row), ("v_low", 2, 0.0))
        self.assertEqual((point.X(), point.Y(), point.Z()), rows[0][0])
        _, rows = patch_model(end="v_high", band=1, power=1.2)
        self.assertEqual(collapsed_band(reference_surface(rows), (0, 2), 1e-6)[:3], ("v_high", 1, 1.0))

    def test_band_needs_the_matching_code_and_an_opening_surface(self):
        from cq_acis._singular_rows import collapsed_band
        _, rows = patch_model()
        surface = reference_surface(rows)
        for codes in (None, (0, 0), (0, 2), (1, 0), (1, 1), (0, 3)):
            with self.subTest(codes=codes):
                self.assertIsNone(collapsed_band(surface, codes, 1e-6))
        # Every row collapsed: the surface is degenerate as a whole.
        _, rows = patch_model(scale=0.0)
        self.assertIsNone(collapsed_band(reference_surface(rows), (0, 1), 1e-6))
        # A row spreading by more than the tolerance is not collapsed.
        _, rows = patch_model(band=0, overrides={0: 5e-5})
        self.assertIsNone(collapsed_band(reference_surface(rows), (0, 1), 1e-6))


if __name__ == "__main__":
    unittest.main()
