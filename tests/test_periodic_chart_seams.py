"""Chart seams for periodic cylinder/cone faces without one (synthetic geometry only).

A periodic face whose boundary has no seam coedge is unorientable for OCCT. The seam has to be added at a
parameter clear of every source vertex and hole, keeping every oriented source curve and trim interval; any face
that does not qualify stays ``geometry.face_invalid``.
"""
from dataclasses import replace
import math
import os
import tempfile
import unittest
from unittest import mock

import cq_acis as a
from cq_acis.cadquery import _Placement, _STORED_FIT_MARGIN
from test_m4_geometry import raw, cone

N, R = a.NULL_REF, a.EntityRef
SQRT_HALF = math.sqrt(.5)


class FaceModel:
    """One face on one cone-surface record; loops are lists of coedge uses."""

    def __init__(self, surface):
        self.entries = [None, replace(surface, raw=raw(1, 'cone-surface'))]
        self.loops = []

    def _reserve(self, count=1):
        start = len(self.entries)
        self.entries.extend([None] * count)
        return start

    def vertex(self, x, y, z):
        i = self._reserve(2)
        self.entries[i] = a.VertexEntity(raw(i, 'vertex'), N, N, R(i + 1))
        self.entries[i + 1] = a.PointEntity(raw(i + 1, 'point'), N, a.Vec3(x, y, z))
        return R(i)

    def point(self, ref):
        return self.entries[ref.index + 1].location

    def circle(self, z, radius):
        i = self._reserve()
        self.entries[i] = a.EllipseCurveEntity(raw(i, 'ellipse-curve'), N, a.Vec3(0, 0, z), a.Vec3(0, 0, 1),
                                               a.Vec3(radius, 0, 0), 1., None)
        return R(i)

    def ellipse(self, center, normal, major, ratio):
        i = self._reserve()
        self.entries[i] = a.EllipseCurveEntity(raw(i, 'ellipse-curve'), N, center, normal, major, ratio, None)
        return R(i)

    def line(self, start, end):
        """A straight curve from vertex ``start`` to vertex ``end``; parameters 0 .. length."""
        p, q = self.point(start), self.point(end)
        i = self._reserve()
        self.entries[i] = a.StraightCurveEntity(raw(i, 'straight-curve'), N, p, (q - p).normalized(), None)
        return R(i), (q - p).magnitude

    def nurbs_section(self, radius, c, ca, cb):
        """The exact rational quadratic NURBS of the plane section z = c + ca*x/r + cb*y/r of the cylinder."""
        unit = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
        poles = tuple(a.Vec3(radius * x, radius * y, c + ca * x + cb * y) for x, y in unit)
        weights = tuple(1. if k % 2 == 0 else SQRT_HALF for k in range(9))
        i = self._reserve()
        self.entries[i] = a.BSplineCurveEntity(raw(i, 'intcurve-curve'), N, 2, (0., .25, .5, .75, 1.),
                                               (3, 2, 2, 2, 3), poles, weights, None, 0.)
        return R(i)

    def stored_fit_circle(self, z, radius, tolerance=1e-3):
        """A raw ``int_int_cur`` record with ASM's stored cubic fit of a circle (four Bezier quadrants, the
        parameter is the angle): not an exact curve, so the face counts as bounded by a stored fit."""
        kappa = 4 / 3 * math.tan(math.pi / 8)
        poles = []
        for k in range(4):
            a0, a1 = k * math.pi / 2, (k + 1) * math.pi / 2
            (c0, s0), (c1, s1) = (math.cos(a0), math.sin(a0)), (math.cos(a1), math.sin(a1))
            quadrant = [(radius * c0, radius * s0), (radius * (c0 - kappa * s0), radius * (s0 + kappa * c0)),
                        (radius * (c1 + kappa * s1), radius * (s1 - kappa * c1)), (radius * c1, radius * s1)]
            poles.extend(quadrant if k == 0 else quadrant[1:])
        knots = [value for k in range(5) for value in (k * math.pi / 2, 3)]
        fit = ('nubs', 3, 0, 5, *knots, *[c for x, y in poles for c in (x, y, z)], tolerance)
        values = (N, b'\x0b', b'\x0f', 'int_int_cur', 22220, 0, *fit,
                  'null_surface', 'null_surface', 'nullbs', 'nullbs', b'\x10', b'\x0b', b'\x0b')
        i = self._reserve()
        self.entries[i] = a.RawEntity(i, 'intcurve-curve', N, None, values)
        return R(i)

    def loop(self, uses):
        """``uses``: (curve, edge start vertex, t0, edge end vertex, t1, coedge reversed) in traversal order."""
        loop_index = self._reserve()
        first = self._reserve(2 * len(uses))
        coedges = [first + 2 * j for j in range(len(uses))]
        for j, (curve, start, t0, end, t1, reverse) in enumerate(uses):
            c = coedges[j]
            self.entries[c] = a.CoedgeEntity(raw(c, 'coedge'), N, R(coedges[(j + 1) % len(uses)]),
                                             R(coedges[j - 1]), N, R(c + 1), reverse, R(loop_index), N)
            self.entries[c + 1] = a.EdgeEntity(raw(c + 1, 'edge'), N, start, t0, end, t1, R(c), curve, False, None)
        self.loops.append((loop_index, coedges[0]))

    def model(self, reversed_face=False):
        for k, (loop_index, coedge) in enumerate(self.loops):
            following = R(self.loops[k + 1][0]) if k + 1 < len(self.loops) else N
            self.entries[loop_index] = a.LoopEntity(raw(loop_index, 'loop'), N, following, R(coedge), R(0))
        self.entries[0] = a.FaceEntity(raw(0, 'face'), N, N, R(self.loops[0][0]), N, N, R(1),
                                       reversed_face, False, None)
        return a.AcisModel(a.AcisMetadata(units_mm=1., resabs=1e-6), tuple(self.entries))


def cylinder_surface(radius):
    return cone(major_axis=a.Vec3(radius, 0, 0), reference_radius=radius, parameter_scale=radius,
                sin_half_angle=0., cos_half_angle=1.)


def cone_surface():
    """Radius 4 at z = 0, opening 0.75 per unit of z (sin 0.6, cos 0.8)."""
    return cone(major_axis=a.Vec3(4, 0, 0), reference_radius=4, parameter_scale=4)


def full_rim(m, z, radius, reverse):
    vertex = m.vertex(radius, 0, z)  # at u = 0: the chart's own seam passes through it
    m.loop([(m.circle(z, radius), vertex, 0., vertex, 2 * math.pi, reverse)])


def window(m, radius_at, u1, u2, z1, z2, hole_sense=True):
    """A quadrilateral hole bounded by two isocircle arcs and two generator lines, clockwise in (u, v)."""
    def p(u, z):
        r = radius_at(z)
        return m.vertex(r * math.cos(u), r * math.sin(u), z)
    v11, v12, v22, v21 = p(u1, z1), p(u1, z2), p(u2, z2), p(u2, z1)
    up, h_up = m.line(v11, v12)
    down, h_down = m.line(v22, v21)
    uses = [(up, v11, 0., v12, h_up, False),
            (m.circle(z2, radius_at(z2)), v12, u1, v22, u2, False),
            (down, v22, 0., v21, h_down, False),
            (m.circle(z1, radius_at(z1)), v11, u1, v21, u2, True)]
    if not hole_sense:
        uses = [use[:5] + (not use[5],) for use in reversed(uses)]
    m.loop(uses)


def reversed_loops(m):
    """The same faces with every loop traversed the other way (a consistent reversed face)."""
    for k, (loop_index, first) in enumerate(m.loops):
        coedges = []
        c = first
        while True:
            coedges.append(c)
            c = m.entries[c].next_coedge.index
            if c == first:
                break
        order = coedges[::-1]
        for j, c in enumerate(order):
            m.entries[c] = replace(m.entries[c], next_coedge=R(order[(j + 1) % len(order)]),
                                   previous_coedge=R(order[j - 1]), reversed=not m.entries[c].reversed)
        m.loops[k] = (loop_index, order[0])


class SeamChecks(unittest.TestCase):
    def convert(self, model):
        c = a.CadQueryConverter(model)
        return c, c._face(model.entities[0], _Placement(None, 1.))

    def assert_qualified_seam(self, c, model, face, area, holes, radial_sign=1., normal_at=None, edge_limit=None):
        from OCP.BRep import BRep_Tool
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.gp import gp_Pnt
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        self.assertTrue(face.isValid())
        properties = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face.wrapped, properties, 1e-10)  # adaptive: spline UV boundaries
        self.assertAlmostEqual(properties.Mass(), area, delta=1e-5)
        self.assertEqual(len(face.Wires()), 1 + holes)
        record = c.periodic_seam_faces[-1]
        self.assertEqual(record['face'], 0)
        self.assertEqual(record['holes'], holes)
        self.assertEqual(record['seam_edges'], 1)
        self.assertGreaterEqual(record['seam_clearance_mm'], 100 * c.tolerance)
        seams = [e for e in face.Edges() if BRep_Tool.IsClosed_s(e.wrapped, face.wrapped)]
        self.assertEqual(len(seams), 1)
        points = [e.location for e in model.entities if isinstance(e, a.PointEntity)]
        for p in points:
            self.assertTrue(any(math.dist(v.toTuple(), (p.x, p.y, p.z)) < c.tolerance for v in face.Vertices()))
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(p.x, p.y, p.z)).Vertex()
            distance = BRepExtrema_DistShapeShape(vertex, seams[0].wrapped)
            self.assertTrue(distance.IsDone())
            self.assertGreaterEqual(distance.Value(), record['seam_clearance_mm'] * (1 - 1e-9))
        # A source edge keeps its own tolerance (a stored fit's saved fit tolerance widens it); the seam is exact.
        limit = c.tolerance if edge_limit is None else edge_limit
        self.assertTrue(all(BRep_Tool.Tolerance_s(e.wrapped) <= math.nextafter(c.tolerance, math.inf)
                            for e in seams))
        self.assertTrue(all(BRep_Tool.Tolerance_s(e.wrapped) <= math.nextafter(limit, math.inf)
                            for e in face.Edges()))
        if normal_at is not None:
            location, radial = normal_at
            from cadquery import Vector
            normal = face.normalAt(Vector(*location))
            self.assertGreater(radial_sign * (normal.x * radial[0] + normal.y * radial[1] + normal.z * radial[2]), .5)

    @staticmethod
    def periodic_inputs(model):
        """What ``_cone_face`` hands to the seam code for a face with more than two loops."""
        c = a.CadQueryConverter(model)
        face, placement = model.entities[0], _Placement(None, 1.)
        surface = c._resolve_geometry(face.surface)
        loops = c._linked_entities(face.loop, a.LoopEntity, 'next_loop', context='test')
        geometry = c._cone_geometry(face, surface, placement)
        return c, face, placement, loops, geometry, c._cone_wires(face, surface, loops, placement, geometry)

    def assert_refused(self, model, reason):
        c = a.CadQueryConverter(model)
        with self.assertRaises(a.CadQueryConversionError) as caught:
            c._face(model.entities[0], _Placement(None, 1.))
        self.assertEqual(caught.exception.code, 'geometry.face_invalid')
        self.assertEqual(c.periodic_seam_faces, [])
        self.assertTrue(any(reason in r['reason'] for r in c.periodic_seam_refusals), c.periodic_seam_refusals)


class PeriodicSeamTests(SeamChecks):
    def test_two_loop_cylinder_bounded_by_exact_intersection_curves(self):
        radius = 2.
        m = FaceModel(cylinder_surface(radius))
        lower = m.nurbs_section(radius, 0., .5, 0.)
        l0, l1 = m.vertex(radius, 0, .5), m.vertex(-radius, 0, -.5)
        m.loop([(lower, l0, 0., l1, .5, False), (lower, l1, .5, l0, 1., False)])
        upper = m.nurbs_section(radius, 4., 0., .3)
        u0, u1 = m.vertex(radius, 0, 4.), m.vertex(-radius, 0, 4.)
        m.loop([(upper, u1, .5, u0, 1., True), (upper, u0, 0., u1, .5, True)])
        model = m.model()
        c, face = self.convert(model)
        # Plane sections z = c + a cos(u) + b sin(u): the band area is 2 pi r (c_upper - c_lower).
        self.assert_qualified_seam(c, model, face, 2 * math.pi * radius * 4., 0,
                                   normal_at=((0., radius, 2.), (0., 1., 0.)))
        self.assertEqual(c.periodic_seam_faces[-1]['source_edges'], 4)

    def test_two_loop_cylinder_bounded_by_ellipse_arcs(self):
        radius, slope = 2., .5
        m = FaceModel(cylinder_surface(radius))
        k = math.hypot(radius, slope)
        section = m.ellipse(a.Vec3(0, 0, 0), a.Vec3(-slope / k, 0, radius / k), a.Vec3(radius, 0, slope), radius / k)
        l0, l1 = m.vertex(radius, 0, slope), m.vertex(-radius, 0, -slope)
        m.loop([(section, l0, 0., l1, math.pi, False), (section, l1, math.pi, l0, 2 * math.pi, False)])
        full_rim(m, 4., radius, True)
        model = m.model()
        c, face = self.convert(model)
        self.assert_qualified_seam(c, model, face, 2 * math.pi * radius * 4., 0,
                                   normal_at=((0., -radius, 2.), (0., -1., 0.)))

    def test_cone_with_two_rims_and_two_holes(self):
        def radius_at(z):
            return 4. + .75 * z
        m = FaceModel(cone_surface())
        full_rim(m, 0., 4., False)
        full_rim(m, 4., 7., True)
        window(m, radius_at, -.4, .6, 1., 2.)  # crosses u = 0, where the rims have their vertices
        window(m, radius_at, 2., 3., 2.5, 3.2)
        model = m.model()
        c, face = self.convert(model)
        # Frustum 0 <= z <= 4 (slant 5): pi (r0 + r4) s; each window: du (r_a + r_b) / 2 * dz / cos.
        holes = 1. * (4.75 + 5.5) / 2 * 1.25 + 1. * (5.875 + 6.4) / 2 * .875
        self.assert_qualified_seam(c, model, face, math.pi * 11. * 5. - holes, 2,
                                   normal_at=((-radius_at(3.6), 0., 3.6), (-.8, 0., -.6)))

    def test_cylinder_with_two_rims_and_one_hole_both_senses(self):
        radius = 2.
        for reversed_face in (False, True):
            m = FaceModel(cylinder_surface(radius))
            full_rim(m, 0., radius, False)
            full_rim(m, 6., radius, True)
            window(m, lambda z: radius, -.5, .35, 2., 4.)  # crosses u = 0, like treppenturm 208BCE8 $56
            if reversed_face:
                reversed_loops(m)
            model = m.model(reversed_face)
            c, face = self.convert(model)
            self.assert_qualified_seam(c, model, face, 2 * math.pi * radius * 6. - radius * .85 * 2., 1,
                                       radial_sign=-1. if reversed_face else 1.,
                                       normal_at=((-radius, 0., 1.), (-1., 0., 0.)))

    def test_no_seam_parameter_clear_of_every_hole_fails_closed(self):
        radius = 2.
        m = FaceModel(cylinder_surface(radius))
        full_rim(m, 0., radius, False)
        full_rim(m, 10., radius, True)
        for u1, u2, z in ((.2, 2.4, 1.), (2.2, 4.4, 3.), (4.2, 6.5, 5.)):  # together they cover every u
            window(m, lambda z: radius, u1, u2, z, z + 1.)
        self.assert_refused(m.model(), 'seam parameter')

    def test_winding_mismatch_fails_closed(self):
        radius = 2.
        m = FaceModel(cylinder_surface(radius))
        full_rim(m, 0., radius, False)
        full_rim(m, 6., radius, False)  # same sense as the lower rim
        window(m, lambda z: radius, 1., 2., 2., 4.)
        self.assert_refused(m.model(), 'winding')

    def test_hole_against_the_band_sense_fails_closed(self):
        radius = 2.
        m = FaceModel(cylinder_surface(radius))
        full_rim(m, 0., radius, False)
        full_rim(m, 6., radius, True)
        window(m, lambda z: radius, 1., 2., 2., 4., hole_sense=False)
        self.assert_refused(m.model(), 'hole')


class ConeSenseTests(SeamChecks):
    """The chart sense of the seamed cone face: cos < 0 cones and reversed faces (the f9aa9dd rule)."""

    def test_cone_senses_cos_positive_reversed_and_cos_negative_forward_and_reversed(self):
        # ACIS cone sense: away from the axis for cos > 0, toward it for cos < 0; a reversed face turns it.
        for cos, reversed_face in ((.8, True), (-.8, False), (-.8, True)):
            with self.subTest(cos=cos, reversed_face=reversed_face):
                sign = (1. if cos > 0 else -1.) * (-1. if reversed_face else 1.)
                slope = .75 * (1. if cos > 0 else -1.)  # radius = 4 + slope * z

                def radius_at(z):
                    return 4. + slope * z

                def strip(du, z1, z2):
                    return du * (radius_at(z1) + radius_at(z2)) / 2 * (z2 - z1) / .8

                m = FaceModel(cone_surface() if cos > 0 else replace(cone_surface(), cos_half_angle=cos))
                full_rim(m, 0., 4., False)
                full_rim(m, 4., radius_at(4.), True)
                window(m, radius_at, -.4, .6, 1., 2.)  # crosses u = 0, where the rims have their vertices
                window(m, radius_at, 2., 3., 2.5, 3.2)
                if reversed_face:
                    reversed_loops(m)
                model = m.model(reversed_face)
                c, face = self.convert(model)
                area = math.pi * (radius_at(0.) + radius_at(4.)) * 5. - strip(1., 1., 2.) - strip(1., 2.5, 3.2)
                outward = (-.8, 0., -.6 if cos > 0 else .6)  # the cone's outward normal at u = pi
                self.assert_qualified_seam(c, model, face, area, 2, radial_sign=sign,
                                           normal_at=((-radius_at(3.6), 0., 3.6), outward))


def stored_fit_model(reversed_face):
    """A 2-loop cylinder band whose rims are stored fits (two edges each on one fit curve)."""
    radius = 2.
    m = FaceModel(cylinder_surface(radius))
    for z, reverse in ((0., False), (4., True)):
        fit = m.stored_fit_circle(z, radius)
        v0, v1 = m.vertex(radius, 0, z), m.vertex(-radius, 0, z)
        first, second = (fit, v0, 0., v1, math.pi, reverse), (fit, v1, math.pi, v0, 2 * math.pi, reverse)
        m.loop([second, first] if reverse else [first, second])
    if reversed_face:
        reversed_loops(m)
    return m.model(reversed_face)


class StoredFitSeamTests(SeamChecks):
    """A face bounded by stored fits: the deferred seam runs after the heal, in the face sense (F7 face $54)."""

    def convert_with_heal_spy(self, model):
        """Convert with the heal left to run unchanged but its outcome withheld: it leaves the face invalid."""
        c = a.CadQueryConverter(model)
        calls = []

        def heal(face, result):
            calls.append(dict(deferred=face.index in c._deferred_periodic_seams, valid=result.isValid(),
                              seams=len(c.periodic_seam_faces)))
            return result

        with mock.patch.object(c, '_heal_approximated_face', side_effect=heal):
            face = c._face(model.entities[0], _Placement(None, 1.))
        return c, face, calls

    def test_stored_fit_band_is_deferred_healed_first_and_seamed_in_both_senses(self):
        from OCP.BRep import BRep_Tool
        for reversed_face in (False, True):
            with self.subTest(reversed_face=reversed_face):
                model = stored_fit_model(reversed_face)
                c, face, calls = self.convert_with_heal_spy(model)
                self.assertEqual({x["subtype"] for x in c.approximations}, {'int_int_cur'})
                self.assertEqual(c.periodic_seam_refusals, [])
                # The heal ran first, once, on the still invalid face, with the seam waiting; no seam existed yet.
                self.assertEqual(calls, [dict(deferred=True, valid=False, seams=0)])
                self.assertEqual(c._deferred_periodic_seams, {})
                self.assertTrue(c.periodic_seam_faces[-1]['after_heal'])
                # The band's own area (2 pi r h, exact: the fits keep z constant) and the face sense.
                self.assert_qualified_seam(c, model, face, 2 * math.pi * 2. * 4., 0,
                                           radial_sign=-1. if reversed_face else 1.,
                                           normal_at=((0., 2., 2.), (0., 1., 0.)),
                                           edge_limit=c.tolerance + _STORED_FIT_MARGIN * 1e-3 + 1e-9)  # margin x the saved fit tolerance
                # Left to the heal (not withheld), this synthetic band is repaired by it and takes no new seam.
                healed = a.CadQueryConverter(model)
                self.assertTrue(healed._face(model.entities[0], _Placement(None, 1.)).isValid())
                self.assertEqual((len(healed.approximated_face_repairs), healed.periodic_seam_faces), (1, []))
                # Every source curve, trim interval and orientation kept: 4 source edges on stored fits.
                self.assertEqual(c.periodic_seam_faces[-1]['source_edges'], 4)
                pieces = [e for e in face.Edges() if not BRep_Tool.IsClosed_s(e.wrapped, face.wrapped)]
                curves = []
                for e in pieces:
                    curve = BRep_Tool.Curve_s(e.wrapped, 0., 0.)
                    if not any(curve == known for known in curves):
                        curves.append(curve)
                self.assertEqual(len(curves), 4)  # one stored-fit curve handle per source edge, none replaced
                length = 0.
                for e in pieces:
                    first, last = BRep_Tool.Range_s(e.wrapped)
                    length += last - first
                self.assertAlmostEqual(length, 2 * 2 * math.pi, places=9)
                senses = {z: {e.wrapped.Orientation() for e in pieces if round(e.Center().z, 6) == z} for z in (0., 4.)}
                self.assertTrue(all(len(s) == 1 for s in senses.values()), senses)
                self.assertNotEqual(senses[0.], senses[4.])

    def test_stored_fit_seam_refusals_leave_the_heal_result_and_fail_closed(self):
        model = stored_fit_model(False)
        c = a.CadQueryConverter(model)
        with mock.patch.object(c, '_heal_approximated_face', side_effect=lambda face, result: result):
            with mock.patch('cq_acis._analytic_seams._check_periodic_seam', side_effect=ValueError('forced')):
                with self.assertRaises(a.CadQueryConversionError) as caught:
                    c._face(model.entities[0], _Placement(None, 1.))
        self.assertEqual(caught.exception.code, 'geometry.face_invalid')
        self.assertEqual(c.periodic_seam_faces, [])
        self.assertTrue(any(r['after_heal'] and 'forced' in r['reason'] for r in c.periodic_seam_refusals))


class FailClosedErrorTests(SeamChecks):
    """OCC and conversion errors inside the seam code refuse the seam path: geometry.face_invalid (M1, M2)."""

    def band_with_hole(self):
        radius = 2.
        m = FaceModel(cylinder_surface(radius))
        full_rim(m, 0., radius, False)
        full_rim(m, 6., radius, True)
        window(m, lambda z: radius, 1., 2., 2., 4.)
        return m.model()

    def injected(self):
        from OCP.Standard import Standard_Failure
        from cq_acis.cadquery import CadQueryConversionError
        return [('_uv_curve', Standard_Failure('injected interpolation failure')),
                ('_settle_seam', Standard_Failure('injected compose failure')),
                ('_check_periodic_seam', CadQueryConversionError('injected deviation failure',
                                                                 code='geometry.supported_curve_mismatch'))]

    def test_occ_and_conversion_errors_in_the_seam_code_are_a_face_invalid_refusal(self):
        for name, error in self.injected():
            with self.subTest(name=name):
                model = self.band_with_hole()
                c = a.CadQueryConverter(model)
                with mock.patch(f'cq_acis._analytic_seams.{name}', side_effect=error):
                    with self.assertRaises(a.CadQueryConversionError) as caught:
                        c._face(model.entities[0], _Placement(None, 1.))
                self.assertEqual(caught.exception.code, 'geometry.face_invalid', str(caught.exception))
                self.assertEqual(c.periodic_seam_faces, [])
                self.assertTrue(any('injected' in r['reason'] for r in c.periodic_seam_refusals), c.periodic_seam_refusals)

    def test_stored_fit_deferred_seam_errors_are_a_face_invalid_refusal(self):
        for name, error in self.injected():
            with self.subTest(name=name):
                model = stored_fit_model(False)
                c = a.CadQueryConverter(model)
                with mock.patch.object(c, '_heal_approximated_face', side_effect=lambda face, result: result), \
                        mock.patch(f'cq_acis._analytic_seams.{name}', side_effect=error):
                    with self.assertRaises(a.CadQueryConversionError) as caught:
                        c._face(model.entities[0], _Placement(None, 1.))
                self.assertEqual(caught.exception.code, 'geometry.face_invalid', str(caught.exception))
                self.assertTrue(any(r['after_heal'] and 'injected' in r['reason'] for r in c.periodic_seam_refusals),
                                c.periodic_seam_refusals)

    @staticmethod
    def brep_text(wire):
        from OCP.BRepTools import BRepTools
        handle, path = tempfile.mkstemp(suffix='.brep')
        os.close(handle)
        try:
            assert BRepTools.Write_s(wire.wrapped, path)
            with open(path, 'rb') as stream:
                return stream.read()
        finally:
            os.remove(path)

    def test_the_callers_wires_are_not_mutated_by_a_refusal_or_by_the_seam(self):
        from cq_acis import _analytic_seams as seams
        for forced in (True, False):
            with self.subTest(forced_late_refusal=forced):
                c, face, placement, loops, geometry, wires = self.periodic_inputs(self.band_with_hole())
                before = [self.brep_text(w) for w in wires]
                if forced:
                    with mock.patch.object(seams, '_check_periodic_seam', side_effect=ValueError('forced late')):
                        self.assertIsNone(seams.periodic_seam_face(c, face, loops, wires, geometry, placement))
                    self.assertTrue(any('forced late' in r['reason'] for r in c.periodic_seam_refusals))
                else:
                    self.assertIsNotNone(seams.periodic_seam_face(c, face, loops, wires, geometry, placement))
                self.assertEqual(before, [self.brep_text(w) for w in wires])


class FailClosedBranchTests(SeamChecks):
    """Refusals of the seam code that no other test reaches (M4)."""

    def band_with_hole(self, u1=1., u2=2.):
        radius = 2.
        m = FaceModel(cylinder_surface(radius))
        full_rim(m, 0., radius, False)
        full_rim(m, 6., radius, True)
        window(m, lambda z: radius, u1, u2, 2., 4.)
        return m.model()

    def test_mirrored_placement_fails_closed(self):
        from cq_acis import _analytic_seams as seams
        c, face, placement, loops, geometry, wires = self.periodic_inputs(self.band_with_hole())
        with mock.patch.object(_Placement, 'orientation_sign', new=-1.):
            self.assertIsNone(seams.periodic_seam_face(c, face, loops, wires, geometry, placement))
        self.assertEqual(c.periodic_seam_faces, [])
        self.assertTrue(any('mirrored placement' in r['reason'] for r in c.periodic_seam_refusals),
                        c.periodic_seam_refusals)

    def test_a_hole_crossing_the_seam_parameter_fails_closed(self):
        model = self.band_with_hole()
        with mock.patch('cq_acis._analytic_seams._free_parameter', return_value=1.5):  # inside the hole's u-span
            self.assert_refused(model, 'hole crosses the seam parameter')

    def test_uv_curve_over_the_edge_tolerance_is_refused(self):
        from OCP.Geom import Geom_Circle, Geom_CylindricalSurface
        from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt
        from cq_acis import _analytic_seams as seams
        surface = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), 2.)
        chart = seams._Chart(surface)
        on = Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 1), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), 2.)
        off = Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 1), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), 2.01)
        pcurve, _, distance, kind = seams._uv_curve(on, 0., 2 * math.pi, chart, surface, 1e-6)
        self.assertLessEqual(distance, 1e-6)
        with self.assertRaises(seams._Refusal) as caught:
            seams._uv_curve(off, 0., 2 * math.pi, chart, surface, 1e-6)  # 0.01 off the support
        self.assertIn('exceeds the edge tolerance', str(caught.exception))

    def test_a_point_at_or_beyond_the_cone_apex_is_refused(self):
        from OCP.Geom import Geom_ConicalSurface
        from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt
        from cq_acis import _analytic_seams as seams
        semi = math.atan2(.6, .8)
        surface = Geom_ConicalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), semi, 4.)
        chart = seams._Chart(surface)
        apex_z = -4. / .6 * .8  # radius 4 + v * .6 = 0 at v = -4 / .6, z = v * cos
        u, v = chart.inverse(gp_Pnt(4., 0., 0.))  # a regular point still maps
        self.assertAlmostEqual(v, 0.)
        for z in (apex_z, apex_z - 1.):
            with self.subTest(z=z):
                with self.assertRaises(seams._Refusal) as caught:
                    chart.inverse(gp_Pnt(0., 0., z))
                self.assertIn('apex', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
