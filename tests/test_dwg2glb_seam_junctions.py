"""dwg2glb (M4b F7, ruling R205): a shell left open by the T-junction of a FixMissingSeam seam vertex.

The M4 heal (`_heal_approximated_face`, `ShapeFix_Face.FixMissingSeam`) adds the chart seam of a periodic face bounded
by stored fits. Where the seam meets a boundary loop inside a source edge, the healed face carries that edge as two
pieces while the neighbouring face keeps it whole; sewing does not cut the T-junction and the shell stays open
(`geometry.shell_open`; medium-block 4B70 face $8 / edge $65 / face $33). These tests pin the fallback that splits the
partner edge at the seam vertex, only after the existing path refused the shell as open, and everything it refuses.

Fixture (synthetic, OCC only): a cylinder band r = 2, 0 <= z <= 4, built without a seam and healed by the converter's
own heal; the seam runs along u = 0, through a vertex of the bottom rim (vertices at u = 0, pi; 4B70 loop $11) and
inside an edge of the top rim (vertices at u = 0.5, 0.5 + pi; loop $18). The top and bottom disks keep their rim
edges whole. The converter's `_face` is replaced by these faces, so `_shell` (sewing, closure, the fallback) runs
unchanged.

Why sewing leaves the real junction open (native dev loop, 4B70): the heal puts the seam vertex on the surface through
the boundary pcurve, 1.008e-6 mm off the partner edge's 3D curve: above the 1e-6 sewing tolerance (no cut), inside
its own 9.44e-5 vertex tolerance. An exact fixture is cut by sewing (`test_exact_fixture_is_closed_by_sewing`); so
the band's top-rim pcurves sit DV above their 3D circles (edge tolerance RIM_TOL), as the stored-fit pcurves do.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from unittest import mock

from cq_acis import (
    NULL_REF, AcisMetadata, AcisModel, BodyEntity, CadQueryConversionError, CadQueryConverter, EntityRef, FaceEntity,
    LumpEntity, RawEntity, ShellEntity,
)
from cq_acis.cadquery import _Placement

OCP_AVAILABLE = importlib.util.find_spec("OCP") is not None

RADIUS, HEIGHT = 2.0, 4.0
TOP_ANGLES = (0.5, 0.5 + math.pi)  # the seam (u = 0) falls inside the second top edge
BOTTOM_ANGLES = (0.0, math.pi)     # the seam passes through a bottom vertex
EDGE_TOL = 1e-7
DV, RIM_TOL = 5e-6, 1e-5  # band top-rim pcurve offset and edge tolerance (4B70 $8: 1.008e-6 off, vertex tol 9.44e-5)
PARTNER_TOL = 0.0201  # 4B70 face $33 keeps edge $65 at a tolerant 0.0201 mm
PLACEMENT = _Placement(None, 1.0)
SHELL, SIDE, TOP, BOTTOM = 2, 3, 4, 5
SPLIT = "cq_acis._seam_junctions.split_seam_junctions"


def raw(index, name):
    return RawEntity(index, name, NULL_REF, None)


def shell_model(face_count=3, outer=True):
    """body 0, lump 1, shell 2 and faces 3.. (their geometry comes from the patched `_face`). Not `outer`: the lump's
    first shell is another (empty) shell after the faces, whose next shell is shell 2 (a cavity shell)."""
    first = 3
    other = first + face_count
    entities = [
        BodyEntity(raw(0, "body"), NULL_REF, EntityRef(1), NULL_REF, NULL_REF),
        LumpEntity(raw(1, "lump"), NULL_REF, NULL_REF, EntityRef(SHELL if outer else other), EntityRef(0)),
        ShellEntity(raw(2, "shell"), NULL_REF, NULL_REF, NULL_REF, EntityRef(first), NULL_REF, EntityRef(1)),
    ]
    for k in range(face_count):
        following = EntityRef(first + k + 1) if k + 1 < face_count else NULL_REF
        entities.append(FaceEntity(raw(first + k, "face"), NULL_REF, following, NULL_REF, EntityRef(SHELL), NULL_REF,
                                   NULL_REF, False, False, None))
    if not outer:
        entities.append(ShellEntity(raw(other, "shell"), NULL_REF, EntityRef(SHELL), NULL_REF, NULL_REF, NULL_REF,
                                    EntityRef(1)))
    return AcisModel(AcisMetadata(units_mm=1.0, resabs=1e-6), tuple(entities))


def rim(z, angles, cylinder=None, tolerance=None, dv=0.0):
    """The circle r = RADIUS at height z as edges between `angles` (ascending), in increasing angle.

    With `cylinder`, every edge also carries its pcurve on it (the line v = z + dv, u = the circle parameter)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.Geom import Geom_Circle
    from OCP.Geom2d import Geom2d_Line
    from OCP.gp import gp_Ax2, gp_Dir, gp_Dir2d, gp_Pnt, gp_Pnt2d
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Vertex
    builder = BRep_Builder()
    circle = Geom_Circle(gp_Ax2(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), RADIUS)
    vertices = []
    for t in angles:
        vertex = TopoDS_Vertex()
        builder.MakeVertex(vertex, circle.Value(t), EDGE_TOL if tolerance is None else tolerance)
        vertices.append(vertex)
    edges = []
    for k, t0 in enumerate(angles):
        t1 = angles[k + 1] if k + 1 < len(angles) else angles[0] + 2 * math.pi
        edge = BRepBuilderAPI_MakeEdge(circle, vertices[k], vertices[(k + 1) % len(angles)], t0, t1).Edge()
        if cylinder is not None:
            builder.UpdateEdge(edge, Geom2d_Line(gp_Pnt2d(0, z + dv), gp_Dir2d(1, 0)), cylinder, TopLoc_Location(),
                               EDGE_TOL if tolerance is None else tolerance)
        if tolerance is not None:
            builder.UpdateEdge(edge, tolerance)
        edges.append(edge)
    return edges


def band(top_angles=TOP_ANGLES, bottom_angles=BOTTOM_ANGLES, dv=DV):
    """The cylinder band without a seam (invalid until healed), outward normal; top-rim pcurves dv above the rim."""
    from OCP.BRep import BRep_Builder
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt
    from OCP.TopoDS import TopoDS_Face, TopoDS_Wire
    builder = BRep_Builder()
    cylinder = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), RADIUS)
    face = TopoDS_Face()
    builder.MakeFace(face, cylinder, EDGE_TOL)
    top = rim(HEIGHT, top_angles, cylinder, tolerance=RIM_TOL if dv else None, dv=dv)
    for edges, forward in ((rim(0.0, bottom_angles, cylinder), True), (top, False)):
        wire = TopoDS_Wire()
        builder.MakeWire(wire)
        for edge in (edges if forward else reversed(edges)):
            builder.Add(wire, edge if forward else edge.Reversed())
        builder.Add(face, wire)
    return face


def disk(z, angles, up, tolerance=None):
    """A planar disk bounded by its own rim edges; normal +z if `up`, else -z."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopoDS import TopoDS, TopoDS_Wire
    builder = BRep_Builder()
    wire = TopoDS_Wire()
    builder.MakeWire(wire)
    for edge in rim(z, angles, tolerance=tolerance):
        builder.Add(wire, edge)
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), wire).Face()
    return face if up else TopoDS.Face_s(face.Reversed())


def fixture_faces(partner_tolerance=None, top_angles=TOP_ANGLES, with_bottom=True, dv=DV, top_z=HEIGHT):
    faces = {SIDE: band(top_angles, dv=dv), TOP: disk(top_z, top_angles, True, partner_tolerance)}
    if with_bottom:
        faces[BOTTOM] = disk(0.0, BOTTOM_ANGLES, False)
    return faces


def healing_face(c, faces, vertex_tolerance=None):
    """The `_face` replacement: the band goes through the converter's own M4 heal, the disks are returned as built.

    With `vertex_tolerance`, the tolerance of the healed band's top seam vertex is then set to exactly that value."""
    def face(entity, placement):
        shape = c.cq.Face(faces[entity.index])
        if entity.index == SIDE:
            healed = c._heal_approximated_face(entity, shape)
            if vertex_tolerance is not None:
                set_vertex_tolerance(top_seam_vertex(healed.wrapped), vertex_tolerance)
            return healed
        return shape
    return face


def top_seam_vertex(face):
    from cq_acis import _seam_junctions as J
    from OCP.BRep import BRep_Tool
    return next(v for v in J._seam_vertices(face) if BRep_Tool.Pnt_s(v).Z() > HEIGHT / 2)


def set_vertex_tolerance(vertex, tolerance):
    """Enforce (also lower) the tolerance of a shared vertex."""
    from OCP.ShapeFix import ShapeFix_ShapeTolerance
    from OCP.TopAbs import TopAbs_VERTEX
    ShapeFix_ShapeTolerance().SetTolerance(vertex, tolerance, TopAbs_VERTEX)


def convert_shell(faces, model=None):
    model = model or shell_model(len(faces))
    c = CadQueryConverter(model)
    with mock.patch.object(c, "_face", side_effect=healing_face(c, faces)):
        return c, model, c._shell(model.entities[SHELL], PLACEMENT)


def built_faces(c, model, faces, vertex_tolerance=None):
    """What `_shell` hands to the fallback: the converter's faces in shell order (the band healed by `c`)."""
    face = healing_face(c, faces, vertex_tolerance)
    entities = [e for e in model.entities if isinstance(e, FaceEntity)]
    return entities, [face(e, PLACEMENT) for e in entities]


def solid_volume(shell):
    from OCP.BRep import BRep_Builder
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS_Solid
    builder = BRep_Builder()
    solid = TopoDS_Solid()
    builder.MakeSolid(solid)
    builder.Add(solid, shell.wrapped)
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, properties)
    return properties.Mass()


def max_tolerance(shapes):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    worst = 0.0
    for shape in shapes:
        for kind, tolerance, cast in ((TopAbs_EDGE, BRep_Tool.Tolerance_s, TopoDS.Edge_s),
                                      (TopAbs_VERTEX, BRep_Tool.Tolerance_s, TopoDS.Vertex_s)):
            explorer = TopExp_Explorer(shape, kind)
            while explorer.More():
                worst = max(worst, tolerance(cast(explorer.Current())))
                explorer.Next()
    return worst


@unittest.skipUnless(OCP_AVAILABLE, "needs OCP")
class SeamJunctionTests(unittest.TestCase):
    def assert_closed(self, component):
        from OCP.BRep import BRep_Tool
        from OCP.BRepCheck import BRepCheck_NoError, BRepCheck_Shell
        self.assertTrue(component.isValid())
        check = BRepCheck_Shell(component.wrapped)
        self.assertTrue(BRep_Tool.IsClosed_s(component.wrapped))
        self.assertEqual(check.Closed(), BRepCheck_NoError)
        self.assertEqual(check.Orientation(), BRepCheck_NoError)

    def test_fixture_is_open_on_the_existing_path(self):
        """Characterisation (passes before the change): the healed band and the whole top rim do not sew closed."""
        from OCP.BRep import BRep_Tool
        for tolerance in (None, PARTNER_TOL):
            with self.subTest(partner_tolerance=tolerance):
                faces = fixture_faces(tolerance)
                model = shell_model(len(faces))
                c = CadQueryConverter(model)
                entities, built = built_faces(c, model, faces)
                self.assertEqual([r["face"] for r in c.approximated_face_repairs], [SIDE])
                seams = [e for e in built[0].Edges() if BRep_Tool.IsClosed_s(e.wrapped, built[0].wrapped)]
                self.assertEqual(len(seams), 1)
                low, high = sorted((v.toTuple() for v in seams[0].Vertices()), key=lambda p: p[2])
                self.assertLess(math.dist(low, (RADIUS, 0.0, 0.0)), 1e-9)
                # The top seam vertex lies on the surface through the raised pcurve: off the top circle by more
                # than the 1e-6 sewing tolerance (4B70: 1.008e-6), inside its own tolerance.
                self.assertTrue(1e-6 < math.dist(high, (RADIUS, 0.0, HEIGHT)) <= 2 * DV)
                components = c.cq.Shell.makeShell(built).Shells()
                self.assertEqual(len(components), 1)
                with self.assertRaises(CadQueryConversionError) as caught:
                    c._check_shell_closure(components[0], SHELL, 0)
                self.assertEqual(caught.exception.code, "geometry.shell_open")

    def test_exact_fixture_is_closed_by_sewing(self):
        """Characterisation: with exact pcurves the seam vertex is on the top circle and sewing cuts the junction."""
        faces = fixture_faces(dv=0.0)
        model = shell_model(len(faces))
        c = CadQueryConverter(model)
        built = built_faces(c, model, faces)[1]
        self.assertEqual(len(c.approximated_face_repairs), 1)
        components = c.cq.Shell.makeShell(built).Shells()
        self.assertEqual(len(components), 1)
        c._check_shell_closure(components[0], SHELL, 0)

    def test_seam_vertex_inside_a_partner_edge_closes_the_shell(self):
        for tolerance in (None, PARTNER_TOL):
            with self.subTest(partner_tolerance=tolerance):
                faces = fixture_faces(tolerance)
                c, model, components = convert_shell(faces)
                self.assertEqual(len(components), 1)
                self.assert_closed(components[0])
                self.assertEqual(c.seam_junction_refusals, [])
                self.assertEqual(len(c.seam_junction_splits), 1)
                split = c.seam_junction_splits[0]
                self.assertEqual((split["shell"], split["face"], split["partner_face"]), (SHELL, SIDE, TOP))
                self.assertEqual(split["method"], "partner edge split at the FixMissingSeam seam vertex")
                self.assertTrue(1e-6 < math.dist(split["point"], (RADIUS, 0.0, HEIGHT)) <= 2 * DV)
                # R206: the 4B70-like case (gap above the 1e-6 sewing tolerance, inside the seam vertex's own
                # tolerance) is accepted, and the split lists the gap and the tolerances it was judged against.
                self.assertTrue(1e-6 < split["gap_mm"] <= split["vertex_tolerance_mm"] < 1e-4)
                self.assertEqual(split["partner_edge_tolerance_mm"], EDGE_TOL if tolerance is None else tolerance)
                self.assertAlmostEqual(split["parameter"] % (2 * math.pi), 0.0, delta=1e-9)
                # One closure record, from the accepted shell only.
                self.assertEqual(len(c.shell_closure_checks), 1)
                # Between the cylinder volume to the top circle and to the raised top-rim pcurve.
                volume = solid_volume(components[0])
                self.assertGreaterEqual(volume, math.pi * RADIUS ** 2 * HEIGHT * (1 - 1e-9))
                self.assertLessEqual(volume, math.pi * RADIUS ** 2 * (HEIGHT + DV) * (1 + 1e-9))
                # No tolerance raised: nothing in the shell exceeds the faces' own tolerances.
                inputs = [shape.wrapped for shape in built_faces(CadQueryConverter(model), model, faces)[1]]
                self.assertLessEqual(max_tolerance([components[0].wrapped]), max_tolerance(inputs))

    def test_a_shell_closed_on_the_existing_path_never_enters_the_fallback(self):
        faces = fixture_faces(top_angles=(0.0, math.pi), dv=0.0)  # the seam meets both rims at a vertex
        with mock.patch(SPLIT) as spy:
            c, _, components = convert_shell(faces)
        spy.assert_not_called()
        self.assertEqual(len(c.approximated_face_repairs), 1)  # a healed face in a closed shell, as M4 emits it
        self.assertEqual(len(components), 1)
        self.assert_closed(components[0])
        self.assertEqual((c.seam_junction_splits, c.seam_junction_refusals), ([], []))

    def test_other_shell_failures_never_enter_the_fallback(self):
        faces = fixture_faces()
        model = shell_model(len(faces))
        c = CadQueryConverter(model)
        forced = CadQueryConversionError("forced", code="geometry.shell_orientation")
        with mock.patch(SPLIT) as spy, mock.patch.object(c, "_face", side_effect=healing_face(c, faces)), \
                mock.patch.object(c, "_check_shell_closure", side_effect=forced):
            with self.assertRaises(CadQueryConversionError) as caught:
                c._shell(model.entities[SHELL], PLACEMENT)
        spy.assert_not_called()
        self.assertIs(caught.exception, forced)

    def refused_like_today(self, faces, reason, model=None, vertex_tolerance=None):
        """The fallback refuses: today's error (code and message), state and no split; the refusal is recorded."""
        model = model or shell_model(len(faces))
        today = CadQueryConverter(model)
        with mock.patch(SPLIT, return_value=None), \
                mock.patch.object(today, "_face", side_effect=healing_face(today, faces, vertex_tolerance)):
            with self.assertRaises(CadQueryConversionError) as expected:
                today._shell(model.entities[SHELL], PLACEMENT)
        c = CadQueryConverter(model)
        with mock.patch.object(c, "_face", side_effect=healing_face(c, faces, vertex_tolerance)):
            with self.assertRaises(CadQueryConversionError) as caught:
                c._shell(model.entities[SHELL], PLACEMENT)
        self.assertEqual(caught.exception.code, "geometry.shell_open")
        self.assertEqual((caught.exception.code, str(caught.exception)),
                         (expected.exception.code, str(expected.exception)))
        self.assertEqual(c.shell_closure_checks, today.shell_closure_checks)
        self.assertEqual(c.approximated_face_repairs, today.approximated_face_repairs)
        self.assertEqual(c.seam_junction_splits, [])
        self.assertEqual(len(c.seam_junction_refusals), 1)
        self.assertEqual(c.seam_junction_refusals[0]["shell"], SHELL)
        self.assertIn(reason, c.seam_junction_refusals[0]["reason"])
        return c

    def test_still_open_after_the_split_returns_todays_refusal(self):
        self.refused_like_today(fixture_faces(with_bottom=False), "not closed")

    def test_partner_rim_off_the_band_returns_todays_refusal(self):
        """A partner whose rim is 1e-4 above the band's (inside its 0.0201 tolerance): split, still not sewn.

        R206: the seam vertex's tolerance is set to 2e-4 (the gap is 9.5e-5) so that the vertex-tolerance bound lets
        the split through, as the partner edge's 0.0201 did before; this test is about the later sewing refusing it.
        The bound's own refusal is `test_gap_beyond_the_vertex_tolerance_but_inside_the_edge_tolerance_is_refused`."""
        self.refused_like_today(fixture_faces(PARTNER_TOL, top_z=HEIGHT + 1e-4), "not closed", vertex_tolerance=2e-4)

    def test_open_shell_without_a_seam_heal_refused(self):
        faces = fixture_faces()
        healed = CadQueryConverter(shell_model())  # healed elsewhere: this converter recorded no heal
        side = healed._heal_approximated_face(shell_model().entities[SIDE], healed.cq.Face(faces[SIDE]))
        faces[SIDE] = side.wrapped
        model = shell_model(len(faces))
        c = CadQueryConverter(model)
        with mock.patch.object(c, "_face", side_effect=lambda entity, placement: c.cq.Face(faces[entity.index])):
            with self.assertRaises(CadQueryConversionError) as caught:
                c._shell(model.entities[SHELL], PLACEMENT)
        self.assertEqual(caught.exception.code, "geometry.shell_open")
        self.assertEqual(c.seam_junction_splits, [])
        self.assertIn("FixMissingSeam", c.seam_junction_refusals[0]["reason"])

    def test_cavity_shell_refused(self):
        self.refused_like_today(fixture_faces(), "outer shell", model=shell_model(3, outer=False))

    def test_occ_error_fails_closed(self):
        from OCP.Standard import Standard_Failure
        with mock.patch("cq_acis._seam_junctions._split_face", side_effect=Standard_Failure("injected")):
            self.refused_like_today(fixture_faces(), "Standard_Failure")

    def direct(self, faces, change=None, vertex_tolerance=None):
        """Call the fallback directly on the converter's faces (optionally changed), as `_shell` would."""
        from cq_acis._seam_junctions import split_seam_junctions
        model = shell_model(len(faces))
        c = CadQueryConverter(model)
        entities, built = built_faces(c, model, faces, vertex_tolerance)
        if change is not None:
            entities, built = change(c, entities, built)
        before = list(c.shell_closure_checks)
        result = split_seam_junctions(c, model.entities[SHELL], entities, built, len(c.shell_closure_checks))
        return c, result, before

    def test_ambiguous_partner_edges_refused(self):
        def duplicate_top(c, entities, built):
            extra = FaceEntity(raw(6, "face"), NULL_REF, NULL_REF, NULL_REF, EntityRef(SHELL), NULL_REF, NULL_REF,
                               False, False, None)
            return entities + [extra], built + [c.cq.Face(disk(HEIGHT, TOP_ANGLES, True))]
        c, result, before = self.direct(fixture_faces(), duplicate_top)
        self.assertIsNone(result)
        self.assertEqual(c.shell_closure_checks, before)
        self.assertEqual(c.seam_junction_splits, [])
        self.assertIn("ambiguous", c.seam_junction_refusals[0]["reason"])

    def test_negative_volume_refused(self):
        def reverse_all(c, entities, built):
            return entities, [c.cq.Face(face.wrapped.Reversed()) for face in built]
        c, result, before = self.direct(fixture_faces(), reverse_all)
        self.assertIsNone(result)
        self.assertEqual(c.shell_closure_checks, before)
        self.assertEqual(c.seam_junction_splits, [])
        self.assertIn("volume", c.seam_junction_refusals[0]["reason"])

    # --- R206: the split-point bound is the seam vertex's own tolerance

    def accepted_split_gap(self):
        """The unmodified fixture's split: the distance of the seam vertex to the partner rim circle."""
        c, result, _ = self.direct(fixture_faces(PARTNER_TOL))
        self.assertIsNotNone(result)
        (split,) = c.seam_junction_splits
        x, y, z = split["point"]
        return math.hypot(math.hypot(x, y) - RADIUS, z - HEIGHT)

    def test_gap_beyond_the_vertex_tolerance_but_inside_the_edge_tolerance_is_refused(self):
        """The partner edge's tolerance (0.0201, as 4B70 $33) must not widen the bound: the vertex's own decides."""
        gap = self.accepted_split_gap()
        tolerance = gap / 2  # beyond the vertex's tolerance, far inside the partner edge's 0.0201
        self.assertTrue(1e-6 < tolerance < gap < PARTNER_TOL)
        c, result, before = self.direct(fixture_faces(PARTNER_TOL), vertex_tolerance=tolerance)
        self.assertIsNone(result)
        self.assertEqual(c.shell_closure_checks, before)
        self.assertEqual(c.seam_junction_splits, [])
        (refusal,) = c.seam_junction_refusals
        self.assertEqual(refusal["shell"], SHELL)
        self.assertEqual(refusal["reason"], f"seam split refused: gap {refusal['gap_mm']:.6g} mm > "
                                            f"vertex tolerance {tolerance:.6g} mm")
        self.assertEqual((refusal["face"], refusal["partner_face"]), (SIDE, TOP))
        self.assertAlmostEqual(refusal["gap_mm"], gap, delta=1e-12)
        self.assertEqual(refusal["vertex_tolerance_mm"], tolerance)
        self.assertEqual(refusal["bound"], "seam vertex tolerance")

    def test_the_bound_is_the_vertex_tolerance_exactly(self):
        gap = self.accepted_split_gap()
        for factor, accepted in ((1.001, True), (0.999, False)):
            with self.subTest(factor=factor):
                c, result, _ = self.direct(fixture_faces(PARTNER_TOL), vertex_tolerance=gap * factor)
                self.assertEqual(result is not None, accepted)
                self.assertEqual(len(c.seam_junction_splits), int(accepted))
                self.assertEqual(len(c.seam_junction_refusals), int(not accepted))

    def test_bound_refusal_returns_todays_refusal_and_is_recorded(self):
        """Through `_shell`: today's shell_open error unchanged, and the bound refusal is the recorded fact."""
        gap = self.accepted_split_gap()
        c = self.refused_like_today(fixture_faces(PARTNER_TOL), "seam split refused: gap", vertex_tolerance=gap / 2)
        self.assertIn("> vertex tolerance", c.seam_junction_refusals[0]["reason"])

    # --- M2: fail-closed branches reached directly

    def split_inputs(self):
        """(converter, built faces, [(partner edge, parameter)]) of the fixture's one split."""
        from cq_acis import _seam_junctions as J
        faces = fixture_faces(PARTNER_TOL)
        model = shell_model(len(faces))
        c = CadQueryConverter(model)
        _, built = built_faces(c, model, faces)
        (hit,) = [h for v in J._seam_vertices(built[0].wrapped) for h in J._partner_hits(built, 0, v)]
        return c, built, hit

    def test_located_face_refused(self):
        from OCP.gp import gp_Trsf, gp_Vec
        from OCP.TopLoc import TopLoc_Location
        def move_top(c, entities, built):
            shift = gp_Trsf()
            built = list(built)
            shift.SetTranslation(gp_Vec(1, 0, 0))
            built[1] = c.cq.Face(built[1].wrapped.Moved(TopLoc_Location(shift)))
            return entities, built
        c, result, before = self.direct(fixture_faces(PARTNER_TOL), move_top)
        self.assertIsNone(result)
        self.assertEqual(c.shell_closure_checks, before)
        self.assertEqual(c.seam_junction_refusals[0]["reason"], "a face of the shell is located")

    def test_two_seam_vertices_in_one_edge_refused(self):
        from cq_acis import _seam_junctions as J
        def same_edge(built, k, vertex):
            return [(1, J._edges(built[1].wrapped)[0], 0.3, 0.0)]
        with mock.patch.object(J, "_partner_hits", side_effect=same_edge):
            c, result, before = self.direct(fixture_faces(PARTNER_TOL))
        self.assertIsNone(result)
        self.assertEqual(c.seam_junction_splits, [])
        self.assertIn("two seam vertices inside one edge", c.seam_junction_refusals[0]["reason"])

    def test_located_partner_edge_refused(self):
        from cq_acis import _seam_junctions as J
        from OCP.gp import gp_Trsf, gp_Vec
        from OCP.TopLoc import TopLoc_Location
        c, built, (j, edge, parameter, *_) = self.split_inputs()
        shift = gp_Trsf()
        shift.SetTranslation(gp_Vec(1, 0, 0))
        with self.assertRaisesRegex(J._Refused, "partner edge is located"):
            J._split_face(c, built[j], [(edge.Moved(TopLoc_Location(shift)), parameter)])

    def test_partner_edge_without_same_parameter_refused(self):
        from cq_acis import _seam_junctions as J
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS
        c, built, (j, edge, parameter, *_) = self.split_inputs()
        copy = TopoDS.Edge_s(edge.EmptyCopied())
        BRep_Builder().SameParameter(copy, False)
        with self.assertRaisesRegex(J._Refused, "not same-parameter / same-range"):
            J._split_face(c, built[j], [(copy, parameter)])

    def test_split_face_missing_its_edges_refused(self):
        from cq_acis import _seam_junctions as J
        c, built, (j, edge, parameter, *_) = self.split_inputs()
        with mock.patch.object(J, "_edges", return_value=[]):
            with self.assertRaisesRegex(J._Refused, "does not carry the split edges"):
                J._split_face(c, built[j], [(edge, parameter)])

    def test_invalid_split_face_refused(self):
        from cq_acis import _seam_junctions as J
        c, built, (j, edge, parameter, *_) = self.split_inputs()
        with mock.patch.object(type(built[j]), "isValid", return_value=False):
            with self.assertRaisesRegex(J._Refused, "split partner face is invalid"):
                J._split_face(c, built[j], [(edge, parameter)])

    def test_split_face_area_mismatch_refused(self):
        from cq_acis import _seam_junctions as J
        c, built, (j, edge, parameter, *_) = self.split_inputs()
        with mock.patch.object(J, "_area", side_effect=[1.0, 1.5]):
            with self.assertRaisesRegex(J._Refused, "area .* differs from"):
                J._split_face(c, built[j], [(edge, parameter)])


if __name__ == "__main__":
    unittest.main()
