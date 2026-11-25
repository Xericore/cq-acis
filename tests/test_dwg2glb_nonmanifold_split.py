"""dwg2glb (M4b F5, rulings R200 / R202 / R203): a shell with a non-manifold source edge split into body-consistent solids.

simple-stairs 22B8 (5 definitions, 320 occurrences): one ACIS shell holds a rod and a sleeve that meet only along one
circle, the source edge `$692` with a ring of 4 coedges (rod side, rod top, sleeve bore, sleeve bottom). Sewing all
faces together pairs the 4 coincident edges across the two bodies; orienting that shell inverts the rod and OCCT
certifies a wrong solid. The fork now splits such a shell, before sewing, into the bodies given by the manifold
(2-coedge) source edges plus the unique body-consistent pairing of every non-manifold ring, sews every body on its own,
and lets both solids use the same `TopoDS_Edge`s on the shared loop (so BRepMesh discretises it once). Anything that
does not qualify takes today's path.

Fixture (synthetic, OCC only), mirroring 22B8: a rod (cylinder r = 2, -4 <= z <= 0, with its top and bottom disks)
and a sleeve (bore r = 2 and outer cylinder R = 3, 0 <= z <= 2, with its bottom and top annuli). The shell's source
topology is a real loop / coedge / edge graph; the converter's `_face` is replaced by OCC faces built per source face
(each with its own edges, as the converter builds them), so `_shell` / `_lump` run unchanged. As in 22B8 the rod top
disk touches the rest of the rod only through the ring, so it is a body of its own over the manifold edges, and the
ring's senses are rod side reversed, rod top forward, bore reversed, sleeve bottom forward.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from types import SimpleNamespace
from unittest import mock

from cq_acis import (
    NULL_REF, AcisMetadata, AcisModel, BodyEntity, CadQueryConversionError, CadQueryConverter, CoedgeEntity, EdgeEntity,
    EntityRef, FaceEntity, LoopEntity, LumpEntity, RawEntity, ShellEntity,
)
from cq_acis.cadquery import _Placement

OCP_AVAILABLE = importlib.util.find_spec("OCP") is not None

R_ROD, R_OUT, H_ROD, H_SLEEVE = 2.0, 3.0, 4.0, 2.0
TOL = 1e-7
PLACEMENT = _Placement(None, 1.0)
LUMP, SHELL = 1, 2
ROD_SIDE, ROD_TOP, ROD_BOTTOM, BORE, SLEEVE_BOTTOM, SLEEVE_TOP, OUTER = range(3, 10)
RING, ROD_RIM, BORE_RIM, OUTER_LOW, OUTER_HIGH = "ring", "rod_rim", "bore_rim", "outer_low", "outer_high"
ROD = (ROD_SIDE, ROD_TOP, ROD_BOTTOM)
SLEEVE = (BORE, SLEEVE_BOTTOM, SLEEVE_TOP, OUTER)
V_ROD = math.pi * R_ROD ** 2 * H_ROD
V_SLEEVE = math.pi * (R_OUT ** 2 - R_ROD ** 2) * H_SLEEVE
SPLIT = "cq_acis._nonmanifold_split"

# Per source face: its loops, each a list of (source edge key, coedge reversed). The ring senses are 22B8's.
TWO_BODY_LOOPS = {
    ROD_SIDE: [[(RING, True)], [(ROD_RIM, False)]],
    ROD_TOP: [[(RING, False)]],
    ROD_BOTTOM: [[(ROD_RIM, True)]],
    BORE: [[(RING, True)], [(BORE_RIM, False)]],
    SLEEVE_BOTTOM: [[(OUTER_LOW, False)], [(RING, False)]],
    SLEEVE_TOP: [[(OUTER_HIGH, True)], [(BORE_RIM, True)]],
    OUTER: [[(OUTER_LOW, True)], [(OUTER_HIGH, False)]],
}
ROD_LOOPS = {face: TWO_BODY_LOOPS[face] for face in ROD}


def raw(index, name):
    return RawEntity(index, name, NULL_REF, None)


def topology_model(face_loops):
    """body 0, lump 1, shell 2, faces 3.. (sorted keys of `face_loops`), then loops, coedges and edges.

    Each coedge's partner is the next coedge of its edge (a cycle, as ACIS rings are); `_face` is patched, so the
    edges carry no geometry."""
    faces = sorted(face_loops)
    assert faces == list(range(3, 3 + len(faces)))
    index = 3 + len(faces)
    loop_ids, coedge_ids = {}, {}
    for face in faces:
        for k, _ in enumerate(face_loops[face]):
            loop_ids[face, k] = index
            index += 1
    for face in faces:
        for k, loop in enumerate(face_loops[face]):
            for j, _ in enumerate(loop):
                coedge_ids[face, k, j] = index
                index += 1
    edge_keys = sorted({edge for face in faces for loop in face_loops[face] for edge, _ in loop})
    edge_ids = {key: index + n for n, key in enumerate(edge_keys)}
    ring = {key: [] for key in edge_keys}
    for face in faces:
        for k, loop in enumerate(face_loops[face]):
            for j, (edge, _) in enumerate(loop):
                ring[edge].append(coedge_ids[face, k, j])
    entities = [None] * (index + len(edge_keys))
    entities[0] = BodyEntity(raw(0, "body"), NULL_REF, EntityRef(1), NULL_REF, NULL_REF)
    entities[LUMP] = LumpEntity(raw(LUMP, "lump"), NULL_REF, NULL_REF, EntityRef(SHELL), EntityRef(0))
    entities[SHELL] = ShellEntity(raw(SHELL, "shell"), NULL_REF, NULL_REF, NULL_REF, EntityRef(faces[0]), NULL_REF,
                                  EntityRef(LUMP))
    for n, face in enumerate(faces):
        following = EntityRef(faces[n + 1]) if n + 1 < len(faces) else NULL_REF
        entities[face] = FaceEntity(raw(face, "face"), NULL_REF, following, EntityRef(loop_ids[face, 0]),
                                    EntityRef(SHELL), NULL_REF, NULL_REF, False, False, None)
        loops = face_loops[face]
        for k, loop in enumerate(loops):
            following = EntityRef(loop_ids[face, k + 1]) if k + 1 < len(loops) else NULL_REF
            entities[loop_ids[face, k]] = LoopEntity(raw(loop_ids[face, k], "loop"), NULL_REF, following,
                                                     EntityRef(coedge_ids[face, k, 0]), EntityRef(face))
            for j, (edge, reversed_) in enumerate(loop):
                me = coedge_ids[face, k, j]
                nxt = coedge_ids[face, k, (j + 1) % len(loop)]
                prv = coedge_ids[face, k, (j - 1) % len(loop)]
                members = ring[edge]
                partner = members[(members.index(me) + 1) % len(members)]
                entities[me] = CoedgeEntity(raw(me, "coedge"), NULL_REF, EntityRef(nxt), EntityRef(prv),
                                            EntityRef(partner) if partner != me else NULL_REF,
                                            EntityRef(edge_ids[edge]), reversed_, EntityRef(loop_ids[face, k]),
                                            NULL_REF)
    for key in edge_keys:
        entities[edge_ids[key]] = EdgeEntity(raw(edge_ids[key], "edge"), NULL_REF, NULL_REF, None, NULL_REF, None,
                                             EntityRef(ring[key][0]), NULL_REF, False, None)
    return AcisModel(AcisMetadata(units_mm=1.0, resabs=1e-6), tuple(entities)), edge_ids


def _circle(z, radius, xdir=1.0):
    """The circle about the z axis; its parameter 0 lies on the x axis at sign `xdir` (22B8: the rod's loop circle
    starts at -x, the sleeve's at +x, so the same circle is parameterised with an offset of pi)."""
    from OCP.Geom import Geom_Circle
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    return Geom_Circle(gp_Ax2(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1), gp_Dir(xdir, 0, 0)), radius)


def cylinder_face(radius, z0, z1, outward, xdir=1.0):
    """A full cylinder band (its own seam and rim edges); outward = radial normal away from the axis."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt
    surface = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(xdir, 0, 0)), radius)
    face = BRepBuilderAPI_MakeFace(surface, 0.0, 2 * math.pi, z0, z1, TOL).Face()
    return face if outward else TopoDS.Face_s(face.Reversed())


def cylinder_face_cut_top(radius, z0, z1, outward, top_angles):
    """A full cylinder band built from its own edges, its top rim cut into pieces between `top_angles` (ascending,
    first 0): the rim of a face whose chart seam meets the loop (22B8: rod side $166 carries the ring as two
    halves, the other faces as one circle)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.Geom import Geom_CylindricalSurface, Geom_Line
    from OCP.Geom2d import Geom2d_Line
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Vertex, TopoDS_Wire
    from OCP.gp import gp_Ax3, gp_Dir, gp_Dir2d, gp_Pnt, gp_Pnt2d
    builder = BRep_Builder()
    surface = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), radius)
    location = TopLoc_Location()

    def vertex(point):
        made = TopoDS_Vertex()
        builder.MakeVertex(made, point, TOL)
        return made

    low = _circle(z0, radius)
    high = _circle(z1, radius)
    v_low = vertex(low.Value(0.0))
    tops = [vertex(high.Value(t)) for t in top_angles]
    bottom = BRepBuilderAPI_MakeEdge(low, v_low, v_low, 0.0, 2 * math.pi).Edge()
    builder.UpdateEdge(bottom, Geom2d_Line(gp_Pnt2d(0, z0), gp_Dir2d(1, 0)), surface, location, TOL)
    pieces = []
    for k, t0 in enumerate(top_angles):
        t1 = top_angles[k + 1] if k + 1 < len(top_angles) else 2 * math.pi
        piece = BRepBuilderAPI_MakeEdge(high, tops[k], tops[(k + 1) % len(tops)], t0, t1).Edge()
        builder.UpdateEdge(piece, Geom2d_Line(gp_Pnt2d(0, z1), gp_Dir2d(1, 0)), surface, location, TOL)
        pieces.append(piece)
    seam = BRepBuilderAPI_MakeEdge(Geom_Line(gp_Pnt(radius, 0, 0), gp_Dir(0, 0, 1)), v_low, tops[0], z0, z1).Edge()
    builder.UpdateEdge(seam, Geom2d_Line(gp_Pnt2d(2 * math.pi, 0), gp_Dir2d(0, 1)),
                       Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(0, 1)), surface, location, TOL)
    wire = TopoDS_Wire()
    builder.MakeWire(wire)
    builder.Add(wire, bottom)
    builder.Add(wire, seam)
    for piece in reversed(pieces):
        builder.Add(wire, piece.Reversed())
    builder.Add(wire, seam.Reversed())
    face = TopoDS_Face()
    builder.MakeFace(face, surface, TOL)
    builder.Add(face, wire)
    return face if outward else TopoDS.Face_s(face.Reversed())


def disk_face(z, radius, up, hole=None, xdir=1.0):
    """A planar disk (or an annulus with an inner radius `hole`) bounded by its own full-circle edges."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    outer = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(_circle(z, radius, xdir)).Edge()).Wire()
    maker = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), outer)
    if hole is not None:
        inner = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(_circle(z, hole, xdir)).Edge()).Wire()
        maker.Add(TopoDS.Wire_s(inner.Reversed()))
    face = maker.Face()
    return face if up else TopoDS.Face_s(face.Reversed())


def fixture_faces(invert_sleeve=False, rod_side_cut=False, sleeve_xdir=1.0):
    x = sleeve_xdir
    faces = {
        ROD_SIDE: (cylinder_face_cut_top(R_ROD, -H_ROD, 0.0, True, (0.0, math.pi)) if rod_side_cut
                   else cylinder_face(R_ROD, -H_ROD, 0.0, True)),
        ROD_TOP: disk_face(0.0, R_ROD, True),
        ROD_BOTTOM: disk_face(-H_ROD, R_ROD, False),
        BORE: cylinder_face(R_ROD, 0.0, H_SLEEVE, False, x),
        SLEEVE_BOTTOM: disk_face(0.0, R_OUT, False, hole=R_ROD, xdir=x),
        SLEEVE_TOP: disk_face(H_SLEEVE, R_OUT, True, hole=R_ROD, xdir=x),
        OUTER: cylinder_face(R_OUT, 0.0, H_SLEEVE, True, x),
    }
    if invert_sleeve:
        from OCP.TopoDS import TopoDS
        for face in SLEEVE:
            faces[face] = TopoDS.Face_s(faces[face].Reversed())
    return faces


def face_side_effect(c, faces):
    def face(entity, placement):
        return c.cq.Face(faces[entity.index])
    return face


def convert_lump(face_loops, faces):
    """(converter, model, result or the raised CadQueryConversionError) of `_lump` on the fixture."""
    model, _ = topology_model(face_loops)
    c = CadQueryConverter(model)
    with mock.patch.object(c, "_face", side_effect=face_side_effect(c, faces)):
        try:
            return c, model, c._lump(model.entities[LUMP], PLACEMENT)
        except CadQueryConversionError as error:
            return c, model, error


def todays_path(face_loops, faces):
    """Today's shell path written out (sew all faces once, then the existing shell checks), independent of the split:
    (('error', code, message) or ('shells', [signed volume per component]), closure records written)."""
    model, _ = topology_model(face_loops)
    c = CadQueryConverter(model)
    entities = [e for e in model.entities if isinstance(e, FaceEntity)]
    built = [c.cq.Face(faces[e.index]) for e in entities]
    try:
        result = c.cq.Shell.makeShell(built)
        components = c._shell_components(result, model.entities[SHELL], entities)
    except CadQueryConversionError as error:
        return ("error", error.code, str(error)), len(c.shell_closure_checks)
    return ("shells", [round(shell_volume(s), 6) for s in components]), len(c.shell_closure_checks)


def shell_outcome(face_loops, faces, model=None):
    """The converter's `_shell` on the fixture in the same terms as `todays_path`, plus the converter."""
    model = model or topology_model(face_loops)[0]
    c = CadQueryConverter(model)
    with mock.patch.object(c, "_face", side_effect=face_side_effect(c, faces)):
        try:
            components = c._shell(model.entities[SHELL], PLACEMENT)
        except CadQueryConversionError as error:
            return c, ("error", error.code, str(error))
    return c, ("shells", [round(shell_volume(s), 6) for s in components])


def shell_volume(shell):
    """Signed volume enclosed by the shell as oriented (no orientation fix)."""
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


def explore(shape, kind):
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    found = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, found)
    return [found.FindKey(i) for i in range(1, found.Extent() + 1)]


def oriented_normal(face):
    """(point, unit normal of the face as oriented) at a UV point inside the face (an annulus's hole is not)."""
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.BRepTools import BRepTools
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCP.TopAbs import TopAbs_IN
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
    face = TopoDS.Face_s(face)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    classifier = BRepTopAdaptor_FClass2d(face, 1e-9)
    inside = next((u0 + (u1 - u0) * a / 10, v0 + (v1 - v0) * b / 10) for a in range(1, 10) for b in range(1, 10)
                  if classifier.Perform(gp_Pnt2d(u0 + (u1 - u0) * a / 10, v0 + (v1 - v0) * b / 10)) == TopAbs_IN)
    point, normal = gp_Pnt(), gp_Vec()
    BRepGProp_Face(face).Normal(inside[0], inside[1], point, normal)
    normal.Normalize()
    return (point.X(), point.Y(), point.Z()), (normal.X(), normal.Y(), normal.Z())


def outward(point, face_index):
    """The analytic outward normal of the fixture's source face `face_index` at `point`."""
    x, y, _ = point
    rho = math.hypot(x, y)
    radial = (x / rho, y / rho, 0.0) if rho > 0 else None
    return {
        ROD_SIDE: radial, ROD_TOP: (0.0, 0.0, 1.0), ROD_BOTTOM: (0.0, 0.0, -1.0),
        BORE: radial and tuple(-k for k in radial), SLEEVE_BOTTOM: (0.0, 0.0, -1.0), SLEEVE_TOP: (0.0, 0.0, 1.0),
        OUTER: radial,
    }[face_index]


def source_face_of(point):
    """Which fixture face a point on the boundary of the two solids lies on."""
    x, y, z = point
    rho = math.hypot(x, y)
    if z < -1e-9:
        if abs(z + H_ROD) < 1e-7:
            return ROD_BOTTOM
        return ROD_SIDE
    if abs(z) < 1e-7:
        return ROD_TOP if rho < R_ROD - 1e-7 else SLEEVE_BOTTOM
    if abs(z - H_SLEEVE) < 1e-7:
        return SLEEVE_TOP
    return BORE if abs(rho - R_ROD) < 1e-7 else OUTER


def max_tolerance(shapes):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopoDS import TopoDS
    worst = 0.0
    for shape in shapes:
        worst = max([worst] + [BRep_Tool.Tolerance_s(TopoDS.Edge_s(e)) for e in explore(shape, TopAbs_EDGE)]
                    + [BRep_Tool.Tolerance_s(TopoDS.Vertex_s(v)) for v in explore(shape, TopAbs_VERTEX)])
    return worst


# ------------------------------------------------------------------ the partition alone (no OCC)

def ring_faces(spec):
    """`spec` = {edge: [(coedge, face, reversed), ...]} -> (faces as objects with an `index`, rings)."""
    indices = sorted({face for ring in spec.values() for _, face, _ in ring})
    return [SimpleNamespace(index=i) for i in indices], spec


class PartitionTests(unittest.TestCase):
    def split(self, spec):
        from cq_acis._nonmanifold_split import split_bodies
        faces, rings = ring_faces(spec)
        bodies = split_bodies(faces, rings)
        return None if bodies is None else [sorted(f.index for f in body) for body in bodies]

    def test_22b8_like_ring_has_one_body_consistent_partition(self):
        """The fixture's rings (rod top a body of its own over the manifold edges): exactly one pairing qualifies."""
        spec = {
            "ring": [(10, 3, True), (11, 4, False), (12, 6, True), (13, 7, False)],
            "rod_rim": [(14, 3, False), (15, 5, True)], "bore_rim": [(16, 6, False), (17, 8, True)],
            "outer_low": [(18, 7, False), (19, 9, True)], "outer_high": [(20, 8, True), (21, 9, False)],
        }
        self.assertEqual(self.split(spec), [[3, 4, 5], [6, 7, 8, 9]])

    def test_ambiguous_partition_refused(self):
        """Four faces meeting only at one 4-coedge ring, senses alternating: two valid pairings -> refused."""
        spec = {"ring": [(10, 3, True), (11, 4, False), (12, 5, True), (13, 6, False)]}
        self.assertIsNone(self.split(spec))

    def test_odd_coedge_ring_refused(self):
        spec = {"ring": [(10, 3, True), (11, 4, False), (12, 5, True)],
                "a": [(14, 3, False), (15, 4, True)]}
        self.assertIsNone(self.split(spec))

    def test_same_body_ring_refused(self):
        """All 4 coedges of the ring in one body (the faces are also joined by manifold edges) -> refused."""
        spec = {"ring": [(10, 3, True), (11, 4, False), (12, 5, True), (13, 6, False)],
                "a": [(14, 3, False), (15, 4, True)], "b": [(16, 4, True), (17, 5, False)],
                "c": [(18, 5, False), (19, 6, True)]}
        self.assertIsNone(self.split(spec))

    def test_same_sense_pairs_refused(self):
        """Two bodies, but each body's two coedges of the ring run the same way: no manifold pair -> refused."""
        spec = {"ring": [(10, 3, True), (11, 4, True), (12, 5, False), (13, 6, False)],
                "a": [(14, 3, False), (15, 4, True)], "b": [(16, 5, False), (17, 6, True)]}
        self.assertIsNone(self.split(spec))

    def test_manifold_rings_are_not_split(self):
        spec = {"a": [(10, 3, True), (11, 4, False)], "b": [(12, 3, False), (13, 4, True)]}
        self.assertIsNone(self.split(spec))


# ------------------------------------------------------------------ the converter

@unittest.skipUnless(OCP_AVAILABLE, "needs OCP")
class NonmanifoldSplitTests(unittest.TestCase):
    def test_fixture_faces_are_valid(self):
        """Characterisation: every fixture face is a valid face."""
        from OCP.BRepCheck import BRepCheck_Analyzer
        for cut in (False, True):
            for x in (1.0, -1.0):
                for index, face in fixture_faces(rod_side_cut=cut, sleeve_xdir=x).items():
                    with self.subTest(face=index, rod_side_cut=cut, sleeve_xdir=x):
                        self.assertTrue(BRepCheck_Analyzer(face).IsValid())

    def test_two_body_ring_splits_into_two_positive_solids(self):
        self.assert_split_into_two_positive_solids(fixture_faces(), loop_edge_splits=0)

    def test_loop_cut_differently_per_body_is_shared(self):
        """22B8: the rod side carries the ring as two halves, every other face as one circle, so the rod's sewn loop
        has 2 edges and the sleeve's 1. The sleeve's loop edge is cut at the rod's loop vertex (on its own curve,
        within that vertex's own tolerance; no re-fit, no tolerance raised), then both solids share 2 edges."""
        c = self.assert_split_into_two_positive_solids(fixture_faces(rod_side_cut=True), loop_edge_splits=1)
        self.assertEqual(c.nonmanifold_splits[0]["shared_loop_edges"], 2)
        self.assertEqual(c.nonmanifold_splits[0]["parameter_shifts"], [])

    def test_loop_parameterised_differently_per_body_is_shared(self):
        """22B8 (native dev loop): the bodies' loop circles coincide but start at -x and +x, an offset of pi in the
        parameter. Each loop is cut at the other's vertices; the second body's pcurves are re-expressed exactly by
        that shift (a line moved along itself, a circle's frame turned; no fit), within the edge's own tolerance."""
        for cut, splits in ((False, 2), (True, 1)):
            with self.subTest(rod_side_cut=cut):
                c = self.assert_split_into_two_positive_solids(
                    fixture_faces(rod_side_cut=cut, sleeve_xdir=-1.0), loop_edge_splits=splits)
                split = c.nonmanifold_splits[0]
                self.assertEqual(split["shared_loop_edges"], 2)
                self.assertEqual(len(split["parameter_shifts"]), 2)
                for shift in split["parameter_shifts"]:
                    self.assertAlmostEqual(abs(shift["shift"]), math.pi, delta=1e-12)

    def assert_split_into_two_positive_solids(self, faces, loop_edge_splits):
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.BRepCheck import BRepCheck_NoError, BRepCheck_Shell
        from OCP.GCPnts import GCPnts_AbscissaPoint
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
        from OCP.TopoDS import TopoDS
        c, model, result = convert_lump(TWO_BODY_LOOPS, faces)
        self.assertNotIsInstance(result, CadQueryConversionError, msg=str(result))
        self.assertTrue(result.isValid())
        solids = sorted(explore(result.wrapped, TopAbs_SOLID), key=lambda s: c.cq.Solid(s).Volume())
        self.assertEqual(len(solids), 2)
        volumes = []
        for solid, expected in zip(solids, sorted([V_ROD, V_SLEEVE])):
            shells = explore(solid, TopAbs_SHELL)
            self.assertEqual(len(shells), 1)
            check = BRepCheck_Shell(TopoDS.Shell_s(shells[0]))
            self.assertTrue(BRep_Tool.IsClosed_s(shells[0]))
            self.assertEqual(check.Closed(), BRepCheck_NoError)
            self.assertEqual(check.Orientation(), BRepCheck_NoError)
            volume = shell_volume(c.cq.Shell(shells[0]))  # as oriented: no orientation fix hides an inversion
            self.assertAlmostEqual(volume / expected, 1.0, delta=1e-6)
            self.assertAlmostEqual(c.cq.Solid(solid).Volume() / expected, 1.0, delta=1e-6)
            volumes.append(volume)
        # As-built face orientation kept on every face: each final face's normal is the source face's outward one.
        final_faces = explore(result.wrapped, TopAbs_FACE)
        self.assertEqual(len(final_faces), len(faces))
        seen = set()
        for face in final_faces:
            point, normal = oriented_normal(face)
            index = source_face_of(point)
            seen.add(index)
            expected = outward(point, index)
            self.assertGreater(sum(a * b for a, b in zip(normal, expected)), 0.999, msg=f"face {index}")
        self.assertEqual(seen, set(faces))
        # The loop edge shared: the two solids use the same TopoDS_Edges on the circle z = 0, r = R_ROD, together
        # covering it once, and share nothing else.
        edges = [explore(solid, TopAbs_EDGE) for solid in solids]
        common = [e for e in edges[0] if any(e.IsSame(o) for o in edges[1])]
        self.assertTrue(common)
        length = 0.0
        for edge in common:
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(edge))
            first, last = curve.FirstParameter(), curve.LastParameter()
            for k in range(9):
                p = curve.Value(first + (last - first) * k / 8)
                self.assertLess(abs(p.Z()), 1e-9)
                self.assertLess(abs(math.hypot(p.X(), p.Y()) - R_ROD), 1e-9)
            length += GCPnts_AbscissaPoint.Length_s(curve)
        self.assertAlmostEqual(length / (2 * math.pi * R_ROD), 1.0, delta=1e-9)
        vertices = [explore(solid, TopAbs_VERTEX) for solid in solids]
        for vertex in (v for v in vertices[0] if any(v.IsSame(o) for o in vertices[1])):
            p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertex))
            self.assertLess(abs(p.Z()), 1e-9)
            self.assertLess(abs(math.hypot(p.X(), p.Y()) - R_ROD), 1e-9)
        # No tolerance was raised.
        self.assertLessEqual(max_tolerance([result.wrapped]), max_tolerance(list(faces.values())))
        # Recorded (no dwg2glb schema change): one split per shell, with its edges, bodies and volumes.
        _, edge_ids = topology_model(TWO_BODY_LOOPS)
        self.assertEqual(len(c.nonmanifold_splits), 1)
        split = c.nonmanifold_splits[0]
        self.assertEqual(split["shell"], SHELL)
        self.assertEqual(split["edges"], [edge_ids[RING]])
        self.assertEqual(split["bodies"], [list(ROD), list(SLEEVE)])
        self.assertEqual(len(split["volumes"]), 2)
        self.assertAlmostEqual(split["volumes"][0] / V_ROD, 1.0, delta=1e-6)
        self.assertAlmostEqual(split["volumes"][1] / V_SLEEVE, 1.0, delta=1e-6)
        self.assertEqual(split["shared_loop_edges"], len(common))
        self.assertEqual(len(split["loop_edge_splits"]), loop_edge_splits)
        for cut in split["loop_edge_splits"]:
            self.assertLessEqual(cut["gap_mm"], cut["vertex_tolerance_mm"])
        self.assertEqual(c.nonmanifold_split_refusals, [])
        self.assertEqual(sorted(r["component"] for r in c.shell_closure_checks), [0, 1])
        return c

    def test_loop_cut_beyond_vertex_tolerance_refused(self):
        """The loop cut is bounded by the cutting vertex's OWN tolerance (R206), never the cut edge's: a vertex 3e-7
        mm off the circle is refused against its 1e-7 mm tolerance although the edge carries 1e-6 mm; 3e-8 mm is
        cut."""
        from OCP.BRep import BRep_Builder
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.TopoDS import TopoDS_Vertex
        from cq_acis import _nonmanifold_split as S
        builder = BRep_Builder()
        whole = BRepBuilderAPI_MakeEdge(_circle(0.0, R_ROD)).Edge()
        builder.UpdateEdge(whole, 1e-6)

        def halves(radius):
            circle = _circle(0.0, radius)
            ends = []
            for t in (math.pi / 2, 3 * math.pi / 2):
                vertex = TopoDS_Vertex()
                builder.MakeVertex(vertex, circle.Value(t), TOL)
                ends.append(vertex)
            return [BRepBuilderAPI_MakeEdge(circle, ends[0], ends[1], math.pi / 2, 3 * math.pi / 2).Edge(),
                    BRepBuilderAPI_MakeEdge(circle, ends[1], ends[0], 3 * math.pi / 2, 5 * math.pi / 2).Edge()]

        with self.assertRaises(S._Refused) as caught:
            S._cuts_for(1, [whole], halves(R_ROD + 3e-7))
        self.assertIn("loop cut refused: gap", str(caught.exception))
        cuts = S._cuts_for(1, [whole], halves(R_ROD + 3e-8))
        ((edge, items),) = cuts.values()
        self.assertTrue(edge.IsSame(whole))
        self.assertEqual(len(items), 2)
        for parameter, vertex, gap in items:
            self.assertLessEqual(gap, TOL)

    def test_pcurve_shift_is_exact_and_limited_to_lines_and_circles(self):
        from OCP.Geom2d import Geom2d_BSplineCurve, Geom2d_Circle, Geom2d_Line
        from OCP.TColgp import TColgp_Array1OfPnt2d
        from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
        from OCP.gp import gp_Ax22d, gp_Circ2d, gp_Dir2d, gp_Pnt2d
        from cq_acis import _nonmanifold_split as S
        shift = math.pi
        curves = [Geom2d_Line(gp_Pnt2d(0.5, -1.0), gp_Dir2d(1, 0)),
                  Geom2d_Circle(gp_Circ2d(gp_Ax22d(gp_Pnt2d(0.1, 0.2), gp_Dir2d(-1, 0), gp_Dir2d(0, -1)), 2.0)),
                  Geom2d_Circle(gp_Circ2d(gp_Ax22d(gp_Pnt2d(0.1, 0.2), gp_Dir2d(1, 0), gp_Dir2d(0, -1)), 2.0))]
        for curve in curves:
            moved = S._shifted(curve, shift)
            for k in range(9):
                t = 0.7 * k
                self.assertLess(moved.Value(t).Distance(curve.Value(t + shift)), 1e-12)
        self.assertIs(S._shifted(curves[0], 0.0), curves[0])
        poles = TColgp_Array1OfPnt2d(1, 2)
        poles.SetValue(1, gp_Pnt2d(0, 0))
        poles.SetValue(2, gp_Pnt2d(1, 0))
        knots, mults = TColStd_Array1OfReal(1, 2), TColStd_Array1OfInteger(1, 2)
        for i, (knot, mult) in enumerate(((0.0, 2), (1.0, 2)), start=1):
            knots.SetValue(i, knot)
            mults.SetValue(i, mult)
        with self.assertRaises(S._Refused):
            S._shifted(Geom2d_BSplineCurve(poles, knots, mults, 1), shift)

    def test_manifold_shell_never_enters_split(self):
        """The census assertion in code form: a shell without a > 2-coedge source edge never reaches the split."""
        refused = AssertionError("the split was entered for a manifold shell")
        with mock.patch(f"{SPLIT}.split_bodies", side_effect=refused), \
                mock.patch(f"{SPLIT}.split_nonmanifold_shell", side_effect=refused):
            c, outcome = shell_outcome(ROD_LOOPS, {face: fixture_faces()[face] for face in ROD})
        self.assertEqual(outcome, todays_path(ROD_LOOPS, {face: fixture_faces()[face] for face in ROD})[0])
        self.assertAlmostEqual(outcome[1][0] / V_ROD, 1.0, delta=1e-6)
        self.assertEqual(len(outcome[1]), 1)
        self.assertEqual((c.nonmanifold_splits, c.nonmanifold_split_refusals), ([], []))

    def refused_like_today(self, faces, reason, face_loops=TWO_BODY_LOOPS, model=None):
        """The split is refused with `reason` in its record, and `_shell` ends exactly as today's path."""
        c, outcome = shell_outcome(face_loops, faces, model)
        expected, records = todays_path(face_loops, faces)
        self.assertEqual(outcome, expected)
        self.assertEqual(c.nonmanifold_splits, [])
        self.assertEqual(len(c.nonmanifold_split_refusals), 1)
        self.assertEqual(c.nonmanifold_split_refusals[0]["shell"], SHELL)
        self.assertIn(reason, c.nonmanifold_split_refusals[0]["reason"])
        self.assertEqual(len(c.shell_closure_checks), records)  # the refused attempt leaves no closure record
        return c, outcome

    def test_open_body_after_split_refused(self):
        """The rod without its bottom disk: the partition qualifies, the rod body does not close -> today's path."""
        loops = {face: loops for face, loops in TWO_BODY_LOOPS.items() if face != ROD_BOTTOM}
        faces = fixture_faces()
        renumbered_loops, renumbered_faces = {}, {}
        for new, old in enumerate(sorted(loops), start=3):
            renumbered_loops[new] = loops[old]
            renumbered_faces[new] = faces[old]
        self.refused_like_today(renumbered_faces, "not closed", renumbered_loops)

    def test_negative_body_after_split_refused(self):
        """Every sleeve face inverted: the sleeve body sews closed and oriented but encloses a negative volume."""
        self.refused_like_today(fixture_faces(invert_sleeve=True), "volume")

    def test_occ_error_fails_closed(self):
        from OCP.Standard import Standard_Failure
        with mock.patch(f"{SPLIT}._sew_body", side_effect=Standard_Failure("injected")):
            self.refused_like_today(fixture_faces(), "Standard_Failure")

    def test_cavity_shell_in_the_lump_refused(self):
        """A lump with a further (cavity) shell: the split is refused, so the lump reports as today."""
        model, _ = topology_model(TWO_BODY_LOOPS)
        entities = list(model.entities)
        shell = entities[SHELL]
        extra = len(entities)
        entities.append(ShellEntity(raw(extra, "shell"), NULL_REF, NULL_REF, NULL_REF, NULL_REF, NULL_REF,
                                    EntityRef(LUMP)))
        entities[SHELL] = ShellEntity(shell.raw, shell.pattern, EntityRef(extra), shell.subshell, shell.face,
                                      shell.wire, shell.lump)
        model = AcisModel(model.metadata, tuple(entities))
        self.refused_like_today(fixture_faces(), "cavity", model=model)


if __name__ == "__main__":
    unittest.main()
