"""Analytic arcs whose saved parameters miss their source vertices by more
than OCCT's confusion (1e-7) but within the model resolution are seated on those vertices, so the loop closes.

The real loop: two half-ellipses on a cone face share two vertices (gap 0.0 in the source); the arcs built from the
saved parameters end 1.3e-7..1.9e-7 mm from them (resabs 1e-6), and Wire.assembleEdges left the wire open.
"""
import math
import unittest

from cq_acis import (
    AcisMetadata, AcisModel, CadQueryConverter, CadQueryConversionError, CoedgeEntity, EdgeEntity,
    EllipseCurveEntity, EntityRef, LoopEntity, NULL_REF, PointEntity, RawEntity, StraightCurveEntity, Vec3,
    VertexEntity,
)
from cq_acis.cadquery import _Placement

RADIUS = 10.0


def raw(i, name):
    return RawEntity(i, name, NULL_REF, None)


def ref(i):
    return EntityRef(i)


def circle_loop(miss_rad: float, resabs: float = 1e-6):
    """loop 0 -> coedges 1, 2 -> edges 3, 4 on circle 5 (radius 10 about z) between vertices 6 (+x) and 7 (-x),
    points 8 and 9. Edge 3 runs 0..pi but starts ``miss_rad`` late, so its arc misses vertex 6 by ~10 * miss_rad."""
    entities = (
        LoopEntity(raw(0, "loop"), NULL_REF, NULL_REF, ref(1), NULL_REF),
        CoedgeEntity(raw(1, "coedge"), NULL_REF, ref(2), ref(2), NULL_REF, ref(3), False, ref(0), NULL_REF),
        CoedgeEntity(raw(2, "coedge"), NULL_REF, ref(1), ref(1), NULL_REF, ref(4), False, ref(0), NULL_REF),
        EdgeEntity(raw(3, "edge"), NULL_REF, ref(6), miss_rad, ref(7), math.pi, ref(1), ref(5), False, "unknown"),
        EdgeEntity(raw(4, "edge"), NULL_REF, ref(7), math.pi, ref(6), 2 * math.pi, ref(2), ref(5), False, "unknown"),
        EllipseCurveEntity(raw(5, "ellipse-curve"), NULL_REF, Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(RADIUS, 0, 0), 1.0,
                           None),
        VertexEntity(raw(6, "vertex"), NULL_REF, ref(3), ref(8)),
        VertexEntity(raw(7, "vertex"), NULL_REF, ref(3), ref(9)),
        PointEntity(raw(8, "point"), NULL_REF, Vec3(RADIUS, 0, 0)),
        PointEntity(raw(9, "point"), NULL_REF, Vec3(-RADIUS, 0, 0)),
    )
    return CadQueryConverter(AcisModel(AcisMetadata(units_mm=1, resabs=resabs), entities)), entities


class ArcVertexSeatTests(unittest.TestCase):
    def test_an_arc_within_resolution_of_its_vertices_closes_the_loop(self):
        converter, entities = circle_loop(4e-8)  # 4e-7 mm: above OCCT's 1e-7, below resabs 1e-6
        wire = converter._wire(entities[0], _Placement(None, 1.0))
        self.assertTrue(wire.IsClosed() and wire.isValid())
        seats = converter.edge_vertex_seats
        self.assertEqual([(s["edge"], s["vertex"]) for s in seats], [(3, 6)])
        self.assertAlmostEqual(seats[0]["gap_mm"], RADIUS * 4e-8, delta=1e-12)
        self.assertLessEqual(seats[0]["gap_mm"], seats[0]["tolerance_mm"])
        self.assertLessEqual(seats[0]["tolerance_mm"], converter.tolerance)
        # The vertex is on the source point; its tolerance covers the arc's own end.
        start = [v for v in converter._edge(entities[1], _Placement(None, 1.0)).Vertices()
                 if abs(v.Center().x - RADIUS) < 1e-3]
        self.assertTrue(start and max(abs(a - b) for a, b in zip(start[0].Center().toTuple(), (RADIUS, 0, 0))) < 1e-12)

    def test_an_exact_arc_is_not_touched(self):
        converter, entities = circle_loop(0.0)
        self.assertTrue(converter._wire(entities[0], _Placement(None, 1.0)).IsClosed())
        self.assertEqual(converter.edge_vertex_seats, [])

    def test_an_arc_beyond_the_resolution_is_not_seated(self):
        converter, entities = circle_loop(3e-7)  # 3e-6 mm > resabs 1e-6: the source does not close; refuse
        with self.assertRaises(CadQueryConversionError) as caught:
            converter._wire(entities[0], _Placement(None, 1.0))
        self.assertIn("wire is not closed", str(caught.exception))
        self.assertEqual(converter.edge_vertex_seats, [])


def saved_line(miss_mm: float):
    """edge 1 on line 2 (origin 0, direction +x) from vertex 3 (0,0,0) to vertex 4 (10,0,0), saved end parameter
    10 - miss_mm: the converter's tolerant-line path (_straight_edge with the curve) builds from the saved
    parameters. treppenturm 208A676: lines meeting at tvertex $5637 end 8.4e-7 mm apart."""
    entities = (
        CoedgeEntity(raw(0, "coedge"), NULL_REF, ref(0), ref(0), NULL_REF, ref(1), False, NULL_REF, NULL_REF),
        EdgeEntity(raw(1, "edge"), NULL_REF, ref(3), 0.0, ref(4), 10.0 - miss_mm, ref(0), ref(2), False, "unknown"),
        StraightCurveEntity(raw(2, "straight-curve"), NULL_REF, Vec3(0, 0, 0), Vec3(1, 0, 0), None),
        VertexEntity(raw(3, "vertex"), NULL_REF, ref(1), ref(5)),
        VertexEntity(raw(4, "vertex"), NULL_REF, ref(1), ref(6)),
        PointEntity(raw(5, "point"), NULL_REF, Vec3(0, 0, 0)),
        PointEntity(raw(6, "point"), NULL_REF, Vec3(10, 0, 0)),
    )
    return CadQueryConverter(AcisModel(AcisMetadata(units_mm=1, resabs=1e-6), entities)), entities


class SavedLineVertexSeatTests(unittest.TestCase):
    def build(self, miss_mm):
        converter, e = saved_line(miss_mm)
        edge = converter._straight_edge(e[1], e[0], _Placement(None, 1.0), curve=e[2])
        return converter, edge

    def test_a_saved_line_end_within_resolution_is_seated_on_its_vertex(self):
        converter, edge = self.build(5e-7)
        ends = sorted(v.Center().x for v in edge.Vertices())
        self.assertEqual(ends[-1], 10.0)
        self.assertEqual([(s["edge"], s["vertex"]) for s in converter.edge_vertex_seats], [(1, 4)])
        self.assertAlmostEqual(converter.edge_vertex_seats[0]["gap_mm"], 5e-7, delta=1e-12)

    def test_a_saved_line_end_beyond_the_resolution_is_left_alone(self):
        converter, edge = self.build(3e-6)
        self.assertLess(max(v.Center().x for v in edge.Vertices()), 10.0 - 2e-6)
        self.assertEqual(converter.edge_vertex_seats, [])


if __name__ == "__main__":
    unittest.main()
