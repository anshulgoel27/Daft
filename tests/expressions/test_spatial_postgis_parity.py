"""Regression tests for daft-geo correctness fixes found while auditing against PostGIS.

Each test names the defect it guards. Coordinates are synthetic (near -77.03, 38.90 for
lon/lat cases, so feature sizes match real parcels and buildings).
"""

from __future__ import annotations

import io
import math
import struct

import pytest

import daft
from daft import col, lit
from daft.functions import (
    great_circle_distance,
    list_count,
    st_area,
    st_astext,
    st_buffer,
    st_contains,
    st_convexhull,
    st_difference,
    st_distance,
    st_dump,
    st_dwithin,
    st_envelope,
    st_equals,
    st_geohash,
    st_geometrytype,
    st_geomfromtext,
    st_intersection,
    st_intersects,
    st_isvalid,
    st_makeline,
    st_normalize,
    st_simplify,
    st_symdifference,
    st_union,
    st_within,
    st_x,
    st_y,
)


def _g(name: str = "g"):
    return st_geomfromtext(col(name))


def _rect(w, h, x0=-77.03, y0=38.90):
    return f"POLYGON(({x0} {y0},{x0 + w} {y0},{x0 + w} {y0 + h},{x0} {y0 + h},{x0} {y0}))"


SQUARE = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
POINTS = ["POINT(0.5 0.5)", "POINT(5 5)", "POINT(0.2 0.2)"]


# ── Scalar broadcast on the LEFT operand (used to repeat row 0's answer) ─────


def test_literal_lhs_predicate_is_broadcast():
    df = daft.from_pydict({"p": POINTS})
    out = df.select(st_contains(st_geomfromtext(lit(SQUARE)), _g("p")).alias("c")).to_pydict()
    assert out["c"] == [True, False, True]


def test_literal_lhs_distance_is_broadcast():
    df = daft.from_pydict({"p": POINTS})
    out = df.select(st_distance(st_geomfromtext(lit(SQUARE)), _g("p")).alias("d")).to_pydict()
    assert out["d"] == pytest.approx([0.0, math.hypot(4, 4), 0.0])


def test_literal_lhs_overlay_and_makeline_are_broadcast():
    df = daft.from_pydict({"p": POINTS})
    out = df.select(
        st_astext(st_intersection(st_geomfromtext(lit(SQUARE)), _g("p"))).alias("i"),
        st_astext(st_makeline(st_geomfromtext(lit("POINT(0 0)")), _g("p"))).alias("l"),
    ).to_pydict()
    assert out["i"] == ["POINT(0.5 0.5)", "GEOMETRYCOLLECTION EMPTY", "POINT(0.2 0.2)"]
    assert out["l"] == ["LINESTRING(0 0,0.5 0.5)", "LINESTRING(0 0,5 5)", "LINESTRING(0 0,0.2 0.2)"]


def test_great_circle_distance_broadcasts_any_argument():
    df = daft.from_pydict({"lat": [0.0, 1.0, 2.0], "lon": [0.0, 0.0, 0.0]})
    lhs_lit = df.select(great_circle_distance(lit(0.0), lit(0.0), col("lat"), col("lon")).alias("d")).to_pydict()
    rhs_lit = df.select(great_circle_distance(col("lat"), col("lon"), lit(0.0), lit(0.0)).alias("d")).to_pydict()
    one_degree = 6_371_000.0 * math.pi / 180
    assert lhs_lit["d"] == pytest.approx([0.0, one_degree, 2 * one_degree])
    assert rhs_lit["d"] == lhs_lit["d"]


# ── Geohash pruning must not drop polygon rows whose centroid is in another cell ─
#
# GeohashPruning only fires for a raw WKB literal query (not st_geomfromtext(lit(...))).


def _point_wkb(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, x, y)


def _rect_wkb(x0: float, y0: float, x1: float, y1: float) -> bytes:
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return struct.pack("<BII", 1, 3, 1) + struct.pack("<I", len(ring)) + b"".join(struct.pack("<dd", *c) for c in ring)


def _explain(df) -> str:
    buf = io.StringIO()
    df.explain(show_all=True, file=buf)
    return buf.getvalue()


def _straddling_parcel_df():
    # Precision-5 geohash cells have a lon boundary at -77.0361328125. This 17 m wide
    # parcel straddles it; its centroid is east of the boundary.
    parcel = "POLYGON((-77.0362 38.9,-77.036 38.9,-77.036 38.9001,-77.0362 38.9001,-77.0362 38.9))"
    df = daft.from_pydict({"wkt": [parcel]}).with_column("geom", st_geomfromtext(col("wkt")))
    return df.with_column("geom_geohash", st_geohash(col("geom"), precision=7))


@pytest.mark.parametrize("pred", [st_intersects, st_contains])
def test_geohash_pruning_keeps_polygon_rows(pred):
    # Query point inside the parcel, WEST of the cell boundary.
    query = lit(_point_wkb(-77.03619, 38.90005))
    filtered = _straddling_parcel_df().where(pred(col("geom"), query))
    # The rule fired, with the geometry argument that lets non-Point rows through ...
    assert "st_geohash_covers(col(geom_geohash), col(geom))" in _explain(filtered)
    assert filtered.count_rows() == 1  # ... and did not drop the row


def test_geohash_pruning_within_still_pruned_and_correct():
    df = _straddling_parcel_df()
    big = lit(_rect_wkb(-77.1, 38.8, -76.9, 39.0))
    far = lit(_rect_wkb(-70.1, 30.8, -69.9, 31.0))
    # within: centroid-bounded for every geometry type, so no geometry argument.
    assert "st_geohash_covers(col(geom_geohash))" in _explain(df.where(st_within(col("geom"), big)))
    assert df.where(st_within(col("geom"), big)).count_rows() == 1
    assert df.where(st_within(col("geom"), far)).count_rows() == 0


def test_geohash_pruning_still_applies_to_point_rows():
    pts = ["POINT(-77.03 38.90)", "POINT(-70.0 30.0)"]
    df = daft.from_pydict({"wkt": pts}).with_column("geom", st_geomfromtext(col("wkt")))
    df = df.with_column("geom_geohash", st_geohash(col("geom"), precision=7))
    query = lit(_rect_wkb(-77.0305, 38.8995, -77.0295, 38.9005))
    filtered = df.where(st_intersects(col("geom"), query))
    assert "st_geohash_covers" in _explain(filtered)
    assert filtered.count_rows() == 1


# ── st_buffer: scale-independent, round joins, all geometry types ────────────


@pytest.mark.parametrize("side", [3e-5, 1e-5])
def test_buffer_small_lonlat_polygon_outward(side):
    df = daft.from_pydict({"g": [_rect(side, side)]})
    d = 1e-5
    out = df.select(st_area(st_buffer(_g(), d)).alias("a")).to_pydict()
    # Square plus four edge strips plus four quarter-circles (32-gon approximation).
    expected = side * side + 4 * side * d + math.pi * d * d
    assert out["a"][0] == pytest.approx(expected, rel=1e-2)


def test_buffer_small_lonlat_polygon_inward():
    df = daft.from_pydict({"g": [_rect(3e-5, 3e-5)]})
    out = df.select(
        st_area(st_buffer(_g(), -1e-5)).alias("shrunk"),
        st_astext(st_buffer(_g(), -2e-5)).alias("gone"),
    ).to_pydict()
    assert out["shrunk"][0] == pytest.approx(1e-10, rel=1e-3)
    assert out["gone"] == ["MULTIPOLYGON EMPTY"]


def test_buffer_scaled_geometry_scales_area():
    """Buffering is scale invariant: scaling coordinates by k scales the area by k^2."""
    small = daft.from_pydict({"g": [_rect(3e-5, 3e-5, 0.0, 0.0)]})
    big = daft.from_pydict({"g": [_rect(3.0, 3.0, 0.0, 0.0)]})
    a_small = small.select(st_area(st_buffer(_g(), 1e-5)).alias("a")).to_pydict()["a"][0]
    a_big = big.select(st_area(st_buffer(_g(), 1.0)).alias("a")).to_pydict()["a"][0]
    assert a_small * 1e10 == pytest.approx(a_big, rel=1e-6)


def test_buffer_linestring_is_a_corridor():
    df = daft.from_pydict({"g": ["LINESTRING(0 0,10 0)"]})
    out = df.select(
        st_area(st_buffer(_g(), 1.0)).alias("a"), st_geometrytype(st_buffer(_g(), 1.0)).alias("t")
    ).to_pydict()
    assert out["t"] == ["Polygon"]
    assert out["a"][0] == pytest.approx(20.0 + 16 * math.sin(2 * math.pi / 32), rel=1e-3)


def test_buffer_multipoint_and_collection():
    df = daft.from_pydict({"g": ["MULTIPOINT((0 0),(10 10))", "GEOMETRYCOLLECTION(POINT(0 0),LINESTRING(10 0,20 0))"]})
    out = df.select(st_area(st_buffer(_g(), 1.0)).alias("a")).to_pydict()
    circle = 16 * math.sin(2 * math.pi / 32)
    assert out["a"][0] == pytest.approx(2 * circle, rel=1e-3)
    assert out["a"][1] == pytest.approx(circle + 20.0 + circle, rel=1e-3)


@pytest.mark.parametrize("wkt", ["POINT(1 2)", "LINESTRING(0 0,1 1)"])
@pytest.mark.parametrize("d", [0.0, -1.0])
def test_buffer_non_areal_by_nonpositive_distance_is_empty(wkt, d):
    df = daft.from_pydict({"g": [wkt]})
    out = df.select(st_astext(st_buffer(_g(), d)).alias("w")).to_pydict()
    assert out["w"] == ["MULTIPOLYGON EMPTY"]


def test_buffer_polygon_uses_round_joins():
    """Triangle (0 0, 10 0, 0 1) buffered by 1: PostGIS gives ~29.2; mitre joins gave 48.2."""
    df = daft.from_pydict({"g": ["POLYGON((0 0,10 0,0 1,0 0))"]})
    out = df.select(st_area(st_buffer(_g(), 1.0)).alias("a")).to_pydict()
    assert out["a"][0] == pytest.approx(29.2, abs=0.2)


def test_buffer_never_writes_corrupt_wkb():
    df = daft.from_pydict({"g": ["LINESTRING(0 0,10 10)", "MULTIPOINT((0 0),(2 2))"]})
    out = df.select(st_isvalid(st_buffer(_g(), 1.0)).alias("v"), st_area(st_buffer(_g(), 1.0)).alias("a")).to_pydict()
    assert out["v"] == [True, True]
    assert all(a > 0 for a in out["a"])


# ── st_distance / st_dwithin for every geometry pair ─────────────────────────


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)),((5 5,6 5,6 6,5 6,5 5)))", "POINT(3 0.5)", 2.0),
        ("LINESTRING(0 0,1 0)", "LINESTRING(0 3,1 3)", 3.0),
        (SQUARE, "LINESTRING(3 0,3 1)", 2.0),
        ("MULTIPOINT((0 0),(10 0))", "POINT(7 0)", 3.0),
        (SQUARE, "MULTIPOLYGON(((4 0,5 0,5 1,4 1,4 0)))", 3.0),
        (SQUARE, "POINT(0.5 0.5)", 0.0),
    ],
)
def test_distance_all_pairs(a, b, expected):
    df = daft.from_pydict({"a": [a], "b": [b]})
    out = df.select(
        st_distance(_g("a"), _g("b")).alias("d"),
        st_dwithin(_g("a"), _g("b"), expected + 1e-9).alias("w"),
        st_dwithin(_g("a"), _g("b"), expected - 0.5).alias("not_w"),
    ).to_pydict()
    assert out["d"][0] == pytest.approx(expected)
    assert out["w"] == [True]
    assert out["not_w"] == [False]


def test_distance_geodesic_polygon_point():
    # Point 0.001° due east of the parcel's east edge at lat 38.90: ~86.6 m.
    df = daft.from_pydict({"a": [_rect(1e-4, 1e-4)], "b": ["POINT(-77.0289 38.90005)"]})
    out = df.select(
        st_distance(_g("a"), _g("b"), use_spheroid=True).alias("d"),
        st_distance(st_geomfromtext(lit("POINT(-77.0299 38.90005)")), _g("b"), use_spheroid=True).alias("pp"),
    ).to_pydict()
    assert out["d"][0] == pytest.approx(out["pp"][0], rel=1e-6)
    assert out["d"][0] == pytest.approx(86.7, abs=0.5)


def test_distance_empty_is_null():
    df = daft.from_pydict({"a": ["MULTIPOLYGON EMPTY"], "b": ["POINT(0 0)"]})
    out = df.select(st_distance(_g("a"), _g("b")).alias("d")).to_pydict()
    assert out["d"] == [None]


# ── Overlays: exact input vertices, single part as POLYGON, mixed types ──────


def test_intersection_preserves_input_vertices():
    parcel = _rect(6e-5, 6e-5)
    region = "POLYGON((-100 10,-40 10,-40 60,-100 60,-100 10))"
    df = daft.from_pydict({"p": [parcel], "r": [region]})
    out = df.select(
        st_equals(st_intersection(_g("p"), _g("r")), _g("p")).alias("eq"),
        st_astext(st_normalize(st_intersection(_g("p"), _g("r")))).alias("got"),
        st_astext(st_normalize(_g("p"))).alias("want"),
    ).to_pydict()
    assert out["eq"] == [True]
    assert out["got"] == out["want"]


def test_areal_overlay_output_types():
    a, b = "POLYGON((0 0,2 0,2 2,0 2,0 0))", "POLYGON((1 1,3 1,3 3,1 3,1 1))"
    far = "POLYGON((10 10,11 10,11 11,10 11,10 10))"
    df = daft.from_pydict({"a": [a, a], "b": [b, far]})
    out = df.select(
        st_geometrytype(st_union(_g("a"), _g("b"))).alias("u"),
        st_geometrytype(st_intersection(_g("a"), _g("b"))).alias("i"),
        st_geometrytype(st_symdifference(_g("a"), _g("b"))).alias("x"),
        st_astext(st_intersection(_g("a"), _g("b"))).alias("i_wkt"),
    ).to_pydict()
    assert out["u"] == ["Polygon", "MultiPolygon"]
    assert out["i"][0] == "Polygon"
    assert out["i_wkt"][1] == "MULTIPOLYGON EMPTY"
    assert out["x"] == ["MultiPolygon", "MultiPolygon"]


@pytest.mark.parametrize(
    "fn,a,b,expected",
    [
        (st_intersection, "LINESTRING(-1 0.5,2 0.5)", SQUARE, "LINESTRING(0 0.5,1 0.5)"),
        (st_difference, "LINESTRING(-1 0.5,2 0.5)", SQUARE, "MULTILINESTRING((-1 0.5,0 0.5),(1 0.5,2 0.5))"),
        (st_intersection, "POINT(0.5 0.5)", SQUARE, "POINT(0.5 0.5)"),
        (st_difference, "POINT(0.5 0.5)", SQUARE, "GEOMETRYCOLLECTION EMPTY"),
        (st_union, "POINT(0.5 0.5)", SQUARE, "POLYGON((0 0,0 1,1 1,1 0,0 0))"),
        (st_union, "POINT(5 5)", SQUARE, "GEOMETRYCOLLECTION(POLYGON((0 0,0 1,1 1,1 0,0 0)),POINT(5 5))"),
        (st_intersection, "LINESTRING(0 0,2 2)", "LINESTRING(0 2,2 0)", "POINT(1 1)"),
        (st_difference, SQUARE, "LINESTRING(0 0,1 1)", "POLYGON((0 0,0 1,1 1,1 0,0 0))"),
    ],
)
def test_mixed_type_overlays(fn, a, b, expected):
    df = daft.from_pydict({"a": [a], "b": [b]})
    out = df.select(st_astext(st_normalize(fn(_g("a"), _g("b")))).alias("w")).to_pydict()
    want = daft.from_pydict({"e": [expected]}).select(st_astext(st_normalize(_g("e"))).alias("w")).to_pydict()
    assert out["w"] == want["w"]


# ── Degenerate convex hull / envelope ────────────────────────────────────────


@pytest.mark.parametrize(
    "wkt,hull,hull_type,env,env_type",
    [
        ("POINT(1 2)", "POINT(1 2)", "Point", "POINT(1 2)", "Point"),
        ("LINESTRING(0 0,1 1,2 2)", "LINESTRING(0 0,2 2)", "LineString", "POLYGON((0 0,2 0,2 2,0 2,0 0))", "Polygon"),
        ("LINESTRING(0 0,5 0)", "LINESTRING(0 0,5 0)", "LineString", "LINESTRING(0 0,5 0)", "LineString"),
        ("MULTIPOINT((3 3),(3 3))", "POINT(3 3)", "Point", "POINT(3 3)", "Point"),
    ],
)
def test_degenerate_hull_and_envelope(wkt, hull, hull_type, env, env_type):
    df = daft.from_pydict({"g": [wkt]})
    out = df.select(
        st_geometrytype(st_convexhull(_g())).alias("ht"),
        st_geometrytype(st_envelope(_g())).alias("et"),
        st_equals(st_convexhull(_g()), st_geomfromtext(lit(hull))).alias("h"),
        st_equals(st_envelope(_g()), st_geomfromtext(lit(env))).alias("e"),
    ).to_pydict()
    assert out["ht"] == [hull_type]
    assert out["et"] == [env_type]
    assert out["h"] == [True] and out["e"] == [True]


# ── st_simplify collapse (PostGIS drops collapsed rings) ─────────────────────


def test_simplify_is_monotonic_and_drops_collapsed_polygons():
    l_shape = "POLYGON((0 0,10 0,10 1,1 1,1 10,0 10,0 0))"
    tolerances = [0.0, 0.5, 2.0, 5.0, 20.0]
    df = daft.from_pydict({"g": [l_shape] * len(tolerances)})
    counts = []
    for tol in tolerances:
        out = df.limit(1).select(st_astext(st_simplify(_g(), tol)).alias("w")).to_pydict()["w"][0]
        counts.append(0 if out is None else out.count(",") + 1)
    assert counts == sorted(counts, reverse=True), counts
    assert counts[-1] == 0  # whole polygon collapsed -> null, as PostGIS


def test_simplify_multipolygon_drops_collapsed_part():
    mp = "MULTIPOLYGON(((0 0,100 0,100 100,0 100,0 0)),((200 0,201 0,201 1,200 1,200 0)))"
    df = daft.from_pydict({"g": [mp]})
    out = df.select(list_count(st_dump(st_simplify(_g(), 5.0))).alias("n")).to_pydict()
    assert out["n"] == [1]


# ── Small output fixes ───────────────────────────────────────────────────────


def test_normalize_orients_linestrings():
    df = daft.from_pydict({"a": ["LINESTRING(1 1,0 0)"], "b": ["LINESTRING(0 0,1 1)"]})
    out = df.select(
        st_astext(st_normalize(_g("a"))).alias("a"), st_astext(st_normalize(_g("b"))).alias("b")
    ).to_pydict()
    assert out["a"] == out["b"] == ["LINESTRING(0 0,1 1)"]


def test_point_empty_round_trips():
    df = daft.from_pydict({"g": ["POINT EMPTY"]})
    out = df.select(st_astext(_g()).alias("w"), st_x(_g()).alias("x")).to_pydict()
    assert out["w"] == ["POINT EMPTY"]
    assert out["x"] == [None]


def test_st_xy_non_point_is_null():
    df = daft.from_pydict({"g": [SQUARE, "POINT(3 4)"]})
    out = df.select(st_x(_g()).alias("x"), st_y(_g()).alias("y")).to_pydict()
    assert out["x"] == [None, 3.0]
    assert out["y"] == [None, 4.0]


# ── Review follow-ups ────────────────────────────────────────────────────────


def test_overlay_dissolves_overlapping_collection_polygons():
    gc = "GEOMETRYCOLLECTION(POLYGON((0 0,2 0,2 2,0 2,0 0)),POLYGON((1 1,3 1,3 3,1 3,1 1)))"
    big = "POLYGON((-10 -10,10 -10,10 10,-10 10,-10 -10))"
    df = daft.from_pydict({"gc": [gc], "big": [big]})
    out = df.select(
        st_area(st_intersection(_g("gc"), _g("big"))).alias("i"),
        st_area(st_difference(_g("big"), _g("gc"))).alias("d"),
        st_isvalid(st_union(_g("gc"), st_geomfromtext(lit("POINT(50 50)")))).alias("u_valid"),
    ).to_pydict()
    assert out["i"] == [pytest.approx(7.0)]
    assert out["d"] == [pytest.approx(393.0)]
    assert out["u_valid"] == [True]


@pytest.mark.parametrize(
    "fn,a,b,expected",
    [
        # A line touching a polygon at one point intersects it in that point.
        (st_intersection, "LINESTRING(-1 3,0 2,1 3)", "POLYGON((0 0,2 0,2 2,0 2,0 0))", "POINT(0 2)"),
        # Identical lines are not duplicated.
        (st_union, "LINESTRING(0 0,1 1)", "LINESTRING(1 1,0 0)", "LINESTRING(0 0,1 1)"),
        # Lower-dimensional parts inside the areal intersection are absorbed.
        (
            st_intersection,
            "GEOMETRYCOLLECTION(POLYGON((0 0,4 0,4 4,0 4,0 0)),POINT(1 1))",
            "POLYGON((0 0,2 0,2 2,0 2,0 0))",
            "POLYGON((0 0,2 0,2 2,0 2,0 0))",
        ),
    ],
)
def test_overlay_review_cases(fn, a, b, expected):
    df = daft.from_pydict({"a": [a], "b": [b]})
    out = df.select(st_astext(st_normalize(fn(_g("a"), _g("b")))).alias("w")).to_pydict()
    want = daft.from_pydict({"e": [expected]}).select(st_astext(st_normalize(_g("e"))).alias("w")).to_pydict()
    assert out["w"] == want["w"]


def test_empty_point_inputs_stay_empty():
    df = daft.from_pydict({"g": ["POINT EMPTY"]})
    out = df.select(
        st_astext(st_buffer(_g(), 1.0)).alias("buf"),
        st_astext(st_envelope(_g())).alias("env"),
        st_astext(st_convexhull(_g())).alias("hull"),
        st_isvalid(_g()).alias("valid"),
        st_astext(st_union(_g(), _g())).alias("u"),
    ).to_pydict()
    assert out["buf"] == ["MULTIPOLYGON EMPTY"]
    assert out["env"] == ["POINT EMPTY"]
    assert out["hull"] == ["POINT EMPTY"]
    assert out["valid"] == [True]
    assert out["u"] == ["GEOMETRYCOLLECTION EMPTY"]


def test_makevalid_keeps_valid_polygon_with_collinear_vertices():
    # Real parcels often carry collinear vertices; valid input must come back untouched.
    wkt = "POLYGON((-77.03 38.9,-77.02995 38.9,-77.0299 38.9,-77.0299 38.9001,-77.03 38.9001,-77.03 38.9))"
    from daft.functions import st_makevalid

    df = daft.from_pydict({"g": [wkt]})
    out = df.select(st_astext(st_makevalid(_g())).alias("w"), st_astext(_g()).alias("in")).to_pydict()
    assert out["w"] == out["in"]


def test_makevalid_t_junction_keeps_even_odd_area():
    # A vertex lying inside another edge (a T-junction), exactly or a few ulps off it
    # after rounding. geo left the edge unsplit, dropped the edges meeting at the vertex
    # from its triangulation, and got the area wrong. Areas from GEOS (even-odd).
    from daft.functions import st_makevalid

    cell = (6e-5 / 8) ** 2
    cases = {
        # (5 3) lies on the edge (7 1)-(4 4).
        "POLYGON((7 1,4 4,5 6,5 3,8 5,4 2,7 1))": 26 / 7,
        # POLYGON((3 4,1 0,1 7,3 3,2 2,3 4)) in 7.5e-6 degree cells at -77.03: (2 2) sits
        # a few ulps off the edge (3 4)-(1 0), and the closing edge overlaps it.
        "POLYGON((-77.0299775 -77.02997,-77.0299925 -77.03,-77.0299925 -77.0299475,"
        "-77.0299775 -77.0299775,-77.029985 -77.029985,-77.0299775 -77.02997))": 6.5 * cell,
    }
    df = daft.from_pydict({"g": list(cases)})
    out = df.select(st_area(st_makevalid(_g())).alias("a"), st_isvalid(st_makevalid(_g())).alias("v")).to_pydict()
    assert out["a"] == pytest.approx(list(cases.values()), rel=1e-6)
    assert out["v"] == [True, True]


# ── Line noding (results checked against GEOS 3.14 via shapely) ──────────────


@pytest.mark.parametrize(
    "fn,a,b,expected",
    [
        (st_difference, "LINESTRING(0 0,10 0)", "LINESTRING(2 0,4 0)", "MULTILINESTRING((0 0,2 0),(4 0,10 0))"),
        (st_difference, "LINESTRING(0 0,10 0)", "LINESTRING(5 -1,5 1)", "MULTILINESTRING((0 0,5 0),(5 0,10 0))"),
        (st_intersection, "LINESTRING(0 0,5 0,10 0)", "LINESTRING(1 0,9 0)", "MULTILINESTRING((1 0,5 0),(5 0,9 0))"),
        (st_union, "LINESTRING(0 0,10 0)", "LINESTRING(5 0,15 0)", "MULTILINESTRING((0 0,5 0),(5 0,10 0),(10 0,15 0))"),
        (st_symdifference, "LINESTRING(0 0,10 0)", "LINESTRING(0 0,10 0)", "GEOMETRYCOLLECTION EMPTY"),
        (st_symdifference, "LINESTRING(0 0,10 0)", "LINESTRING(5 0,15 0)", "MULTILINESTRING((0 0,5 0),(10 0,15 0))"),
        (
            st_difference,
            "LINESTRING(-1 3,0 2,1 3)",
            "POLYGON((0 0,2 0,2 2,0 2,0 0))",
            "MULTILINESTRING((-1 3,0 2),(0 2,1 3))",
        ),
    ],
)
def test_line_overlays_are_noded_like_geos(fn, a, b, expected):
    df = daft.from_pydict({"a": [a], "b": [b]})
    out = df.select(st_astext(st_normalize(fn(_g("a"), _g("b")))).alias("w")).to_pydict()
    want = daft.from_pydict({"e": [expected]}).select(st_astext(st_normalize(_g("e"))).alias("w")).to_pydict()
    assert out["w"] == want["w"]


def test_simplify_breaks_ties_like_postgis():
    """Equally-far vertices: the first one splits (PostGIS/GEOS), not the last (geo)."""
    df = daft.from_pydict({"g": ["POLYGON((0 0,10 0,10 1,1 1,1 10,0 10,0 0))"]})
    out = df.select(st_astext(st_simplify(_g(), 2.0)).alias("w")).to_pydict()
    assert out["w"] == ["POLYGON((0 0,10 1,1 1,0 10,0 0))"]


def test_overlay_with_literal_region_over_many_rows():
    """A literal operand is prepared once and reused; every row must still be right."""
    parcels = [_rect(6e-5, 6e-5, -77.03 + i * 1e-4, 38.90) for i in range(50)]
    region = _rect(25 * 1e-4, 1.0, -77.03 - 1e-5, 38.0)  # covers the first 25 parcels
    df = daft.from_pydict({"p": parcels})
    out = df.select(
        st_area(st_intersection(_g("p"), st_geomfromtext(lit(region)))).alias("a"),
        st_equals(st_intersection(st_geomfromtext(lit(region)), _g("p")), _g("p")).alias("inside"),
    ).to_pydict()
    assert out["a"][:25] == [pytest.approx(3.6e-9, rel=1e-6)] * 25
    assert out["a"][25:] == [0.0] * 25
    assert out["inside"] == [True] * 25 + [False] * 25


# ── Touching polygons intersect in their shared boundary (as GEOS 3.14) ──────


@pytest.mark.parametrize(
    "a,b,expected",
    [
        pytest.param(SQUARE, "POLYGON((1 0,2 0,2 1,1 1,1 0))", "LINESTRING(1 0,1 1)", id="shared_edge"),
        pytest.param(SQUARE, "POLYGON((1 0.5,2 0.5,2 2,1 2,1 0.5))", "LINESTRING(1 0.5,1 1)", id="partial_edge"),
        pytest.param(SQUARE, "POLYGON((1 1,2 1,2 2,1 2,1 1))", "POINT(1 1)", id="corner"),
        pytest.param(
            SQUARE,
            "MULTIPOLYGON(((1 0,2 0,2 1,1 1,1 0)),((-1 1,0 1,0 2,-1 2,-1 1)))",
            "GEOMETRYCOLLECTION(LINESTRING(1 0,1 1),POINT(0 1))",
            id="edge_and_corner",
        ),
        pytest.param(
            "MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)),((3 3,4 3,4 4,3 4,3 3)))",
            "POLYGON((0.5 0,2 0,2 1,3 3,0.5 1,0.5 0))",
            "GEOMETRYCOLLECTION(POLYGON((0.5 0,0.5 1,1 1,1 0,0.5 0)),POINT(3 3))",
            id="overlap_and_corner",
        ),
        pytest.param(
            "POLYGON((0 0,4 0,4 4,0 4,0 0),(1 1,3 1,3 3,1 3,1 1))",
            "POLYGON((1 1,3 1,3 3,1 3,1 1))",
            "MULTILINESTRING((1 1,3 1),(3 1,3 3),(3 3,1 3),(1 3,1 1))",
            id="fills_hole",
        ),
        pytest.param(
            _rect(6e-5, 6e-5),
            _rect(6e-5, 6e-5, -77.02994, 38.90),
            f"LINESTRING(-77.02994 38.9,-77.02994 {38.90 + 6e-5})",
            id="adjacent_lonlat_parcels",
        ),
    ],
)
def test_touching_polygons_intersect_in_shared_boundary(a, b, expected):
    df = daft.from_pydict({"a": [a], "b": [b], "e": [expected]})
    out = df.select(
        st_equals(st_intersection(_g("a"), _g("b")), _g("e")).alias("eq"),
        st_geometrytype(st_intersection(_g("a"), _g("b"))).alias("t"),
        st_geometrytype(_g("e")).alias("te"),
    ).to_pydict()
    assert out["t"] == out["te"]
    assert out["eq"] == [True]


def test_overlapping_polygons_have_no_spurious_boundary_parts():
    """Boundary crossings of overlapping polygons lie on the areal result and are absorbed."""
    df = daft.from_pydict({"a": [_rect(1e-4, 1e-4)], "b": [_rect(1e-4, 1e-4, -77.02995, 38.90003)]})
    out = df.select(st_geometrytype(st_intersection(_g("a"), _g("b"))).alias("t")).to_pydict()
    assert out["t"] == ["Polygon"]
