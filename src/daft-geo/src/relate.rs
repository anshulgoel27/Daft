use geo::coordinate_position::{CoordPos, CoordinatePosition};
use geo::relate::Relate;
use geo::{Contains, CoordsIter, Geometry, Intersects};

/// The DE-9IM spatial predicate to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelatePred {
    Intersects,
    Contains,
    Within,
    Touches,
    Crosses,
    Overlaps,
    Disjoint,
    Equals,
    Covers,
    CoveredBy,
}

/// Point fast path: when one operand is a single `Point`, most DE-9IM predicates
/// reduce to a point-in-geometry test (`coordinate_position` / ray-cast) that
/// avoids building the full topology graph `Relate` requires. Returns `None`
/// when no fast path applies (non-point pairs, or predicates whose point
/// semantics aren't reducible), in which case the caller falls back to relate.
///
/// Semantics per DE-9IM, verified against the relate matrix by
/// `point_fast_path_agrees_with_relate_matrix`:
/// - Contains(a, pt): pt in interior(a) — boundary excluded (`geo::Contains`).
/// - Within(pt, b): mirror of Contains.
/// - Intersects / Disjoint: pt not in exterior / pt in exterior.
/// - Covers(a, pt) / CoveredBy(pt, b): boundary included.
/// - Touches: pt on boundary only (a point has no boundary, so point-point
///   touches is always false — `coordinate_position` yields `Inside` there).
fn point_fast_path(a: &Geometry, b: &Geometry, pred: RelatePred) -> Option<bool> {
    match pred {
        RelatePred::Contains => match b {
            Geometry::Point(p) => Some(a.contains(p)),
            _ => None,
        },
        RelatePred::Within => match a {
            Geometry::Point(p) => Some(b.contains(p)),
            _ => None,
        },
        RelatePred::Intersects | RelatePred::Disjoint => {
            let hit = match (a, b) {
                (_, Geometry::Point(p)) => a.intersects(p),
                (Geometry::Point(p), _) => b.intersects(p),
                _ => return None,
            };
            Some(matches!(pred, RelatePred::Intersects) == hit)
        }
        RelatePred::Covers => match b {
            Geometry::Point(p) => Some(a.coordinate_position(&p.0) != CoordPos::Outside),
            _ => None,
        },
        RelatePred::CoveredBy => match a {
            Geometry::Point(p) => Some(b.coordinate_position(&p.0) != CoordPos::Outside),
            _ => None,
        },
        RelatePred::Touches => match (a, b) {
            (other, Geometry::Point(p)) | (Geometry::Point(p), other) => {
                Some(other.coordinate_position(&p.0) == CoordPos::OnBoundary)
            }
            _ => None,
        },
        // Crosses/Overlaps/Equals with a point operand are rare in joins and
        // have subtler dimension-dependent semantics — leave them to relate.
        RelatePred::Crosses | RelatePred::Overlaps | RelatePred::Equals => None,
    }
}

/// Vertex fast path for `Intersects` / `Disjoint` between two non-point geometries.
///
/// A vertex of a non-empty geometry is a point of that geometry, so if one
/// vertex of `a` is not in the exterior of `b` (or vice versa) the two
/// intersect, which proves `Intersects` without the full topology graph that
/// `relate` builds. That one point-in-geometry test is O(vertices), against
/// relate's graph construction and noding of BOTH operands. In a spatial join
/// (an isochrone or buffer against the many small polygons it covers) most
/// true candidate pairs have one operand inside the other, so they stop at the
/// first test. A miss proves nothing (the boundaries may still cross), so it
/// returns `None` and the caller falls back to relate. Hence this never changes
/// a result, it only skips work. Empty operands have no vertex and also fall
/// back.
fn vertex_fast_path(a: &Geometry, b: &Geometry, pred: RelatePred) -> Option<bool> {
    if !matches!(pred, RelatePred::Intersects | RelatePred::Disjoint) {
        return None;
    }
    let hit = |of: &Geometry, inside: &Geometry| {
        of.coords_iter()
            .next()
            .is_some_and(|c| inside.coordinate_position(&c) != CoordPos::Outside)
    };
    if hit(a, b) || hit(b, a) {
        return Some(matches!(pred, RelatePred::Intersects));
    }
    None
}

/// Evaluate a DE-9IM predicate between two geometries. Correct for all geometry-type pairs.
///
/// Wrapped in `catch_unwind` so that a panic on degenerate input (e.g. a self-intersecting
/// ring triggering a bug in the `geo` relate implementation) does not unwind the compute
/// thread. A panic yields `None` (a null result), never a guessed `false` — otherwise
/// `st_intersects` and `st_disjoint` could both be false for the same pair.
pub fn relate_pred(a: &Geometry, b: &Geometry, pred: RelatePred) -> Option<bool> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Some(fast) = point_fast_path(a, b, pred) {
            return fast;
        }
        if let Some(fast) = vertex_fast_path(a, b, pred) {
            return fast;
        }
        let m = a.relate(b); // geo 0.33: Geometry: Relate -> IntersectionMatrix
        match pred {
            RelatePred::Intersects => m.is_intersects(),
            RelatePred::Contains => m.is_contains(),
            RelatePred::Within => m.is_within(),
            RelatePred::Touches => m.is_touches(),
            RelatePred::Crosses => m.is_crosses(),
            RelatePred::Overlaps => m.is_overlaps(),
            RelatePred::Disjoint => m.is_disjoint(),
            RelatePred::Equals => m.is_equal_topo(),
            RelatePred::Covers => m.is_covers(),
            RelatePred::CoveredBy => m.is_coveredby(),
        }
    }))
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::relate::Relate;
    use geo::{Geometry, MultiPolygon, Point, Polygon, LineString, Coord};

    fn rp(a: &Geometry, b: &Geometry, pred: RelatePred) -> bool {
        relate_pred(a, b, pred).expect("relate should not panic on these inputs")
    }

    fn square() -> Geometry {
        // unit square (0,0)-(2,2)
        let ring = LineString(vec![
            Coord { x: 0.0, y: 0.0 }, Coord { x: 2.0, y: 0.0 },
            Coord { x: 2.0, y: 2.0 }, Coord { x: 0.0, y: 2.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        Geometry::Polygon(Polygon::new(ring, vec![]))
    }

    // ── Point fast-path tests ─────────────────────────────────────────────
    //
    // `point_fast_path` must agree with the full DE-9IM matrix on every input
    // it accepts (returns `Some`). The baseline below IS the spec: the same
    // matrix dispatch `relate_pred` uses, evaluated without any fast path.

    fn relate_baseline(a: &Geometry, b: &Geometry, pred: RelatePred) -> bool {
        let m = a.relate(b);
        match pred {
            RelatePred::Intersects => m.is_intersects(),
            RelatePred::Contains => m.is_contains(),
            RelatePred::Within => m.is_within(),
            RelatePred::Touches => m.is_touches(),
            RelatePred::Crosses => m.is_crosses(),
            RelatePred::Overlaps => m.is_overlaps(),
            RelatePred::Disjoint => m.is_disjoint(),
            RelatePred::Equals => m.is_equal_topo(),
            RelatePred::Covers => m.is_covers(),
            RelatePred::CoveredBy => m.is_coveredby(),
        }
    }

    const ALL_PREDS: [RelatePred; 10] = [
        RelatePred::Intersects, RelatePred::Contains, RelatePred::Within,
        RelatePred::Touches, RelatePred::Crosses, RelatePred::Overlaps,
        RelatePred::Disjoint, RelatePred::Equals, RelatePred::Covers,
        RelatePred::CoveredBy,
    ];

    /// Square (0,0)-(4,4) with a hole (1,1)-(3,3): exercises interior,
    /// exterior ring boundary, hole boundary, and inside-the-hole cases.
    fn holed_square() -> Geometry {
        let outer = LineString(vec![
            Coord { x: 0.0, y: 0.0 }, Coord { x: 4.0, y: 0.0 },
            Coord { x: 4.0, y: 4.0 }, Coord { x: 0.0, y: 4.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        let hole = LineString(vec![
            Coord { x: 1.0, y: 1.0 }, Coord { x: 3.0, y: 1.0 },
            Coord { x: 3.0, y: 3.0 }, Coord { x: 1.0, y: 3.0 },
            Coord { x: 1.0, y: 1.0 },
        ]);
        Geometry::Polygon(Polygon::new(outer, vec![hole]))
    }

    fn containers() -> Vec<Geometry> {
        vec![
            square(),
            holed_square(),
            Geometry::MultiPolygon(MultiPolygon(vec![
                Polygon::new(
                    LineString(vec![
                        Coord { x: 0.0, y: 0.0 }, Coord { x: 2.0, y: 0.0 },
                        Coord { x: 2.0, y: 2.0 }, Coord { x: 0.0, y: 2.0 },
                        Coord { x: 0.0, y: 0.0 },
                    ]),
                    vec![],
                ),
                Polygon::new(
                    LineString(vec![
                        Coord { x: 10.0, y: 10.0 }, Coord { x: 12.0, y: 10.0 },
                        Coord { x: 12.0, y: 12.0 }, Coord { x: 10.0, y: 12.0 },
                        Coord { x: 10.0, y: 10.0 },
                    ]),
                    vec![],
                ),
            ])),
            Geometry::LineString(LineString(vec![
                Coord { x: 0.0, y: 0.0 }, Coord { x: 2.0, y: 2.0 },
                Coord { x: 4.0, y: 0.0 },
            ])),
            Geometry::Point(Point::new(1.0, 1.0)),
        ]
    }

    fn probe_points() -> Vec<Geometry> {
        [
            (0.5, 0.5),   // interior (in-ring for holed square)
            (2.0, 2.0),   // holed-square interior-band / square corner-ish
            (0.0, 1.0),   // exterior-ring boundary edge
            (0.0, 0.0),   // exterior-ring corner
            (1.0, 2.0),   // hole boundary (holed square) / boundary (square)
            (2.0, 2.0),   // dedupe-safe repeat
            (1.5, 1.5),   // inside the hole (exterior for holed square)
            (1.0, 1.0),   // hole corner / point-equal case
            (100.0, 100.0), // far outside
            (11.0, 11.0), // inside second multipolygon part
        ]
        .iter()
        .map(|&(x, y)| Geometry::Point(Point::new(x, y)))
        .collect()
    }

    /// Every (container, point, pred) and (point, container, pred) combination
    /// the fast path accepts must equal the DE-9IM baseline.
    #[test]
    fn point_fast_path_agrees_with_relate_matrix() {
        for container in containers() {
            for pt in probe_points() {
                for pred in ALL_PREDS {
                    for (a, b) in [(&container, &pt), (&pt, &container)] {
                        if let Some(fast) = point_fast_path(a, b, pred) {
                            let expected = relate_baseline(a, b, pred);
                            assert_eq!(
                                fast, expected,
                                "fast path disagrees with relate: {pred:?} a={a:?} b={b:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The workload case must actually take the fast path — not silently
    /// decline and fall back to relate.
    #[test]
    fn point_fast_path_accepts_point_in_polygon_predicates() {
        let poly = square();
        let pt = Geometry::Point(Point::new(1.0, 1.0));
        assert!(point_fast_path(&poly, &pt, RelatePred::Contains).is_some());
        assert!(point_fast_path(&pt, &poly, RelatePred::Within).is_some());
        assert!(point_fast_path(&poly, &pt, RelatePred::Intersects).is_some());
        assert!(point_fast_path(&pt, &poly, RelatePred::Intersects).is_some());
        assert!(point_fast_path(&poly, &pt, RelatePred::Disjoint).is_some());
        assert!(point_fast_path(&poly, &pt, RelatePred::Covers).is_some());
        assert!(point_fast_path(&pt, &poly, RelatePred::CoveredBy).is_some());
    }

    /// Non-point pairs must decline so relate keeps handling them.
    #[test]
    fn point_fast_path_declines_non_point_pairs() {
        let a = square();
        let b = holed_square();
        for pred in ALL_PREDS {
            assert!(
                point_fast_path(&a, &b, pred).is_none(),
                "polygon-polygon must not take the point fast path ({pred:?})"
            );
        }
    }

    /// Boundary semantics through the public entry point: contains is
    /// interior-only, covers includes the boundary.
    #[test]
    fn relate_pred_boundary_point_semantics_unchanged() {
        let poly = square();
        let boundary = Geometry::Point(Point::new(0.0, 1.0));
        let inside = Geometry::Point(Point::new(1.0, 1.0));
        assert!(!rp(&poly, &boundary, RelatePred::Contains));
        assert!(rp(&poly, &boundary, RelatePred::Covers));
        assert!(rp(&poly, &boundary, RelatePred::Intersects));
        assert!(!rp(&poly, &boundary, RelatePred::Disjoint));
        assert!(rp(&poly, &inside, RelatePred::Contains));
        assert!(rp(&inside, &poly, RelatePred::Within));
        assert!(!rp(&boundary, &poly, RelatePred::Within));
        assert!(rp(&boundary, &poly, RelatePred::CoveredBy));
    }

    #[test]
    fn test_contains_and_within_are_symmetric() {
        let poly = square();
        let inside = Geometry::Point(Point::new(1.0, 1.0));
        // previously st_contains returned false for Polygon/Point only via hand-coding;
        // relate must handle it and within is the mirror.
        assert!(rp(&poly, &inside, RelatePred::Contains));
        assert!(rp(&inside, &poly, RelatePred::Within));
        assert!(!rp(&inside, &poly, RelatePred::Contains));
    }

    #[test]
    fn test_disjoint_is_not_intersects() {
        let poly = square();
        let far = Geometry::Point(Point::new(100.0, 100.0));
        assert!(rp(&poly, &far, RelatePred::Disjoint));
        assert!(!rp(&poly, &far, RelatePred::Intersects));
    }

    #[test]
    fn test_linestring_polygon_intersects_now_handled() {
        // a type pair the old hand-coded st_intersects fell through to false on
        let poly = square();
        let line = Geometry::LineString(LineString(vec![
            Coord { x: -1.0, y: 1.0 }, Coord { x: 3.0, y: 1.0 },
        ]));
        assert!(rp(&poly, &line, RelatePred::Intersects));
    }

    #[test]
    fn test_covers_and_covered_by() {
        let poly = square(); // (0,0)-(2,2)
        // Boundary point: contained-by-covers but NOT contains (interior only).
        let boundary = Geometry::Point(Point::new(0.0, 1.0));
        assert!(rp(&poly, &boundary, RelatePred::Covers));
        assert!(rp(&boundary, &poly, RelatePred::CoveredBy));
        assert!(!rp(&poly, &boundary, RelatePred::Contains)); // interior-only
        // A far point is neither covered nor covering.
        let far = Geometry::Point(Point::new(100.0, 100.0));
        assert!(!rp(&poly, &far, RelatePred::Covers));
    }

    // ── Vertex fast-path tests ────────────────────────────────────────────

    /// A closed polygon ring through `pts`.
    fn ring(pts: &[(f64, f64)]) -> Geometry {
        let mut c: Vec<Coord> = pts.iter().map(|&(x, y)| Coord { x, y }).collect();
        c.push(c[0]);
        Geometry::Polygon(Polygon::new(LineString(c), vec![]))
    }

    /// A regular n-gon of radius r around (cx, cy): a buffer / isochrone stand-in.
    fn ngon(cx: f64, cy: f64, r: f64, n: usize) -> Geometry {
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / n as f64;
                (cx + r * t.cos(), cy + r * t.sin())
            })
            .collect();
        ring(&pts)
    }

    #[test]
    fn vertex_fast_path_agrees_with_relate_matrix() {
        // Every relation the join meets: inside, containing, crossing, sharing an
        // edge, touching at a corner, disjoint inside the bbox, inside a hole,
        // around a hole, multi-part, empty and degenerate (zero-area) operands.
        let big = ngon(0.0, 0.0, 10.0, 64);
        let holed = holed_square();
        let mut geoms = vec![
            big.clone(),
            ngon(0.0, 0.0, 1.0, 16),                                    // inside big
            ngon(10.0, 0.0, 1.0, 16),                                   // crosses big's boundary
            ring(&[(9.0, 9.0), (10.0, 9.0), (10.0, 10.0), (9.0, 10.0)]), // in big's bbox corner, outside big
            square(),
            ring(&[(2.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0)]),     // shares an edge with square
            ring(&[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]),     // touches square at a corner
            ring(&[(-1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-1.0, 3.0)]), // contains square, no shared vertex
            ring(&[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]),                 // zero-area polygon
            Geometry::Polygon(Polygon::new(LineString(vec![]), vec![])), // empty
            Geometry::MultiPolygon(MultiPolygon(vec![
                Polygon::new(LineString(vec![Coord { x: 30.0, y: 30.0 }, Coord { x: 31.0, y: 30.0 }, Coord { x: 31.0, y: 31.0 }, Coord { x: 30.0, y: 30.0 }]), vec![]),
                Polygon::new(LineString(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 0.5, y: 0.0 }, Coord { x: 0.5, y: 0.5 }, Coord { x: 0.0, y: 0.0 }]), vec![]),
            ])),
            Geometry::LineString(LineString(vec![Coord { x: -20.0, y: 0.5 }, Coord { x: 20.0, y: 0.5 }])), // crosses everything near y=0.5
        ];
        geoms.push(holed);
        geoms.extend(containers());
        for a in &geoms {
            for b in &geoms {
                for pred in [RelatePred::Intersects, RelatePred::Disjoint] {
                    if let Some(fast) = vertex_fast_path(a, b, pred) {
                        assert_eq!(fast, relate_baseline(a, b, pred), "{pred:?} disagrees for\n  a = {a:?}\n  b = {b:?}");
                    }
                    assert_eq!(rp(a, b, pred), relate_baseline(a, b, pred), "{pred:?} for\n  a = {a:?}\n  b = {b:?}");
                }
            }
        }
    }

    #[test]
    fn vertex_fast_path_only_answers_intersects_and_disjoint() {
        let big = ngon(0.0, 0.0, 10.0, 64);
        let small = ngon(0.0, 0.0, 1.0, 16);
        assert_eq!(vertex_fast_path(&big, &small, RelatePred::Intersects), Some(true));
        assert_eq!(vertex_fast_path(&small, &big, RelatePred::Disjoint), Some(false));
        for pred in [RelatePred::Contains, RelatePred::Within, RelatePred::Covers, RelatePred::Touches] {
            assert_eq!(vertex_fast_path(&big, &small, pred), None);
        }
        // A pair whose vertices are all outside the other proves nothing: defer to relate.
        let corner = ring(&[(9.0, 9.0), (10.0, 9.0), (10.0, 10.0), (9.0, 10.0)]);
        assert_eq!(vertex_fast_path(&big, &corner, RelatePred::Intersects), None);
    }

    /// Timing of the case that dominates the adjacency join: a detailed isochrone
    /// against the small polygons it covers. `cargo test -p daft-geo --release
    /// vertex_fast_path_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn vertex_fast_path_bench() {
        let iso = ngon(0.0, 0.0, 10.0, 4000);
        let das: Vec<Geometry> = (0..400)
            .map(|i| ngon(-7.0 + 0.035 * i as f64, 0.0, 0.03, 200))
            .collect();
        let edge: Vec<Geometry> = (0..400)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 400.0;
                ngon(10.0 * t.cos(), 10.0 * t.sin(), 0.03, 200)
            })
            .collect();
        for (name, set) in [("inside", &das), ("crossing boundary", &edge)] {
            let t0 = std::time::Instant::now();
            let n_relate = set.iter().filter(|d| relate_baseline(&iso, d, RelatePred::Intersects)).count();
            let t_relate = t0.elapsed();
            let t1 = std::time::Instant::now();
            let n_fast = set.iter().filter(|d| rp(&iso, d, RelatePred::Intersects)).count();
            let t_fast = t1.elapsed();
            assert_eq!(n_relate, n_fast);
            println!("{name}: relate {t_relate:?}, relate_pred {t_fast:?} ({:.1}x) for {} pairs", t_relate.as_secs_f64() / t_fast.as_secs_f64(), set.len());
        }
    }
}
