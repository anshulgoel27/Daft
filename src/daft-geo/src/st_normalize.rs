use std::cmp::Ordering;

use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::{
    Coord, Geometry, GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon,
    Point, Polygon, Winding,
    orient::{Direction, Orient},
};
use serde::{Deserialize, Serialize};

use crate::utils::{unary_geom_to_geom, validate_geometry_field};

/// Compare two coordinates lexicographically by (x, y).
///
/// NaN components compare as equal rather than panicking; this only affects the
/// deterministic ordering used for canonicalization, not the geometry's validity.
fn coord_cmp(a: &Coord<f64>, b: &Coord<f64>) -> std::cmp::Ordering {
    a.x.partial_cmp(&b.x)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
}

/// Compare two coordinate sequences lexicographically; shorter sequence sorts first on a
/// common-prefix tie. Used to order rings, multi-part members, and holes deterministically.
fn coords_cmp(a: &[Coord<f64>], b: &[Coord<f64>]) -> std::cmp::Ordering {
    a.iter()
        .zip(b.iter())
        .map(|(ca, cb)| coord_cmp(ca, cb))
        .find(|o| *o != std::cmp::Ordering::Equal)
        .unwrap_or_else(|| a.len().cmp(&b.len()))
}

/// Build the open (no duplicate closing point) coordinate sequence of `coords[..open_len]`
/// rotated to start at `start`.
fn rotated_open_seq(coords: &[Coord<f64>], open_len: usize, start: usize) -> Vec<Coord<f64>> {
    let mut seq = Vec::with_capacity(open_len);
    seq.extend_from_slice(&coords[start..open_len]);
    seq.extend_from_slice(&coords[..start]);
    seq
}

/// Rotate a closed ring so it starts at the rotation whose full coordinate sequence is
/// lexicographically smallest, preserving winding direction. Ties on the minimum vertex
/// (e.g. a ring that revisits the same point) are broken by comparing the complete
/// rotated sequence, not just the first coordinate, so the result stays rotation-invariant
/// even when the minimum vertex repeats. No-op for open or degenerate (fewer than 3 point)
/// rings.
fn rotate_ring_to_min_vertex(ring: &LineString<f64>) -> LineString<f64> {
    let coords = &ring.0;
    let open_len = coords.len().saturating_sub(1);
    if open_len < 2 || coords.first() != coords.last() {
        return ring.clone();
    }
    let min_coord = coords[..open_len]
        .iter()
        .min_by(|a, b| coord_cmp(a, b))
        .expect("open_len >= 2");
    let best_start = (0..open_len)
        .filter(|&i| coords[i] == *min_coord)
        .min_by(|&i, &j| {
            coords_cmp(
                &rotated_open_seq(coords, open_len, i),
                &rotated_open_seq(coords, open_len, j),
            )
        })
        .unwrap_or(0);
    if best_start == 0 {
        return ring.clone();
    }
    let mut rotated = rotated_open_seq(coords, open_len, best_start);
    rotated.push(rotated[0]);
    LineString::new(rotated)
}

/// Canonicalize a polygon: orient the exterior ring clockwise and interior rings
/// counter-clockwise, rotate every ring to start at its lexicographically smallest
/// vertex, and sort the interior rings into a deterministic order (hole order is not
/// semantically meaningful).
fn normalize_polygon(poly: &Polygon<f64>) -> Polygon<f64> {
    let oriented = poly.orient(Direction::Reversed);
    let exterior = rotate_ring_to_min_vertex(oriented.exterior());
    let mut interiors: Vec<LineString<f64>> = oriented
        .interiors()
        .iter()
        .map(rotate_ring_to_min_vertex)
        .collect();
    interiors.sort_by(descending(|a: &LineString<f64>, b: &LineString<f64>| {
        curve_cmp(&a.0, &b.0)
    }));
    Polygon::new(exterior, interiors)
}

/// Canonicalize a LineString's direction, as GEOS `normalize` does. An open line is
/// reversed when its reversed coordinate sequence is lexicographically smaller, so
/// `LINESTRING(1 1,0 0)` and `LINESTRING(0 0,1 1)` normalize identically. A closed
/// line is wound clockwise (like a polygon shell) and rotated to its smallest vertex.
fn normalize_linestring(ls: &LineString<f64>) -> LineString<f64> {
    // GEOS `SimpleCurve::normalizeClosed`: rotate to the smallest vertex, then
    // wind clockwise (only when there are at least 4 points).
    let closed = ls.0.len() >= 3 && ls.0.first() == ls.0.last();
    if closed {
        let mut ring = rotate_ring_to_min_vertex(ls);
        if ring.0.len() >= 4 && ring.is_ccw() {
            ring.0.reverse();
        }
        return ring;
    }
    let mut reversed = ls.0.clone();
    reversed.reverse();
    if coords_cmp(&reversed, &ls.0) == std::cmp::Ordering::Less {
        LineString::new(reversed)
    } else {
        ls.clone()
    }
}

/// GEOS's sort rank of a geometry type (`Geometry::getSortIndex`).
fn sort_index(g: &Geometry<f64>) -> u8 {
    match g {
        Geometry::Point(_) => 0,
        Geometry::MultiPoint(_) => 1,
        Geometry::LineString(_) | Geometry::Line(_) => 2,
        Geometry::MultiLineString(_) => 4,
        Geometry::Polygon(_) | Geometry::Rect(_) | Geometry::Triangle(_) => 5,
        Geometry::MultiPolygon(_) => 6,
        Geometry::GeometryCollection(_) => 7,
    }
}

/// GEOS `SimpleCurve::compareToSameClass` (via `Geometry::compareTo` for two
/// curves of one class): point count first, then the coordinates in order.
fn curve_cmp(a: &[Coord<f64>], b: &[Coord<f64>]) -> Ordering {
    // Geometry::compareTo: an empty curve sorts before a non-empty one.
    match (a.is_empty(), b.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }
    a.len().cmp(&b.len()).then_with(|| {
        a.iter()
            .zip(b)
            .map(|(ca, cb)| coord_cmp(ca, cb))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    })
}

/// GEOS `Surface::compareToSameClass`: shell, then hole count, then holes in order.
fn polygon_cmp(a: &Polygon<f64>, b: &Polygon<f64>) -> Ordering {
    match (a.exterior().0.is_empty(), b.exterior().0.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }
    curve_cmp(&a.exterior().0, &b.exterior().0)
        .then_with(|| a.interiors().len().cmp(&b.interiors().len()))
        .then_with(|| {
            a.interiors()
                .iter()
                .zip(b.interiors())
                .map(|(ra, rb)| curve_cmp(&ra.0, &rb.0))
                .find(|o| *o != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        })
}

/// GEOS `Geometry::compare` over member lists: element-wise, then length.
fn members_cmp<T>(a: &[T], b: &[T], cmp: impl Fn(&T, &T) -> Ordering) -> Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| cmp(x, y))
        .find(|o| *o != Ordering::Equal)
        .unwrap_or_else(|| a.len().cmp(&b.len()))
}

/// GEOS `Geometry::compareTo`: type rank, then emptiness, then contents.
fn geos_cmp(a: &Geometry<f64>, b: &Geometry<f64>) -> Ordering {
    use geo::HasDimensions;
    let by_rank = sort_index(a).cmp(&sort_index(b));
    if by_rank != Ordering::Equal {
        return by_rank;
    }
    match (a.is_empty(), b.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }
    match (a, b) {
        (Geometry::Point(p), Geometry::Point(q)) => coord_cmp(&p.0, &q.0),
        (Geometry::LineString(p), Geometry::LineString(q)) => curve_cmp(&p.0, &q.0),
        (Geometry::Polygon(p), Geometry::Polygon(q)) => polygon_cmp(p, q),
        (Geometry::MultiPoint(p), Geometry::MultiPoint(q)) => {
            members_cmp(&p.0, &q.0, |x, y| coord_cmp(&x.0, &y.0))
        }
        (Geometry::MultiLineString(p), Geometry::MultiLineString(q)) => {
            members_cmp(&p.0, &q.0, |x, y| curve_cmp(&x.0, &y.0))
        }
        (Geometry::MultiPolygon(p), Geometry::MultiPolygon(q)) => {
            members_cmp(&p.0, &q.0, polygon_cmp)
        }
        (Geometry::GeometryCollection(p), Geometry::GeometryCollection(q)) => {
            members_cmp(&p.0, &q.0, geos_cmp)
        }
        // Line / Rect / Triangle (not producible from WKB): compare as their
        // WKB-equivalent types.
        _ => {
            let promote = |g: &Geometry<f64>| match g {
                Geometry::Line(l) => Geometry::LineString(LineString::new(vec![l.start, l.end])),
                Geometry::Rect(r) => Geometry::Polygon(r.to_polygon()),
                Geometry::Triangle(t) => Geometry::Polygon(t.to_polygon()),
                other => other.clone(),
            };
            geos_cmp(&promote(a), &promote(b))
        }
    }
}

/// GEOS normalize sorts members, holes, lines and points in *descending*
/// `compareTo` order (`std::sort` with `a.compareTo(b) > 0`).
fn descending<T>(cmp: impl Fn(&T, &T) -> Ordering) -> impl Fn(&T, &T) -> Ordering {
    move |a, b| cmp(b, a)
}

/// Recursively canonicalize a geometry so that spatially-equivalent inputs (differing
/// only in ring orientation, ring starting vertex, or non-semantic part ordering)
/// produce an identical output, matching GEOS `Geometry::normalize` (PostGIS
/// `ST_Normalize`) byte for byte: rings start at their smallest vertex, shells
/// are clockwise and holes counter-clockwise, and members, holes, lines and
/// points are sorted in descending GEOS `compareTo` order. Used upstream of hash
/// generation to avoid false-positive change detections.
fn normalize_geometry(geom: &Geometry<f64>) -> Option<Geometry<f64>> {
    match geom {
        Geometry::Polygon(p) => Some(Geometry::Polygon(normalize_polygon(p))),
        Geometry::MultiPolygon(mp) => {
            let mut polys: Vec<Polygon<f64>> = mp.iter().map(normalize_polygon).collect();
            polys.sort_by(descending(polygon_cmp));
            Some(Geometry::MultiPolygon(MultiPolygon(polys)))
        }
        Geometry::LineString(ls) => Some(Geometry::LineString(normalize_linestring(ls))),
        Geometry::MultiLineString(mls) => {
            let mut lines: Vec<LineString<f64>> = mls.iter().map(normalize_linestring).collect();
            lines.sort_by(descending(|a: &LineString<f64>, b: &LineString<f64>| {
                curve_cmp(&a.0, &b.0)
            }));
            Some(Geometry::MultiLineString(MultiLineString::new(lines)))
        }
        Geometry::MultiPoint(mp) => {
            let mut pts: Vec<Point<f64>> = mp.iter().cloned().collect();
            pts.sort_by(descending(|a: &Point<f64>, b: &Point<f64>| {
                coord_cmp(&a.0, &b.0)
            }));
            Some(Geometry::MultiPoint(MultiPoint(pts)))
        }
        Geometry::GeometryCollection(gc) => {
            let mut members = gc.iter().map(normalize_geometry).collect::<Option<Vec<_>>>()?;
            members.sort_by(descending(geos_cmp));
            Some(Geometry::GeometryCollection(GeometryCollection::new_from(
                members,
            )))
        }
        // Point has no non-semantic ordering to canonicalize; other
        // variants (Line, Rect, Triangle) are not producible from WKB/WKT input.
        other => Some(other.clone()),
    }
}

fn apply_normalize(g: &Geometry) -> Option<Geometry> {
    // Wrapped in catch_unwind for defensive robustness, matching the other geometry
    // transforms in this crate, since sorting/winding can panic on degenerate input.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| normalize_geometry(g)))
        .ok()
        .flatten()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StNormalize;

#[typetag::serde]
impl ScalarUDF for StNormalize {
    fn name(&self) -> &'static str {
        "st_normalize"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        unary_geom_to_geom(inputs.required(0)?, self.name(), apply_normalize)
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom", self.name())?;
        Ok(Field::new(self.name(), DataType::Geometry))
    }

    fn docstring(&self) -> &'static str {
        "Returns the geometry in a canonical, normalized form. Polygon rings are wound \
         consistently (clockwise exterior, counter-clockwise interiors) and rotated to \
         start at their lexicographically smallest vertex; the non-semantic ordering of \
         multi-part geometry and geometry-collection members is sorted deterministically. \
         Geometrically equivalent inputs that differ only in ring orientation, ring \
         starting vertex, or part order produce identical output, making this useful as a \
         pre-step to hashing for change detection."
    }
}

#[must_use]
pub fn st_normalize(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StNormalize, vec![geom]).into()
}

#[cfg(test)]
mod tests {
    use geo::Geometry;
    use wkt::{ToWkt, TryFromWkt};

    use super::normalize_geometry;

    fn norm_wkt(wkt: &str) -> String {
        let geom: Geometry<f64> = Geometry::try_from_wkt_str(wkt).unwrap();
        normalize_geometry(&geom).unwrap().to_wkt().to_string()
    }

    #[test]
    fn polygon_ring_orientation_is_normalized() {
        let cw = "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))";
        let ccw = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))";
        assert_eq!(norm_wkt(cw), norm_wkt(ccw));
    }

    #[test]
    fn polygon_ring_start_vertex_is_normalized() {
        let a = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))";
        let b = "POLYGON((10 10, 0 10, 0 0, 10 0, 10 10))";
        assert_eq!(norm_wkt(a), norm_wkt(b));
    }

    #[test]
    fn polygon_ring_with_repeated_min_vertex_is_rotation_invariant() {
        // The minimum vertex (0 0) appears twice, so a naive "first minimal vertex" rotation
        // picks a different start point depending on which occurrence comes first in the
        // input, producing different sequences for the same ring under rotation.
        let a = "POLYGON((0 0,0 0,0 10,10 10,10 0,0 0))";
        let b = "POLYGON((0 10,10 10,10 0,0 0,0 0,0 10))";
        assert_eq!(norm_wkt(a), norm_wkt(b));
    }

    #[test]
    fn polygon_hole_orientation_and_order_are_normalized() {
        let a = "POLYGON((0 0,0 10,10 10,10 0,0 0),(1 1,1 2,2 2,2 1,1 1),(4 4,4 5,5 5,5 4,4 4))";
        let b = "POLYGON((0 0,10 0,10 10,0 10,0 0),(4 4,5 4,5 5,4 5,4 4),(1 1,2 1,2 2,1 2,1 1))";
        assert_eq!(norm_wkt(a), norm_wkt(b));
    }

    #[test]
    fn multipolygon_part_order_is_normalized() {
        let a = "MULTIPOLYGON(((0 0,0 1,1 1,1 0,0 0)),((10 10,10 11,11 11,11 10,10 10)))";
        let b = "MULTIPOLYGON(((10 10,10 11,11 11,11 10,10 10)),((0 0,0 1,1 1,1 0,0 0)))";
        assert_eq!(norm_wkt(a), norm_wkt(b));
    }

    #[test]
    fn multipolygon_parts_sharing_exterior_but_differing_holes_do_not_tie() {
        // Both parts have the same exterior ring, so an exterior-only comparator would keep
        // input order (a no-op sort) instead of falling through to the holes, and the two
        // permutations below would normalize differently.
        let a = "MULTIPOLYGON(((0 0,0 10,10 10,10 0,0 0),(1 1,1 2,2 2,2 1,1 1)),((0 0,0 10,10 10,10 0,0 0),(4 4,4 5,5 5,5 4,4 4)))";
        let b = "MULTIPOLYGON(((0 0,0 10,10 10,10 0,0 0),(4 4,4 5,5 5,5 4,4 4)),((0 0,0 10,10 10,10 0,0 0),(1 1,1 2,2 2,2 1,1 1)))";
        assert_eq!(norm_wkt(a), norm_wkt(b));
    }

    #[test]
    fn multipoint_order_is_normalized() {
        let a = "MULTIPOINT(3 3, 1 1, 2 2)";
        let b = "MULTIPOINT(1 1, 2 2, 3 3)";
        assert_eq!(norm_wkt(a), norm_wkt(b));
    }

    #[test]
    fn different_geometries_stay_different() {
        let a = "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))";
        let b = "POLYGON((0 0, 0 5, 10 10, 10 0, 0 0))";
        assert_ne!(norm_wkt(a), norm_wkt(b));
    }

    #[test]
    fn normalization_is_idempotent_and_deterministic() {
        let wkt = "MULTIPOLYGON(((10 10,10 11,11 11,11 10,10 10)),((0 0,0 1,1 1,1 0,0 0)))";
        let once = norm_wkt(wkt);
        let twice = norm_wkt(&once);
        assert_eq!(once, twice);
    }

    /// Expected outputs are PostGIS 3.5.4 / GEOS 3.14.1 `ST_AsText(ST_Normalize(g))`.
    #[test]
    fn matches_geos_normalize() {
        for (input, geos) in [
            (
                "MULTIPOLYGON(((0 0,1 0,1 1,0 0)),((5 5,6 5,6 6,5 5)),((0 0,2 0,2 2,0 2,0 0)))",
                "MULTIPOLYGON(((0 0,0 2,2 2,2 0,0 0)),((5 5,6 6,6 5,5 5)),((0 0,1 1,1 0,0 0)))",
            ),
            (
                "POLYGON((0 0,10 0,10 10,0 10,0 0),(1 1,2 1,2 2,1 1),(5 5,6 5,6 6,5 6,5 5),(7 1,8 1,8 2,7 1))",
                "POLYGON((0 0,0 10,10 10,10 0,0 0),(5 5,6 5,6 6,5 6,5 5),(7 1,8 1,8 2,7 1),(1 1,2 1,2 2,1 1))",
            ),
            (
                "MULTILINESTRING((0 0,1 1),(5 5,6 6),(0 0,1 0,2 0))",
                "MULTILINESTRING((0 0,1 0,2 0),(5 5,6 6),(0 0,1 1))",
            ),
            (
                "MULTIPOINT((0 0),(5 5),(1 2))",
                "MULTIPOINT((5 5),(1 2),(0 0))",
            ),
            (
                "GEOMETRYCOLLECTION(POINT(9 9),LINESTRING(0 0,1 1),POLYGON((0 0,1 0,1 1,0 0)),POINT(0 0))",
                "GEOMETRYCOLLECTION(POLYGON((0 0,1 1,1 0,0 0)),LINESTRING(0 0,1 1),POINT(9 9),POINT(0 0))",
            ),
            ("LINESTRING(1 1,0 0,1 1)", "LINESTRING(0 0,1 1,0 0)"),
        ] {
            assert_eq!(norm_wkt(input), norm_wkt_ref(geos), "{input}");
        }
    }

    /// The reference WKT re-printed by the same writer (formatting only).
    fn norm_wkt_ref(wkt: &str) -> String {
        let geom: Geometry<f64> = Geometry::try_from_wkt_str(wkt).unwrap();
        geom.to_wkt().to_string()
    }
}
