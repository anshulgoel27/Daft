use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::{
    Coord, Geometry, GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon,
    Point, Polygon,
    orient::{Direction, Orient},
};
use serde::{Deserialize, Serialize};

use crate::utils::{geom_to_wkb, unary_geom_to_geom, validate_geometry_field};

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
    interiors.sort_by(|a, b| coords_cmp(&a.0, &b.0));
    Polygon::new(exterior, interiors)
}

/// Compare two normalized polygons by exterior ring first, falling through to a
/// pairwise comparison of (already deterministically-ordered) interior rings, then hole
/// count, so polygons that share an exterior but differ only in their holes don't tie.
fn compare_polygons(a: &Polygon<f64>, b: &Polygon<f64>) -> std::cmp::Ordering {
    coords_cmp(&a.exterior().0, &b.exterior().0).then_with(|| {
        let (a_ints, b_ints) = (a.interiors(), b.interiors());
        a_ints
            .iter()
            .zip(b_ints.iter())
            .map(|(ra, rb)| coords_cmp(&ra.0, &rb.0))
            .find(|o| *o != std::cmp::Ordering::Equal)
            .unwrap_or_else(|| a_ints.len().cmp(&b_ints.len()))
    })
}

/// Recursively canonicalize a geometry so that spatially-equivalent inputs (differing
/// only in ring orientation, ring starting vertex, or non-semantic part ordering)
/// produce an identical output, per the `ST_Normalize` convention. Used upstream of
/// hash generation to avoid false-positive change detections. Returns `None` if a member
/// geometry cannot be WKB-encoded during collection sorting, rather than silently treating
/// the encoding failure as a sort tie.
fn normalize_geometry(geom: &Geometry<f64>) -> Option<Geometry<f64>> {
    match geom {
        Geometry::Polygon(p) => Some(Geometry::Polygon(normalize_polygon(p))),
        Geometry::MultiPolygon(mp) => {
            let mut polys: Vec<Polygon<f64>> = mp.iter().map(normalize_polygon).collect();
            polys.sort_by(compare_polygons);
            Some(Geometry::MultiPolygon(MultiPolygon(polys)))
        }
        Geometry::MultiLineString(mls) => {
            let mut lines: Vec<LineString<f64>> = mls.iter().cloned().collect();
            lines.sort_by(|a, b| coords_cmp(&a.0, &b.0));
            Some(Geometry::MultiLineString(MultiLineString::new(lines)))
        }
        Geometry::MultiPoint(mp) => {
            let mut pts: Vec<Point<f64>> = mp.iter().cloned().collect();
            pts.sort_by(|a, b| coord_cmp(&a.0, &b.0));
            Some(Geometry::MultiPoint(MultiPoint(pts)))
        }
        Geometry::GeometryCollection(gc) => {
            let mut members = gc.iter().map(normalize_geometry).collect::<Option<Vec<_>>>()?;
            // `sort_by_cached_key` WKB-encodes each member once (instead of on every
            // comparison) for a total, deterministic order across mixed geometry types.
            let mut encode_failed = false;
            members.sort_by_cached_key(|g| {
                geom_to_wkb(g).unwrap_or_else(|_| {
                    encode_failed = true;
                    Vec::new()
                })
            });
            if encode_failed {
                return None;
            }
            Some(Geometry::GeometryCollection(GeometryCollection::new_from(
                members,
            )))
        }
        // Point and LineString have no non-semantic ordering to canonicalize; other
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
}
