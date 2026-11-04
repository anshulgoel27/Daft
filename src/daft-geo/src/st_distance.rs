use common_error::DaftResult;
use daft_core::{prelude::{DataType, Field, Schema}, series::Series};
use daft_dsl::{ExprRef, functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn}};
use geo::{
    BoundingRect, Closest, ClosestPoint, CoordsIter, Distance, Euclidean, Geodesic, Geometry,
    Intersects, MapCoords, Point,
};
use serde::{Deserialize, Serialize};

use crate::utils::{is_empty_geom, binary_geom_to_f64, validate_geometry_field, read_bool_arg, read_bool_arg_expr};

/// Planar minimum distance (coordinate units) between any two geometries.
///
/// Returns `None` when either geometry is empty (PostGIS returns NULL), or the
/// result is not finite.
pub fn geom_distance(a: &Geometry, b: &Geometry) -> Option<f64> {
    if is_empty_geom(a) || is_empty_geom(b) {
        return None;
    }
    let d = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Euclidean.distance(a, b)))
        .ok()?;
    d.is_finite().then_some(d)
}

/// WGS84 geodesic minimum distance (meters) between lon/lat geometries.
///
/// Point/Point is exact (Karney). For other pairs the closest pair of points is
/// found in a local equirectangular frame (longitude scaled by the cosine of
/// the mean latitude), then measured geodesically. Between non-intersecting
/// geometries the minimum is always attained at a vertex of one of them, so
/// checking every vertex of each against the other finds that pair. The frame
/// is accurate for features up to a few hundred km apart; beyond that the
/// chosen pair can differ slightly from the true geodesic closest pair.
fn geom_distance_geodesic(a: &Geometry, b: &Geometry) -> Option<f64> {
    if is_empty_geom(a) || is_empty_geom(b) {
        return None;
    }
    if let (Geometry::Point(pa), Geometry::Point(pb)) = (a, b) {
        let d = Geodesic.distance(*pa, *pb);
        return d.is_finite().then_some(d);
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if a.intersects(b) {
            return Some(0.0);
        }
        let (ra, rb) = (a.bounding_rect()?, b.bounding_rect()?);
        let mean_lat = f64::midpoint(ra.center().y, rb.center().y);
        let kx = mean_lat.to_radians().cos().max(1e-6);
        let (pa, pb) = (
            a.map_coords(|c| geo::Coord { x: c.x * kx, y: c.y }),
            b.map_coords(|c| geo::Coord { x: c.x * kx, y: c.y }),
        );
        let mut best: Option<(f64, Point, Point)> = None;
        // Vertices are visited in order of their distance to the other side's
        // bbox, a lower bound on their distance to it, and the scan stops once
        // that bound exceeds the best pair so far. Typically only the few
        // facing vertices reach the O(m) `closest_point` call.
        let mut consider = |from: &Geometry, to: &Geometry, swap: bool| {
            let Some(r) = to.bounding_rect() else { return };
            let (lo, hi) = (r.min(), r.max());
            let mut verts: Vec<(f64, Point)> = from
                .coords_iter()
                .map(|c| {
                    let dx = (lo.x - c.x).max(c.x - hi.x).max(0.0);
                    let dy = (lo.y - c.y).max(c.y - hi.y).max(0.0);
                    (dx.hypot(dy), Point::from(c))
                })
                .collect();
            verts.sort_by(|x, y| x.0.total_cmp(&y.0));
            for (bound, p) in verts {
                if best.as_ref().is_some_and(|(bd, _, _)| bound > *bd) {
                    break;
                }
                if let Closest::SinglePoint(q) | Closest::Intersection(q) = to.closest_point(&p) {
                    let d = Euclidean.distance(p, q);
                    if best.as_ref().is_none_or(|(bd, _, _)| d < *bd) {
                        best = Some(if swap { (d, q, p) } else { (d, p, q) });
                    }
                }
            }
        };
        consider(&pa, &pb, false);
        consider(&pb, &pa, true);
        let (_, p, q) = best?;
        let unproject = |p: Point| Point::new(p.x() / kx, p.y());
        let d = Geodesic.distance(unproject(p), unproject(q));
        d.is_finite().then_some(d)
    }))
    .ok()
    .flatten()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StDistance;

#[typetag::serde]
impl ScalarUDF for StDistance {
    fn name(&self) -> &'static str { "st_distance" }

    fn call(&self, inputs: FunctionArgs<Series>, _ctx: &daft_dsl::functions::scalar::EvalContext) -> DaftResult<Series> {
        let use_spheroid = read_bool_arg(&inputs, 2, "use_spheroid", self.name())?;
        let f = if use_spheroid { geom_distance_geodesic } else { geom_distance };
        binary_geom_to_f64(inputs.required(0)?, inputs.required(1)?, self.name(), f)
    }

    fn get_return_field(&self, inputs: FunctionArgs<ExprRef>, schema: &Schema) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom_a", self.name())?;
        validate_geometry_field(&inputs, schema, 1, "geom_b", self.name())?;
        read_bool_arg_expr(&inputs, 2, "use_spheroid", self.name())?;
        Ok(Field::new(self.name(), DataType::Float64))
    }

    fn docstring(&self) -> &'static str {
        "Minimum distance between A and B, for any geometry types. Planar (coordinate units) by default; WGS84 geodesic meters when use_spheroid=true (lon/lat inputs). Null when either input is empty."
    }
}

#[must_use]
pub fn st_distance(geom_a: ExprRef, geom_b: ExprRef) -> ExprRef {
    ScalarFn::builtin(StDistance, vec![geom_a, geom_b]).into()
}
