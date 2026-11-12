use common_error::DaftResult;
use daft_core::{
    prelude::{DataType, Field, Schema},
    series::Series,
};
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::Geometry;
use serde::{Deserialize, Serialize};

use crate::utils::{read_f64_arg, read_f64_arg_expr, unary_geom_to_geom, validate_geometry_field};

/// Apply a planar buffer of `distance` units to geometry `g`, PostGIS-style.
///
/// Uses `geo::Buffer` (i_overlay outline/stroke) with round joins and round end
/// caps, ~32 segments per full circle (PostGIS `quad_segs=8`). i_overlay snaps to
/// an integer grid sized relative to the input's extent, so the result does not
/// depend on the coordinate scale (degrees or metres).
///
/// - **Point / MultiPoint**: a circle per point; `distance <= 0` gives EMPTY.
/// - **LineString / MultiLineString**: a corridor with round caps; `distance <= 0`
///   gives EMPTY.
/// - **Polygon / MultiPolygon**: an outward (or, for negative distances, inward)
///   offset. An inward offset that consumes the polygon gives EMPTY.
/// - **GeometryCollection**: the union of its members' buffers.
///
/// A single-part result is a `Polygon`, otherwise a `MultiPolygon` (EMPTY when
/// nothing is left). Returns null if the engine panics on degenerate input.
///
/// All buffer operations are **planar** (Cartesian). For geodesic buffers, project your
/// coordinates to a local metric CRS before calling this function.
fn apply_buffer(g: &Geometry, distance: f64) -> Option<Geometry> {
    use geo::Buffer;

    if !distance.is_finite() {
        return None;
    }
    if crate::utils::is_empty_geom(g) {
        return Some(Geometry::MultiPolygon(geo::MultiPolygon::empty()));
    }
    // Buffering a geometry with no area inward (or by zero) leaves nothing.
    let areal = matches!(
        g,
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) | Geometry::Rect(_) | Geometry::Triangle(_)
    ) || matches!(g, Geometry::GeometryCollection(gc) if gc.iter().any(|m| matches!(m, Geometry::Polygon(_) | Geometry::MultiPolygon(_) | Geometry::Rect(_) | Geometry::Triangle(_))));
    if distance <= 0.0 && !areal {
        return Some(Geometry::MultiPolygon(geo::MultiPolygon::empty()));
    }

    let mp = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.buffer(distance))).ok()?;
    Some(crate::overlay::areal_geometry(mp))
}

/// Zero-field unit struct; the `distance` parameter is read from the trailing
/// positional argument at call time (not stored in the struct).
///
/// This is necessary because the Python `_call_builtin_scalar_fn` path resolves
/// the registered instance by name and cannot pass struct-field parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StBuffer;

#[typetag::serde]
impl ScalarUDF for StBuffer {
    fn name(&self) -> &'static str {
        "st_buffer"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        let distance = read_f64_arg(&inputs, 1, "distance", self.name())?;
        unary_geom_to_geom(inputs.required(0)?, self.name(), |g| {
            apply_buffer(g, distance)
        })
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom", self.name())?;
        // Validate distance arg exists and is numeric
        let _distance = read_f64_arg_expr(&inputs, 1, "distance", self.name())?;
        Ok(Field::new(self.name(), DataType::Geometry))
    }

    fn docstring(&self) -> &'static str {
        "Returns a geometry covering all points within the given distance of the input \
         (planar Cartesian), with round joins and end caps (~32 segments per circle, like \
         PostGIS quad_segs=8). Works for every geometry type; points and lines buffered by \
         distance <= 0 and polygons fully eroded by a negative distance give MULTIPOLYGON EMPTY. \
         A single part is returned as a POLYGON, several as a MULTIPOLYGON."
    }
}

#[must_use]
pub fn st_buffer(geom: ExprRef, distance: f64) -> ExprRef {
    ScalarFn::builtin(StBuffer, vec![geom, daft_dsl::lit(distance)]).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Area, Geometry, Point};

    #[test]
    fn test_point_buffer_area_approx_pi_r2() {
        let p = Geometry::Point(Point::new(0.0_f64, 0.0));
        let buf = apply_buffer(&p, 1.0).unwrap();
        // A 32-segment circle (PostGIS quad_segs=8) of radius 1 has area
        // 16 * sin(2π/32) ≈ 3.1214, matching ST_Area(ST_Buffer('POINT(0 0)', 1)).
        let a = buf.unsigned_area();
        let expected = 16.0 * (2.0 * std::f64::consts::PI / 32.0).sin();
        assert!((a - expected).abs() < 1e-3, "expected area ≈ {expected}, got {a}");
    }

    #[test]
    fn test_polygon_buffer_expands() {
        use geo::{LineString, Polygon};
        // Unit square
        let poly = Polygon::new(
            LineString::from(vec![(0.0_f64, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]),
            vec![],
        );
        let g = Geometry::Polygon(poly);
        let buf = apply_buffer(&g, 1.0).unwrap();
        // Result should have larger area than the unit square (area = 1)
        let a = buf.unsigned_area();
        assert!(a > 1.0, "buffered polygon area {a} should exceed unit square area");
    }
}
