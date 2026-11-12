use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::{ConvexHull, Geometry};
use serde::{Deserialize, Serialize};

use crate::utils::{unary_geom_to_geom, validate_geometry_field};

/// Compute the convex hull of a geometry.
///
/// `geo::ConvexHull` is implemented for any type that implements `CoordsIter`,
/// which includes `Geometry` directly in geo 0.33.  We wrap the call in
/// `catch_unwind` for defensive robustness (consistent with the buffer/overlay
/// ops pattern), even though convex-hull is unlikely to panic.
///
/// Degenerate hulls follow PostGIS: a single distinct point gives a `Point`,
/// collinear input gives the `LineString` between its two extreme points.
fn apply_convexhull(g: &Geometry) -> Option<Geometry> {
    if crate::utils::is_empty_geom(g) {
        return Some(g.clone());
    }
    let hull = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.convex_hull())).ok()?;
    let mut distinct: Vec<geo::Coord> = hull.exterior().0.clone();
    distinct.dedup();
    if distinct.len() > 1 && distinct.first() == distinct.last() {
        distinct.pop();
    }
    Some(match distinct.len() {
        0 => Geometry::Polygon(hull),
        1 => Geometry::Point(geo::Point::from(distinct[0])),
        2 => Geometry::LineString(geo::LineString::new(distinct)),
        _ => Geometry::Polygon(hull),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StConvexHull;

#[typetag::serde]
impl ScalarUDF for StConvexHull {
    fn name(&self) -> &'static str {
        "st_convexhull"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        unary_geom_to_geom(inputs.required(0)?, self.name(), apply_convexhull)
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
        "Returns the convex hull of a geometry as a Polygon. \
         For a single point returns a zero-area polygon; for two points a linestring-shaped polygon."
    }
}

#[must_use]
pub fn st_convexhull(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StConvexHull, vec![geom]).into()
}
