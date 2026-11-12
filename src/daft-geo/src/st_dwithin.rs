use common_error::DaftResult;
use daft_core::{prelude::{DataType, Field, Schema}, series::Series};
use daft_dsl::{ExprRef, functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn}};
use geo::Geometry;
use serde::{Deserialize, Serialize};

use crate::st_distance::geom_distance;
use crate::utils::{binary_geom_to_bool, read_f64_arg, read_f64_arg_expr, validate_geometry_field};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StDwithin;

#[typetag::serde]
impl ScalarUDF for StDwithin {
    fn name(&self) -> &'static str { "st_dwithin" }

    fn call(&self, inputs: FunctionArgs<Series>, _ctx: &daft_dsl::functions::scalar::EvalContext) -> DaftResult<Series> {
        let d = read_f64_arg(&inputs, 2, "distance", self.name())?;
        binary_geom_to_bool(inputs.required(0)?, inputs.required(1)?, self.name(),
            move |a: &Geometry, b: &Geometry| {
                Some(geom_distance(a, b)? <= d)
            })
    }

    fn get_return_field(&self, inputs: FunctionArgs<ExprRef>, schema: &Schema) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom_a", self.name())?;
        validate_geometry_field(&inputs, schema, 1, "geom_b", self.name())?;
        read_f64_arg_expr(&inputs, 2, "distance", self.name())?; // validate numeric literal at plan time
        Ok(Field::new(self.name(), DataType::Boolean))
    }

    fn docstring(&self) -> &'static str {
        "Returns true if the planar distance between A and B is <= distance (coordinate units)."
    }
}

#[must_use]
pub fn st_dwithin(geom_a: ExprRef, geom_b: ExprRef, distance: ExprRef) -> ExprRef {
    ScalarFn::builtin(StDwithin, vec![geom_a, geom_b, distance]).into()
}

#[cfg(test)]
mod tests {
    use geo::{Geometry, Point, LineString, Coord};
    use crate::st_distance::geom_distance;

    #[test]
    fn test_dwithin_boundary() {
        let a = Geometry::Point(Point::new(0.0, 0.0));
        let b = Geometry::Point(Point::new(3.0, 4.0)); // distance = 5.0
        let d = geom_distance(&a, &b).unwrap();
        assert!(d <= 5.0 + 1e-9);
        assert!(d > 4.9);
    }

    #[test]
    fn test_distance_defined_for_line_pairs() {
        // Previously only Point/Polygon/LineString-Point pairs were handled and every
        // other pair was NaN, so st_dwithin silently returned false for them.
        let line_a = Geometry::LineString(LineString(vec![
            Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 0.0 },
        ]));
        let line_b = Geometry::LineString(LineString(vec![
            Coord { x: 0.0, y: 3.0 }, Coord { x: 1.0, y: 3.0 },
        ]));
        assert_eq!(geom_distance(&line_a, &line_b), Some(3.0));
    }

    #[test]
    fn test_distance_empty_is_none() {
        let empty = Geometry::MultiPolygon(geo::MultiPolygon::empty());
        let p = Geometry::Point(Point::new(0.0, 0.0));
        assert_eq!(geom_distance(&empty, &p), None);
    }
}
