use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::{BoundingRect, Geometry};
use serde::{Deserialize, Serialize};

use crate::utils::{unary_geom_to_geom, validate_geometry_field};

/// Return the bounding-box envelope of a geometry as a Polygon.
///
/// Degenerate envelopes follow PostGIS: a zero-size box is a `Point`, and a
/// zero-width or zero-height box is a `LineString` from its min to max corner.
/// For a geometry with no bounding rect (e.g. an empty geometry collection)
/// this returns null.
fn apply_envelope(g: &Geometry) -> Option<Geometry> {
    if crate::utils::is_empty_geom(g) {
        // PostGIS returns the empty input itself (e.g. POINT EMPTY).
        return Some(g.clone());
    }
    let r = g.bounding_rect()?;
    Some(match (r.width() == 0.0, r.height() == 0.0) {
        (true, true) => Geometry::Point(geo::Point::from(r.min())),
        (true, false) | (false, true) => {
            Geometry::LineString(geo::LineString::new(vec![r.min(), r.max()]))
        }
        (false, false) => Geometry::Polygon(r.to_polygon()),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StEnvelope;

#[typetag::serde]
impl ScalarUDF for StEnvelope {
    fn name(&self) -> &'static str {
        "st_envelope"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        unary_geom_to_geom(inputs.required(0)?, self.name(), apply_envelope)
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
        "Returns the minimum bounding rectangle of a geometry as a Polygon. \
         For points returns a degenerate polygon; for geometries with no extent returns null."
    }
}

#[must_use]
pub fn st_envelope(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StEnvelope, vec![geom]).into()
}
