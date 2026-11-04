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

use crate::overlay::{OverlayOp, Prepared, overlay};
use crate::utils::{binary_prepared_to_geom, validate_geometry_field};

fn op_symdifference(a: &Prepared, b: &Prepared) -> Option<Geometry> {
    overlay(a, b, OverlayOp::SymDifference)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StSymDifference;

#[typetag::serde]
impl ScalarUDF for StSymDifference {
    fn name(&self) -> &'static str {
        "st_symdifference"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        binary_prepared_to_geom(
            inputs.required(0)?,
            inputs.required(1)?,
            self.name(),
            Prepared::new,
            op_symdifference,
        )
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom_a", self.name())?;
        validate_geometry_field(&inputs, schema, 1, "geom_b", self.name())?;
        Ok(Field::new(self.name(), DataType::Geometry))
    }

    fn docstring(&self) -> &'static str {
        "Returns the symmetric difference of two geometries of any type (points, lines, polygons, collections). \
         A single polygon is returned as a POLYGON, several as a MULTIPOLYGON (empty areal result: \
         MULTIPOLYGON EMPTY); mixed-dimension results are a GEOMETRYCOLLECTION. Vertices taken from \
         the inputs are preserved exactly."
    }
}

#[must_use]
pub fn st_symdifference(geom_a: ExprRef, geom_b: ExprRef) -> ExprRef {
    ScalarFn::builtin(StSymDifference, vec![geom_a, geom_b]).into()
}
