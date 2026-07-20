use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::Geometry;
use serde::{Deserialize, Serialize};

use crate::utils::{unary_geom_to_list_geom, validate_geometry_field};

fn dump_geometry_parts(geom: &Geometry) -> Vec<Geometry> {
    match geom {
        Geometry::MultiPoint(multi_point) => multi_point
            .0
            .iter()
            .cloned()
            .map(Geometry::Point)
            .collect(),
        Geometry::MultiLineString(multi_line_string) => multi_line_string
            .0
            .iter()
            .cloned()
            .map(Geometry::LineString)
            .collect(),
        Geometry::MultiPolygon(multi_polygon) => multi_polygon
            .0
            .iter()
            .cloned()
            .map(Geometry::Polygon)
            .collect(),
        Geometry::GeometryCollection(collection) => collection
            .0
            .iter()
            .flat_map(dump_geometry_parts)
            .collect(),
        _ => vec![geom.clone()],
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StDump;

#[typetag::serde]
impl ScalarUDF for StDump {
    fn name(&self) -> &'static str {
        "st_dump"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        unary_geom_to_list_geom(inputs.required(0)?, self.name(), |geom| {
            Some(dump_geometry_parts(geom))
        })
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom", self.name())?;
        Ok(Field::new(
            self.name(),
            DataType::List(Box::new(DataType::Geometry)),
        ))
    }

    fn docstring(&self) -> &'static str {
        "Returns the geometry's components as a List[Geometry]. Atomic geometries return a single-item list; multi-geometries return one item per member; GeometryCollections are recursively flattened."
    }
}

#[must_use]
pub fn st_dump(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StDump, vec![geom]).into()
}