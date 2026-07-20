use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::Geometry;
use serde::{Deserialize, Serialize};

use crate::utils::{unary_geom_to_list_dump_item, validate_geometry_field};

fn dump_geometry_parts_with_paths(geom: &Geometry, prefix: &[i64]) -> Vec<(Vec<i64>, Geometry)> {
    match geom {
        Geometry::MultiPoint(multi_point) => multi_point
            .0
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let mut path = prefix.to_vec();
                path.push((i + 1) as i64);
                (path, Geometry::Point(point.clone()))
            })
            .collect(),
        Geometry::MultiLineString(multi_line_string) => multi_line_string
            .0
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let mut path = prefix.to_vec();
                path.push((i + 1) as i64);
                (path, Geometry::LineString(line.clone()))
            })
            .collect(),
        Geometry::MultiPolygon(multi_polygon) => multi_polygon
            .0
            .iter()
            .enumerate()
            .map(|(i, poly)| {
                let mut path = prefix.to_vec();
                path.push((i + 1) as i64);
                (path, Geometry::Polygon(poly.clone()))
            })
            .collect(),
        Geometry::GeometryCollection(collection) => collection
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, child)| {
                let mut child_prefix = prefix.to_vec();
                child_prefix.push((i + 1) as i64);
                dump_geometry_parts_with_paths(child, &child_prefix)
            })
            .collect(),
        _ => vec![(prefix.to_vec(), geom.clone())],
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
        unary_geom_to_list_dump_item(inputs.required(0)?, self.name(), |geom| {
            Some(dump_geometry_parts_with_paths(geom, &[]))
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
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new("path", DataType::List(Box::new(DataType::Int64))),
                Field::new("geom", DataType::Geometry),
            ]))),
        ))
    }

    fn docstring(&self) -> &'static str {
        "Returns the geometry's components as List[Struct{path, geom}]. Atomic geometries return one item with empty path; multi-geometries return one item per member with 1-based path indexes; GeometryCollections are recursively flattened with nested paths."
    }
}

#[must_use]
pub fn st_dump(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StDump, vec![geom]).into()
}