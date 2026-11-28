use common_error::DaftResult;
use daft_core::{prelude::{DataType, Field, Schema}, series::Series};
use daft_dsl::{ExprRef, functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn}};
use geo::Area;
use geo::algorithm::geodesic_area::GeodesicArea;
use geo::orient::{Direction, Orient};
use geo::{Geometry, Polygon};
use serde::{Deserialize, Serialize};

use crate::utils::{unary_geom_to_f64, validate_geometry_field, read_bool_arg, read_bool_arg_expr};

/// WGS84 geodesic area in m², independent of ring winding.
///
/// geo's unsigned geodesic area takes a clockwise exterior ring to enclose the
/// rest of the globe, so a clockwise (ESRI/shapefile-wound) building measured
/// ~5.1e14 m². Each polygon is re-wound to the Simple Features order (exterior
/// counter-clockwise, holes clockwise) first, so its smaller side is measured,
/// as PostGIS `ST_Area(geography)` does.
fn geodesic_area_m2(g: &Geometry) -> f64 {
    let poly = |p: &Polygon| p.orient(Direction::Default).geodesic_area_unsigned();
    match g {
        Geometry::Polygon(p) => poly(p),
        Geometry::MultiPolygon(mp) => mp.iter().map(poly).sum(),
        Geometry::GeometryCollection(gc) => gc.iter().map(geodesic_area_m2).sum(),
        _ => 0.0,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StArea;

#[typetag::serde]
impl ScalarUDF for StArea {
    fn name(&self) -> &'static str { "st_area" }

    fn call(&self, inputs: FunctionArgs<Series>, _ctx: &daft_dsl::functions::scalar::EvalContext) -> DaftResult<Series> {
        let use_spheroid = read_bool_arg(&inputs, 1, "use_spheroid", self.name())?;
        unary_geom_to_f64(inputs.required(0)?, self.name(), |g| {
            if use_spheroid {
                geodesic_area_m2(g)
            } else {
                g.unsigned_area()
            }
        })
    }

    fn get_return_field(&self, inputs: FunctionArgs<ExprRef>, schema: &Schema) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom", self.name())?;
        read_bool_arg_expr(&inputs, 1, "use_spheroid", self.name())?;
        Ok(Field::new(self.name(), DataType::Float64))
    }

    fn docstring(&self) -> &'static str {
        "2D area. Coordinate units² by default; WGS84 geodesic m² when use_spheroid=true (lon/lat input)."
    }
}

#[must_use]
pub fn st_area(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StArea, vec![geom]).into()
}
