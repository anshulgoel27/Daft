use std::sync::Arc;

use arrow_buffer::NullBufferBuilder;
use daft_core::prelude::{Float64Array, IntoSeries};
use daft_dsl::functions::{prelude::*, scalar::ScalarFn};
use serde::{Deserialize, Serialize};

const EARTH_RADIUS_M: f64 = 6_371_000.0;

fn haversine_meters(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
    let lat1_rad = lat1_deg.to_radians();
    let lon1_rad = lon1_deg.to_radians();
    let lat2_rad = lat2_deg.to_radians();
    let lon2_rad = lon2_deg.to_radians();

    let delta_lat = lat2_rad - lat1_rad;
    let delta_lon = lon2_rad - lon1_rad;

    let sin_half_delta_lat = (delta_lat * 0.5).sin();
    let sin_half_delta_lon = (delta_lon * 0.5).sin();

    let a = (lat1_rad.cos() * lat2_rad.cos() * sin_half_delta_lon)
        .mul_add(sin_half_delta_lon, sin_half_delta_lat * sin_half_delta_lat)
        .clamp(0.0, 1.0);

    let central_angle = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_M * central_angle
}

fn is_valid_lat_lon(lat: f64, lon: f64) -> bool {
    lat.is_finite()
        && lon.is_finite()
        && (-90.0..=90.0).contains(&lat)
        && (-180.0..=180.0).contains(&lon)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct GreatCircleDistance;

#[typetag::serde]
impl ScalarUDF for GreatCircleDistance {
    fn name(&self) -> &'static str {
        "great_circle_distance"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        let lat1 = inputs.required(0)?.cast(&DataType::Float64)?;
        let lon1 = inputs.required(1)?.cast(&DataType::Float64)?;
        let lat2 = inputs.required(2)?.cast(&DataType::Float64)?;
        let lon2 = inputs.required(3)?.cast(&DataType::Float64)?;

        let lat1 = lat1.f64().expect("type should have been validated already");
        let lon1 = lon1.f64().expect("type should have been validated already");
        let lat2 = lat2.f64().expect("type should have been validated already");
        let lon2 = lon2.f64().expect("type should have been validated already");

        // Any argument may be a length-1 literal broadcast over the others.
        let cols = [lat1, lon1, lat2, lon2];
        let len = cols.iter().try_fold(1usize, |acc, c| {
            crate::utils::broadcast_len(acc, c.len(), self.name())
        })?;
        let at = |arr: &Float64Array, i: usize| arr.get(if arr.len() == 1 { 0 } else { i });

        let mut values = Vec::with_capacity(len);
        let mut validity = NullBufferBuilder::new(len);

        for i in 0..len {
            match (at(lat1, i), at(lon1, i), at(lat2, i), at(lon2, i)) {
                (Some(a), Some(b), Some(c), Some(d))
                    if is_valid_lat_lon(a, b) && is_valid_lat_lon(c, d) =>
                {
                    values.push(haversine_meters(a, b, c, d));
                    validity.append_non_null();
                }
                _ => {
                    values.push(0.0);
                    validity.append_null();
                }
            }
        }

        let field = Arc::new(Field::new(self.name(), DataType::Float64));
        let result =
            Float64Array::from_field_and_values(field, values).with_nulls(validity.finish())?;

        Ok(result.into_series())
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        ensure!(
            inputs.len() == 4,
            SchemaMismatch: "Expected 4 inputs, but received {}",
            inputs.len()
        );

        let lat1 = inputs.required(0)?.to_field(schema)?;
        let lon1 = inputs.required(1)?.to_field(schema)?;
        let lat2 = inputs.required(2)?.to_field(schema)?;
        let lon2 = inputs.required(3)?.to_field(schema)?;

        for (arg_name, field) in [
            ("lat1", &lat1),
            ("lon1", &lon1),
            ("lat2", &lat2),
            ("lon2", &lon2),
        ] {
            ensure!(
                field.dtype.is_numeric(),
                TypeError: "{} must be numeric, got {}",
                arg_name,
                field.dtype
            );
        }

        Ok(Field::new(self.name(), DataType::Float64))
    }

    fn docstring(&self) -> &'static str {
        "Computes the great-circle distance in meters between two points from \
         (lat1, lon1) and (lat2, lon2), where latitudes and longitudes are \
         specified in degrees, using the haversine formula. Null inputs or \
         non-finite/out-of-range coordinates produce null."
    }
}

#[must_use]
pub fn great_circle_distance(
    lat1: ExprRef,
    lon1: ExprRef,
    lat2: ExprRef,
    lon2: ExprRef,
) -> ExprRef {
    ScalarFn::builtin(GreatCircleDistance, vec![lat1, lon1, lat2, lon2]).into()
}
