use common_error::{DaftError, DaftResult};
use daft_core::{
    prelude::{BooleanArray, DataType, Field, Schema},
    series::{IntoSeries, Series},
};
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF},
};
use serde::{Deserialize, Serialize};

/// `st_geohash_covers(hash_col, query_wkb, precision)` → Boolean
///
/// Returns true if the geohash cell `hash_col` overlaps the bounding box of `query_wkb`.
/// Used for automatic partition pruning: add a `_geohash` column to your table and Daft
/// will rewrite `st_intersects(geom, query)` to also filter on `_geohash`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StGeohashCovers {
    pub precision: u8,
    /// Pre-computed covering cells (serialized as newline-separated hashes)
    pub covering_cells: String,
}

#[typetag::serde]
impl ScalarUDF for StGeohashCovers {
    fn name(&self) -> &'static str {
        "st_geohash_covers"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        let hash_series = inputs.required(0)?;
        let covering: Vec<String> = self
            .covering_cells
            .split('\n')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        let hash_arr = hash_series.utf8()?;

        // Optional geometry column (argument 1). `{col}_geohash` is the geohash
        // of each row's *centroid*, which only bounds where a Point row lies:
        // a polygon can intersect or contain the query while its centroid sits
        // in a cell outside the covering. When the geometry is supplied, only
        // single-Point rows are pruned; every other row passes through to the
        // exact predicate. A row with a null geohash is never pruned.
        let geoms = match inputs.optional(1)? {
            Some(g) => Some(crate::utils::get_geometry_binary(g)?),
            None => None,
        };
        let values: Vec<Option<bool>> = hash_arr
            .into_iter()
            .enumerate()
            .map(|(i, hash)| {
                let geom = match geoms {
                    None => RowGeom::NotGiven,
                    Some(g) => RowGeom::Wkb(g.get(i)),
                };
                Some(keep_row(hash, geom, &covering))
            })
            .collect();

        Ok(BooleanArray::from_iter(self.name(), values.into_iter()).into_series())
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        let f = inputs.required(0)?.to_field(schema)?;
        if !matches!(f.dtype, DataType::Utf8) {
            return Err(DaftError::TypeError(format!(
                "st_geohash_covers: hash column must be Utf8, got {}",
                f.dtype
            )));
        }
        if let Some(g) = inputs.optional(1)? {
            let gf = g.to_field(schema)?;
            if !matches!(gf.dtype, DataType::Geometry | DataType::Binary) {
                return Err(DaftError::TypeError(format!(
                    "st_geohash_covers: geom must be Geometry or Binary, got {}",
                    gf.dtype
                )));
            }
        }
        Ok(Field::new(self.name(), DataType::Boolean))
    }

    fn docstring(&self) -> &'static str {
        "Returns true if the geohash cell overlaps with the pre-computed covering cells of a query geometry. \
         With the optional geometry argument, non-Point rows always pass (a centroid geohash \
         does not bound where a polygon lies). Rows with a null geohash always pass. \
         Used internally for geohash-based partition pruning."
    }
}

/// The row's geometry, as far as the pre-filter knows it.
#[derive(Clone, Copy)]
enum RowGeom<'a> {
    /// No geometry argument was given: every row is prunable.
    NotGiven,
    /// The row's WKB (`None` = null).
    Wkb(Option<&'a [u8]>),
}

/// Pre-filter decision for one row. Only a row known to be a Point (or any row
/// when no geometry argument was given) with a known geohash can be pruned.
fn keep_row(hash: Option<&str>, geom: RowGeom<'_>, covering: &[String]) -> bool {
    let prunable = match geom {
        RowGeom::NotGiven => true,
        RowGeom::Wkb(wkb) => wkb.is_some_and(wkb_is_point),
    };
    match hash {
        Some(h) if prunable => covering.iter().any(|c| h.starts_with(c.as_str()) || c.starts_with(h)),
        _ => true,
    }
}

/// True when the WKB (ISO or EWKB) encodes a single `Point`. Reads only the
/// 5-byte header, so it is far cheaper than parsing the geometry.
pub(crate) fn wkb_is_point(wkb: &[u8]) -> bool {
    let Some(header) = wkb.get(0..5) else {
        return false;
    };
    let ty = [header[1], header[2], header[3], header[4]];
    let raw = match header[0] {
        0 => u32::from_be_bytes(ty),
        1 => u32::from_le_bytes(ty),
        _ => return false,
    };
    // Strip EWKB Z/M/SRID flag bits, then ISO's +1000/+2000/+3000 dimension offsets.
    (raw & 0x0FFF_FFFF) % 1000 == 1
}

#[cfg(test)]
mod tests {
    use super::{RowGeom, keep_row, wkb_is_point};

    #[test]
    fn prunes_only_points_outside_the_covering() {
        let cover = vec!["dqcjq".to_string()];
        let point = [1u8, 1, 0, 0, 0];
        let polygon = [1u8, 3, 0, 0, 0];
        // Point inside / outside the covering.
        let (pt, poly) = (RowGeom::Wkb(Some(&point)), RowGeom::Wkb(Some(&polygon)));
        assert!(keep_row(Some("dqcjqc1"), pt, &cover));
        assert!(!keep_row(Some("9q8yyk8"), pt, &cover));
        // A polygon is never pruned, even far away.
        assert!(keep_row(Some("9q8yyk8"), poly, &cover));
        // Null geohash or null geometry: never pruned.
        assert!(keep_row(None, pt, &cover));
        assert!(keep_row(Some("9q8yyk8"), RowGeom::Wkb(None), &cover));
        // No geometry argument (st_within / st_covered_by): every row is prunable.
        assert!(!keep_row(Some("9q8yyk8"), RowGeom::NotGiven, &cover));
        assert!(keep_row(Some("dqcjqc1"), RowGeom::NotGiven, &cover));
    }

    #[test]
    fn detects_point_headers() {
        assert!(wkb_is_point(&[1, 1, 0, 0, 0]));
        assert!(wkb_is_point(&[0, 0, 0, 0, 1]));
        assert!(wkb_is_point(&[1, 0xE9, 0x03, 0, 0])); // ISO Point Z (1001)
        assert!(wkb_is_point(&[1, 1, 0, 0, 0x20])); // EWKB Point with SRID
        assert!(!wkb_is_point(&[1, 3, 0, 0, 0])); // Polygon
        assert!(!wkb_is_point(&[1, 4, 0, 0, 0])); // MultiPoint
        assert!(!wkb_is_point(&[1, 1, 0]));
    }
}
