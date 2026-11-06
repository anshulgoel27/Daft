use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::{Geometry, LineString, MakeValid, MultiPolygon, Polygon};
use serde::{Deserialize, Serialize};

use crate::overlay::areal_geometry;
use crate::utils::{unary_geom_to_geom, validate_geometry_field};

/// Repair an invalid geometry.
///
/// The pure-Rust `geo` crate only implements `MakeValid` for `Polygon` and
/// `MultiPolygon` (it repairs self-intersections, bowties, etc. and returns a
/// `MultiPolygon`). Unlike PostGIS/GEOS, it cannot repair other geometry types,
/// so non-polygonal inputs are passed through unchanged — a valid Point or
/// LineString is already valid. Returns `None` (null) when the repair fails.
///
/// Valid polygonal input is returned unchanged, matching PostGIS. A
/// single-part repair is a `Polygon`, otherwise a `MultiPolygon`.
///
/// Needs `geo` with georust/geo#1640 fixed: 0.33.1 snaps within an absolute
/// `1e-4` units (about 10 m in degrees) and empties small polygons.
///
/// Wrapped in `catch_unwind` for defensive robustness, matching the overlay/
/// convex-hull ops, since geometric repair can panic on degenerate input.
fn apply_makevalid(g: &Geometry) -> Option<Geometry> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (input, repaired) = match g {
            Geometry::Polygon(p) => (std::slice::from_ref(p), p.make_valid().ok()?),
            Geometry::MultiPolygon(mp) => (mp.0.as_slice(), mp.make_valid().ok()?),
            other => return Some(other.clone()),
        };
        Some(if is_unchanged(input, &repaired) {
            g.clone()
        } else {
            areal_geometry(repaired)
        })
    }));
    result.ok().flatten()
}

/// True when the repair consists of exactly the input's rings (same vertex
/// cycles, up to start vertex and direction) — i.e. the input was already
/// valid, so it is returned untouched (PostGIS semantics).
///
/// This replaces an `is_valid()` pre-check, which is super-linear (~100x the
/// repair itself on an 8000-vertex polygon) and so dominated the common
/// valid-input case. The repair keeps input vertices bit-exact (its snap
/// radius is far below the spacing of distinct vertices), so a valid polygon's
/// rings compare equal. When they differ, the
/// repaired geometry is returned, which is correct either way. Zero-area rings
/// are dropped by the repair, so they never compare equal.
fn is_unchanged(input: &[Polygon], repaired: &MultiPolygon) -> bool {
    input.len() == repaired.0.len() && polygons_key(input) == polygons_key(&repaired.0)
}

type RingKey = Vec<(u64, u64)>;

fn polygons_key(polys: &[Polygon]) -> Vec<(RingKey, Vec<RingKey>)> {
    let mut keys: Vec<_> = polys
        .iter()
        .map(|p| {
            let mut holes: Vec<RingKey> = p.interiors().iter().map(ring_key).collect();
            holes.sort();
            (ring_key(p.exterior()), holes)
        })
        .collect();
    keys.sort();
    keys
}

/// Canonical vertex cycle of a closed ring: rotated to start at its smallest
/// vertex, in whichever direction compares smaller.
fn ring_key(ring: &LineString) -> RingKey {
    let mut c: Vec<(u64, u64)> = ring.0.iter().map(|c| (c.x.to_bits(), c.y.to_bits())).collect();
    if c.len() > 1 && c.first() == c.last() {
        c.pop();
    }
    let Some(start) = (0..c.len()).min_by_key(|&i| c[i]) else {
        return c;
    };
    c.rotate_left(start);
    let mut rev = c.clone();
    rev[1..].reverse();
    c.min(rev)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StMakeValid;

#[typetag::serde]
impl ScalarUDF for StMakeValid {
    fn name(&self) -> &'static str {
        "st_makevalid"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        unary_geom_to_geom(inputs.required(0)?, self.name(), apply_makevalid)
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
        "Repairs invalid polygonal geometries (self-intersections, bowties, etc.), \
         returning a POLYGON when the repair has one part and a MULTIPOLYGON otherwise; valid \
         polygonal input is returned unchanged. Non-polygonal geometries (Point, LineString, ...) \
         are returned unchanged, since the pure-Rust engine only repairs areal geometries. \
         Returns null when the geometry cannot be repaired."
    }
}

#[must_use]
pub fn st_makevalid(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StMakeValid, vec![geom]).into()
}
