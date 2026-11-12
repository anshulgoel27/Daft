use common_error::DaftResult;
use daft_core::prelude::{DataType, Field, Schema};
use daft_core::series::Series;
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::{Coord, Geometry, Line, LineString, MultiLineString, MultiPolygon, Polygon};
use serde::{Deserialize, Serialize};

use crate::utils::{read_f64_arg, read_f64_arg_expr, unary_geom_to_geom, validate_geometry_field};

/// Apply Ramer–Douglas–Peucker simplification with the given tolerance.
///
/// Lines are simplified with PostGIS's Douglas–Peucker (endpoints always kept). Polygon
/// rings are simplified as closed lines and, as in PostGIS `ST_Simplify`, a
/// ring that collapses below a triangle is dropped: a collapsed hole is
/// removed, and a polygon whose shell collapses is removed. If nothing is left
/// the result is null, as PostGIS returns NULL.
/// geo's own polygon simplification instead keeps a ring *unchanged* when
/// culling would collapse it, so a larger tolerance could simplify less.
///
/// Points pass through unchanged; collections are simplified member-wise.
fn apply_simplify(g: &Geometry, tolerance: f64) -> Option<Geometry> {
    let out =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| simplify_geom(g, tolerance))).ok()?;
    // PostGIS returns NULL when simplification collapses the whole geometry.
    (crate::utils::is_empty_geom(&out) == crate::utils::is_empty_geom(g)).then_some(out)
}

/// Douglas–Peucker as PostGIS implements it (`ptarray_simplify_in_place`):
/// endpoints are kept, and among equally far points the *first* splits the
/// range (geo's `Simplify` takes the last, which can keep a different vertex
/// of a closed ring). The distance is to the segment between the range ends.
fn douglas_peucker(coords: &[Coord], tolerance: f64) -> Vec<Coord> {
    use geo::{Distance, Euclidean};
    if coords.len() <= 2 {
        return coords.to_vec();
    }
    let mut keep = vec![false; coords.len()];
    keep[0] = true;
    keep[coords.len() - 1] = true;
    let mut stack = vec![(0usize, coords.len() - 1)];
    while let Some((first, last)) = stack.pop() {
        if last <= first + 1 {
            continue;
        }
        let base = Line::new(coords[first], coords[last]);
        let mut split = first;
        let mut max_d = -1.0f64;
        for (i, c) in coords.iter().enumerate().take(last).skip(first + 1) {
            let d = Euclidean.distance(*c, &base);
            if d > max_d {
                max_d = d;
                split = i;
            }
        }
        if max_d > tolerance {
            keep[split] = true;
            stack.push((first, split));
            stack.push((split, last));
        }
    }
    coords.iter().zip(keep).filter_map(|(c, k)| k.then_some(*c)).collect()
}

fn simplify_line(ls: &LineString, tolerance: f64) -> LineString {
    LineString::new(douglas_peucker(&ls.0, tolerance))
}

fn simplify_ring(ring: &LineString, tolerance: f64) -> Option<LineString> {
    let simplified = simplify_line(ring, tolerance);
    // A valid ring needs at least 3 distinct vertices plus the closing one.
    (simplified.0.len() >= 4).then_some(simplified)
}

fn simplify_polygon(p: &Polygon, tolerance: f64) -> Option<Polygon> {
    let exterior = simplify_ring(p.exterior(), tolerance)?;
    let interiors = p.interiors().iter().filter_map(|r| simplify_ring(r, tolerance)).collect();
    Some(Polygon::new(exterior, interiors))
}

fn simplify_geom(g: &Geometry, tolerance: f64) -> Geometry {
    if tolerance.is_nan() || tolerance <= 0.0 {
        return g.clone();
    }
    match g {
        Geometry::LineString(ls) => Geometry::LineString(simplify_line(ls, tolerance)),
        Geometry::MultiLineString(mls) => Geometry::MultiLineString(MultiLineString::new(
            mls.iter().map(|ls| simplify_line(ls, tolerance)).collect(),
        )),
        Geometry::Polygon(poly) => Geometry::Polygon(
            simplify_polygon(poly, tolerance)
                .unwrap_or_else(|| Polygon::new(LineString::new(vec![]), vec![])),
        ),
        Geometry::MultiPolygon(mp) => Geometry::MultiPolygon(MultiPolygon::new(
            mp.iter().filter_map(|p| simplify_polygon(p, tolerance)).collect(),
        )),
        Geometry::Rect(r) => simplify_geom(&Geometry::Polygon(r.to_polygon()), tolerance),
        Geometry::Triangle(t) => simplify_geom(&Geometry::Polygon(t.to_polygon()), tolerance),
        Geometry::GeometryCollection(gc) => Geometry::GeometryCollection(
            gc.iter().map(|m| simplify_geom(m, tolerance)).collect(),
        ),
        // Point, MultiPoint, Line pass through unchanged
        other => other.clone(),
    }
}

/// Zero-field unit struct; `tolerance` is read from the trailing positional
/// argument at call time (not stored in the struct).
///
/// This is required because the Python `_call_builtin_scalar_fn` path resolves
/// the registered instance by name and cannot propagate struct-field parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StSimplify;

#[typetag::serde]
impl ScalarUDF for StSimplify {
    fn name(&self) -> &'static str {
        "st_simplify"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        let tolerance = read_f64_arg(&inputs, 1, "tolerance", self.name())?;
        unary_geom_to_geom(inputs.required(0)?, self.name(), |g| {
            apply_simplify(g, tolerance)
        })
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        validate_geometry_field(&inputs, schema, 0, "geom", self.name())?;
        let _tolerance = read_f64_arg_expr(&inputs, 1, "tolerance", self.name())?;
        Ok(Field::new(self.name(), DataType::Geometry))
    }

    fn docstring(&self) -> &'static str {
        "Simplifies a geometry using the Ramer–Douglas–Peucker algorithm with the given tolerance. \
         Applies to lines, polygons and collections of them; points are returned unchanged. \
         As in PostGIS ST_Simplify, polygon rings that collapse are dropped, and the result \
         is null when the whole geometry collapses."
    }
}

#[must_use]
pub fn st_simplify(geom: ExprRef, tolerance: f64) -> ExprRef {
    ScalarFn::builtin(StSimplify, vec![geom, daft_dsl::lit(tolerance)]).into()
}
