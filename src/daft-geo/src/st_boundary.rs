use common_error::DaftResult;
use daft_core::{
    prelude::{DataType, Field, Schema},
    series::Series,
};
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};
use geo::{Geometry, LineString, MultiLineString, Polygon};
use serde::{Deserialize, Serialize};

use crate::utils::{unary_geom_to_geom, validate_geometry_field};

/// A ring usable as a boundary line: closed, at least 4 positions, all finite.
fn usable_ring(ring: &LineString) -> bool {
    ring.0.len() >= 4
        && ring.is_closed()
        && ring.0.iter().all(|c| c.x.is_finite() && c.y.is_finite())
}

/// Append every ring of `p` (exterior first, then holes) as a line, or return
/// `false` if any ring is unusable.
fn push_rings(p: &Polygon, out: &mut Vec<LineString>) -> bool {
    std::iter::once(p.exterior())
        .chain(p.interiors())
        .all(|ring| {
            let ok = usable_ring(ring);
            if ok {
                out.push(ring.clone());
            }
            ok
        })
}

/// Rings of a Polygon or MultiPolygon as a MultiLineString: each polygon's
/// exterior followed by its holes, in input order.
///
/// Null for non-polygonal input, empty input, and polygons with a ring that is
/// open, has fewer than 4 positions, or has a non-finite coordinate.
/// Self-intersection is not checked (as in PostGIS `ST_Boundary`): an OGC
/// validity check is super-linear in the vertex count, and the rings of a
/// self-intersecting polygon are still its edges.
fn apply_boundary(g: &Geometry) -> Option<Geometry> {
    let polys: &[Polygon] = match g {
        Geometry::Polygon(p) => std::slice::from_ref(p),
        Geometry::MultiPolygon(mp) => &mp.0,
        _ => return None,
    };
    let mut lines = Vec::with_capacity(polys.iter().map(|p| 1 + p.interiors().len()).sum());
    for p in polys {
        if p.exterior().0.is_empty() && p.interiors().is_empty() {
            continue;
        }
        if !push_rings(p, &mut lines) {
            return None;
        }
    }
    (!lines.is_empty()).then(|| Geometry::MultiLineString(MultiLineString::new(lines)))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StBoundary;

#[typetag::serde]
impl ScalarUDF for StBoundary {
    fn name(&self) -> &'static str {
        "st_boundary"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        unary_geom_to_geom(inputs.required(0)?, self.name(), apply_boundary)
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
        "Returns the rings of a Polygon or MultiPolygon as a MultiLineString: each polygon's \
         exterior ring followed by its interior rings. Null for null, unparseable, empty or \
         non-polygonal input, and for polygons with an open, too-short or non-finite ring."
    }
}

#[must_use]
pub fn st_boundary(geom: ExprRef) -> ExprRef {
    ScalarFn::builtin(StBoundary, vec![geom]).into()
}

#[cfg(test)]
mod tests {
    use geo::{Coord, coord, polygon, wkt};

    use super::*;

    fn lines(g: Option<Geometry>) -> Vec<Vec<(f64, f64)>> {
        let Some(Geometry::MultiLineString(m)) = g else {
            panic!("expected a MultiLineString, got {g:?}")
        };
        m.0.into_iter()
            .map(|ls| ls.0.into_iter().map(|c| (c.x, c.y)).collect())
            .collect()
    }

    #[test]
    fn polygon_with_hole_gives_exterior_then_hole() {
        let g: Geometry = wkt! {
            POLYGON((0. 0.,10. 0.,10. 10.,0. 10.,0. 0.),(2. 2.,4. 2.,4. 4.,2. 2.))
        }
        .into();
        assert_eq!(
            lines(apply_boundary(&g)),
            vec![
                vec![(0., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.)],
                vec![(2., 2.), (4., 2.), (4., 4.), (2., 2.)],
            ]
        );
    }

    #[test]
    fn multipolygon_gives_every_ring_in_order() {
        let g: Geometry = wkt! {
            MULTIPOLYGON(
                ((0. 0.,1. 0.,1. 1.,0. 0.)),
                ((5. 5.,9. 5.,9. 9.,5. 5.),(6. 6.,7. 6.,7. 7.,6. 6.))
            )
        }
        .into();
        let got = lines(apply_boundary(&g));
        assert_eq!(got.len(), 3);
        assert_eq!(got[0][0], (0., 0.));
        assert_eq!(got[1][0], (5., 5.));
        assert_eq!(got[2][0], (6., 6.));
    }

    #[test]
    fn non_polygonal_empty_and_malformed_input_give_none() {
        let point: Geometry = wkt! { POINT(1. 1.) }.into();
        let line: Geometry = wkt! { LINESTRING(0. 0.,1. 1.) }.into();
        let empty = Geometry::Polygon(Polygon::new(LineString::new(vec![]), vec![]));
        let nan = Geometry::Polygon(polygon![
            (x: 0., y: 0.), (x: f64::NAN, y: 0.), (x: 1., y: 1.),
        ]);
        // geo closes rings on construction; build a too-short one directly.
        let short = Geometry::Polygon(Polygon::new(
            LineString::new(vec![coord! { x: 0., y: 0. }, Coord { x: 1., y: 0. }]),
            vec![],
        ));
        for g in [point, line, empty, nan, short] {
            assert!(apply_boundary(&g).is_none(), "{g:?}");
        }
    }
}
