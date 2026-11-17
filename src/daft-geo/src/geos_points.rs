//! Centroid and interior point computed exactly as GEOS computes them.
//!
//! Ports of GEOS 3.14 `algorithm::Centroid` and `Geometry::getInteriorPoint`
//! (`InteriorPointPoint`, `InteriorPointLine`, `InteriorPointArea`), which
//! back PostGIS `ST_Centroid` and `ST_PointOnSurface`. The arithmetic follows
//! the C++ operation for operation (GEOS builds with `-ffp-contract=off`, and
//! Rust does not fuse multiply-adds), so results match PostGIS bit for bit and
//! hash identically.

use geo::{
    Coord, Geometry, HasDimensions, LineString, Polygon,
    kernels::{Kernel, Orientation},
};

/// `CoordinateXY::distance`.
fn distance(a: Coord, b: Coord) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn avg(a: f64, b: f64) -> f64 {
    (a + b) / 2.0
}

/// GEOS `Orientation::isCCW` for a closed ring (first point repeated last).
fn is_ccw(ring: &[Coord]) -> bool {
    // Number of points without the closing endpoint.
    let Some(n_pts) = ring.len().checked_sub(1) else {
        return false;
    };
    if n_pts < 3 {
        return false;
    }
    // First highest point after a lower point (a rising segment), if any.
    let mut up_hi = ring[0];
    let mut up_low = Coord {
        x: f64::NAN,
        y: f64::NAN,
    };
    let mut prev_y = up_hi.y;
    let mut i_up_hi = 0;
    for (i, p) in ring.iter().enumerate().take(n_pts + 1).skip(1) {
        if p.y > prev_y && p.y >= up_hi.y {
            i_up_hi = i;
            up_hi = *p;
            up_low = ring[i - 1];
        }
        prev_y = p.y;
    }
    // A flat ring.
    if i_up_hi == 0 {
        return false;
    }
    // The next lower point after the high point (a falling segment).
    let mut i_down_low = i_up_hi;
    loop {
        i_down_low = (i_down_low + 1) % n_pts;
        if i_down_low == i_up_hi || ring[i_down_low].y != up_hi.y {
            break;
        }
    }
    let down_low = ring[i_down_low];
    let i_down_hi = if i_down_low > 0 {
        i_down_low - 1
    } else {
        n_pts - 1
    };
    let down_hi = ring[i_down_hi];
    if up_hi == down_hi {
        // A cap A-B-A: fewer than 3 distinct points, or coincident segments.
        if up_low == up_hi || down_low == up_hi || up_low == down_low {
            return false;
        }
        matches!(
            <f64 as geo::GeoNum>::Ker::orient2d(up_low, up_hi, down_low),
            Orientation::CounterClockwise
        )
    } else {
        // A flat cap: the direction of the flat top gives the orientation.
        down_hi.x - up_hi.x < 0.0
    }
}

/// A point that GEOS would treat as empty (WKB `POINT EMPTY` reads as NaN).
fn is_empty_point(c: Coord) -> bool {
    c.x.is_nan() || c.y.is_nan()
}

/// GEOS `algorithm::Centroid`: area-weighted, else length-weighted, else
/// the mean of the points, over every component of the geometry.
#[derive(Default)]
struct Centroid {
    area_base: Option<Coord>,
    cg3: Coord,
    areasum2: f64,
    line_cent_sum: Coord,
    total_length: f64,
    pt_count: u32,
    pt_cent_sum: Coord,
}

impl Centroid {
    fn of(g: &Geometry) -> Option<Coord> {
        let mut c = Self::default();
        c.add(g);
        c.result()
    }

    fn result(&self) -> Option<Coord> {
        if self.areasum2.abs() > 0.0 {
            Some(Coord {
                x: self.cg3.x / 3.0 / self.areasum2,
                y: self.cg3.y / 3.0 / self.areasum2,
            })
        } else if self.total_length > 0.0 {
            Some(Coord {
                x: self.line_cent_sum.x / self.total_length,
                y: self.line_cent_sum.y / self.total_length,
            })
        } else if self.pt_count > 0 {
            let n = f64::from(self.pt_count);
            Some(Coord {
                x: self.pt_cent_sum.x / n,
                y: self.pt_cent_sum.y / n,
            })
        } else {
            None
        }
    }

    fn add(&mut self, g: &Geometry) {
        if g.is_empty() {
            return;
        }
        match g {
            Geometry::Point(p) => {
                if !is_empty_point(p.0) {
                    self.add_point(p.0);
                }
            }
            Geometry::Line(l) => self.add_line_segments(&[l.start, l.end]),
            Geometry::LineString(ls) => self.add_line_segments(&ls.0),
            Geometry::Polygon(p) => self.add_polygon(p),
            Geometry::Rect(r) => self.add_polygon(&r.to_polygon()),
            Geometry::Triangle(t) => self.add_polygon(&t.to_polygon()),
            Geometry::MultiPoint(mp) => mp.iter().for_each(|p| self.add(&Geometry::Point(*p))),
            Geometry::MultiLineString(m) => m.iter().for_each(|ls| {
                if !ls.0.is_empty() {
                    self.add_line_segments(&ls.0);
                }
            }),
            Geometry::MultiPolygon(m) => m.iter().for_each(|p| {
                if !p.exterior().0.is_empty() {
                    self.add_polygon(p);
                }
            }),
            Geometry::GeometryCollection(gc) => gc.iter().for_each(|m| self.add(m)),
        }
    }

    fn add_polygon(&mut self, p: &Polygon) {
        self.add_shell(&p.exterior().0);
        for hole in p.interiors() {
            self.add_hole(&hole.0);
        }
    }

    fn add_shell(&mut self, pts: &[Coord]) {
        let Some(&first) = pts.first() else {
            return;
        };
        self.area_base = Some(first);
        let positive = !is_ccw(pts);
        for w in pts.windows(2) {
            self.add_triangle(first, w[0], w[1], positive);
        }
        self.add_line_segments(pts);
    }

    fn add_hole(&mut self, pts: &[Coord]) {
        if pts.is_empty() {
            return;
        }
        let Some(base) = self.area_base else {
            return;
        };
        let positive = is_ccw(pts);
        for w in pts.windows(2) {
            self.add_triangle(base, w[0], w[1], positive);
        }
        self.add_line_segments(pts);
    }

    fn add_triangle(&mut self, p0: Coord, p1: Coord, p2: Coord, positive: bool) {
        let sign = if positive { 1.0 } else { -1.0 };
        // centroid3: three times the triangle's centroid.
        let c3 = Coord {
            x: p0.x + p1.x + p2.x,
            y: p0.y + p1.y + p2.y,
        };
        let a2 = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
        self.cg3.x += sign * a2 * c3.x;
        self.cg3.y += sign * a2 * c3.y;
        self.areasum2 += sign * a2;
    }

    fn add_line_segments(&mut self, pts: &[Coord]) {
        let mut line_len = 0.0;
        for w in pts.windows(2) {
            let seg_len = distance(w[0], w[1]);
            if seg_len == 0.0 {
                continue;
            }
            line_len += seg_len;
            let midx = (w[0].x + w[1].x) / 2.0;
            self.line_cent_sum.x += seg_len * midx;
            let midy = (w[0].y + w[1].y) / 2.0;
            self.line_cent_sum.y += seg_len * midy;
        }
        self.total_length += line_len;
        if line_len == 0.0
            && let Some(&first) = pts.first()
        {
            self.add_point(first);
        }
    }

    fn add_point(&mut self, p: Coord) {
        self.pt_count += 1;
        self.pt_cent_sum.x += p.x;
        self.pt_cent_sum.y += p.y;
    }
}

/// GEOS `Geometry::getCentroid` (PostGIS `ST_Centroid`); `None` when empty.
pub(crate) fn centroid(g: &Geometry) -> Option<Coord> {
    Centroid::of(g)
}

/// GEOS `Geometry::getDimension`: 0, 1 or 2; a collection's is its largest
/// member's, and an empty collection's is `None`.
fn dimension(g: &Geometry) -> Option<u8> {
    match g {
        Geometry::Point(_) | Geometry::MultiPoint(_) => Some(0),
        Geometry::Line(_) | Geometry::LineString(_) | Geometry::MultiLineString(_) => Some(1),
        Geometry::Polygon(_)
        | Geometry::MultiPolygon(_)
        | Geometry::Rect(_)
        | Geometry::Triangle(_) => Some(2),
        Geometry::GeometryCollection(gc) => gc.iter().filter_map(dimension).max(),
    }
}

/// GEOS `Geometry::getInteriorPoint` (PostGIS `ST_PointOnSurface`): a point
/// on the geometry, chosen by its highest-dimension parts. `None` when empty.
pub(crate) fn interior_point(g: &Geometry) -> Option<Coord> {
    match dimension(g) {
        Some(0) => interior_point_point(g),
        Some(1) => interior_point_line(g),
        _ => interior_point_area(g),
    }
}

/// Every non-empty point of `g`, in order.
fn for_each_point(g: &Geometry, f: &mut impl FnMut(Coord)) {
    match g {
        Geometry::Point(p) if !is_empty_point(p.0) => f(p.0),
        Geometry::MultiPoint(mp) => mp
            .iter()
            .filter(|p| !is_empty_point(p.0))
            .for_each(|p| f(p.0)),
        Geometry::GeometryCollection(gc) => gc.iter().for_each(|m| for_each_point(m, f)),
        _ => {}
    }
}

/// Every linestring of `g` (a `Line` as a two-point linestring), in order.
fn for_each_line(g: &Geometry, f: &mut impl FnMut(&[Coord])) {
    match g {
        Geometry::Line(l) => f(&[l.start, l.end]),
        Geometry::LineString(ls) => f(&ls.0),
        Geometry::MultiLineString(m) => m.iter().for_each(|ls: &LineString| f(&ls.0)),
        Geometry::GeometryCollection(gc) => gc.iter().for_each(|m| for_each_line(m, f)),
        _ => {}
    }
}

/// GEOS `InteriorPointPoint`: the point nearest the centroid (first on ties).
fn interior_point_point(g: &Geometry) -> Option<Coord> {
    let centroid = centroid(g)?;
    let mut best: Option<(f64, Coord)> = None;
    for_each_point(g, &mut |p| {
        let d = distance(p, centroid);
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, p));
        }
    });
    best.map(|(_, p)| p)
}

/// GEOS `InteriorPointLine`: the interior vertex nearest the centroid, else the
/// nearest endpoint.
fn interior_point_line(g: &Geometry) -> Option<Coord> {
    let mut best: Option<(f64, Coord)> = None;
    let add = |best: &mut Option<(f64, Coord)>, p: Coord, centroid: Coord| {
        let d = distance(p, centroid);
        if best.is_none_or(|(bd, _)| d < bd) {
            *best = Some((d, p));
        }
    };
    if let Some(centroid) = centroid(g) {
        for_each_line(g, &mut |pts| {
            if pts.len() >= 3 {
                for &p in &pts[1..pts.len() - 1] {
                    add(&mut best, p, centroid);
                }
            }
        });
        if best.is_none() {
            for_each_line(g, &mut |pts| {
                if let Some(&first) = pts.first() {
                    add(&mut best, first, centroid);
                    if pts.len() > 1 {
                        add(&mut best, pts[pts.len() - 1], centroid);
                    }
                }
            });
        }
    }
    // GEOS's `minDistance` starts at infinity and only an uninitialised
    // (NaN) centroid could skip both loops; then there is no point.
    best.map(|(_, p)| p)
}

/// GEOS `InteriorPointArea`: per polygon, the midpoint of the widest interior
/// section of a horizontal scan line that avoids every vertex's y; the widest
/// over all polygons wins (the first on ties).
fn interior_point_area(g: &Geometry) -> Option<Coord> {
    let mut best: Option<(f64, Coord)> = None;
    for_each_polygon(g, &mut |poly| {
        let (width, point) = interior_point_polygon(poly);
        if best.is_none_or(|(bw, _)| width > bw) {
            best = Some((width, point));
        }
    });
    best.map(|(_, p)| p)
}

/// Every non-empty polygon of `g`, in order.
fn for_each_polygon(g: &Geometry, f: &mut impl FnMut(&Polygon)) {
    match g {
        Geometry::Polygon(p) if !p.exterior().0.is_empty() => f(p),
        Geometry::Rect(r) => f(&r.to_polygon()),
        Geometry::Triangle(t) => f(&t.to_polygon()),
        Geometry::MultiPolygon(m) => m
            .iter()
            .filter(|p| !p.exterior().0.is_empty())
            .for_each(|p| f(p)),
        Geometry::GeometryCollection(gc) => gc.iter().for_each(|m| for_each_polygon(m, f)),
        _ => {}
    }
}

/// `(min_y, max_y)` of a coordinate run.
fn y_range(pts: &[Coord]) -> (f64, f64) {
    pts.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| {
            (lo.min(c.y), hi.max(c.y))
        })
}

/// GEOS `ScanLineYOrdinateFinder::getScanLineY`: midway between the nearest
/// vertex y-values below (or at) and above the centre of the y-extent.
fn scan_line_y(poly: &Polygon) -> f64 {
    let (mut lo_y, mut hi_y) = y_range(&poly.exterior().0);
    let centre_y = avg(lo_y, hi_y);
    let rings = std::iter::once(poly.exterior()).chain(poly.interiors());
    for c in rings.flat_map(|r| r.0.iter()) {
        let y = c.y;
        if y <= centre_y {
            if y > lo_y {
                lo_y = y;
            }
        } else if y < hi_y {
            hi_y = y;
        }
    }
    avg(hi_y, lo_y)
}

/// GEOS `InteriorPointPolygon`: `(width of the widest section, its midpoint)`;
/// the first shell vertex with width 0 if the scan line finds no section.
fn interior_point_polygon(poly: &Polygon) -> (f64, Coord) {
    let scan_y = scan_line_y(poly);
    let mut crossings: Vec<f64> = vec![];
    for ring in std::iter::once(poly.exterior()).chain(poly.interiors()) {
        let (min_y, max_y) = y_range(&ring.0);
        if scan_y < min_y || scan_y > max_y {
            continue;
        }
        for w in ring.0.windows(2) {
            let (p0, p1) = (w[0], w[1]);
            // intersectsHorizontalLine
            if (p0.y > scan_y && p1.y > scan_y) || (p0.y < scan_y && p1.y < scan_y) {
                continue;
            }
            // isEdgeCrossingCounted
            if p0.y == p1.y
                || (p0.y == scan_y && p1.y < scan_y)
                || (p1.y == scan_y && p0.y < scan_y)
            {
                continue;
            }
            // intersection
            let x = if p0.x == p1.x {
                p0.x
            } else {
                let m = (p1.y - p0.y) / (p1.x - p0.x);
                p0.x + ((scan_y - p0.y) / m)
            };
            crossings.push(x);
        }
    }
    let mut width = 0.0;
    let mut point = poly.exterior().0[0];
    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for pair in crossings.chunks_exact(2) {
        let w = pair[1] - pair[0];
        if w > width {
            width = w;
            point = Coord {
                x: avg(pair[0], pair[1]),
                y: scan_y,
            };
        }
    }
    (width, point)
}

#[cfg(test)]
mod tests {
    use wkt::TryFromWkt;

    use super::*;

    fn g(w: &str) -> Geometry {
        Geometry::try_from_wkt_str(w).unwrap()
    }

    fn bits(c: Coord) -> (u64, u64) {
        (c.x.to_bits(), c.y.to_bits())
    }

    /// Expected values are PostGIS 3.5.4 / GEOS 3.14.1 output.
    #[test]
    fn centroid_matches_geos() {
        let l_shape = g(
            "POLYGON((-77.007 38.9006,-77.006 38.9006,-77.006 38.9008,-77.0065 38.9008,-77.0065 38.9016,-77.007 38.9016,-77.007 38.9006))",
        );
        assert_eq!(
            bits(centroid(&l_shape).unwrap()),
            bits(Coord {
                x: -77.00666666666666,
                y: 38.90103333333333
            })
        );
        assert_eq!(centroid(&g("GEOMETRYCOLLECTION EMPTY")), None);
    }

    #[test]
    fn interior_point_matches_geos() {
        let l_shape = g(
            "POLYGON((-77.007 38.9006,-77.006 38.9006,-77.006 38.9008,-77.0065 38.9008,-77.0065 38.9016,-77.007 38.9016,-77.007 38.9006))",
        );
        assert_eq!(
            bits(interior_point(&l_shape).unwrap()),
            bits(Coord {
                x: -77.00675000000001,
                y: 38.9012
            })
        );
        // Lines: the interior vertex nearest the centroid.
        assert_eq!(
            interior_point(&g("LINESTRING(0 0,1 0,2 0,10 0)")),
            Some(Coord { x: 2.0, y: 0.0 })
        );
        // Points: the one nearest the centroid.
        assert_eq!(
            interior_point(&g("MULTIPOINT((0 0),(1 1),(10 10))")),
            Some(Coord { x: 1.0, y: 1.0 })
        );
        assert_eq!(interior_point(&g("GEOMETRYCOLLECTION EMPTY")), None);
    }

    #[test]
    fn is_ccw_matches_orientation() {
        let ring = |w: &str| match g(w) {
            Geometry::Polygon(p) => p.exterior().0.clone(),
            _ => unreachable!(),
        };
        assert!(is_ccw(&ring("POLYGON((0 0,1 0,1 1,0 1,0 0))")));
        assert!(!is_ccw(&ring("POLYGON((0 0,0 1,1 1,1 0,0 0))")));
        // Flat top: direction of the flat segment decides.
        assert!(is_ccw(&ring("POLYGON((0 0,2 0,2 1,1 1,0 1,0 0))")));
    }
}
