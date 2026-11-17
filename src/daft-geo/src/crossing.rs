//! Segment crossings computed exactly as GEOS computes them.
//!
//! GEOS (3.12+) computes the point where two segments properly cross with
//! double-double ("DD") arithmetic in `CGAlgorithmsDD::intersection`, called
//! from `LineIntersector::computeIntersect`. GEOS builds with
//! `-ffp-contract=off`, and Rust does not fuse multiply-adds either, so this
//! port reproduces GEOS's result bit for bit. The DD operations are symmetric
//! in their operands, so the result does not depend on which segment comes
//! first or on the direction of either segment.

use geo::{
    Coord, Line,
    kernels::{Kernel, Orientation},
};

/// A double-double number (`geos::math::DD`): `hi + lo` with `|lo| <= ulp(hi)/2`.
#[derive(Clone, Copy)]
struct DD {
    hi: f64,
    lo: f64,
}

/// `2^27 + 1`, Dekker's splitter for IEEE doubles.
const SPLIT: f64 = 134_217_729.0;

impl DD {
    fn new(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// `DD::selfAdd(double yhi, double ylo)`.
    fn add(self, y: Self) -> Self {
        let (hi, lo, yhi, ylo) = (self.hi, self.lo, y.hi, y.lo);
        let big_s = hi + yhi;
        let big_t = lo + ylo;
        let mut e = big_s - hi;
        let f = big_t - lo;
        let mut s = big_s - e;
        let mut t = big_t - f;
        s = (yhi - e) + (hi - s);
        t = (ylo - f) + (lo - t);
        e = s + big_t;
        let big_h = big_s + e;
        let h = e + (big_s - big_h);
        e = t + h;
        let zhi = big_h + e;
        let zlo = e + (big_h - zhi);
        Self { hi: zhi, lo: zlo }
    }

    /// `DD::selfSubtract(const DD&)`.
    fn sub(self, y: Self) -> Self {
        self.add(Self {
            hi: -y.hi,
            lo: -y.lo,
        })
    }

    /// `DD::selfMultiply(double yhi, double ylo)`.
    fn mul(self, y: Self) -> Self {
        let (hi, lo, yhi, ylo) = (self.hi, self.lo, y.hi, y.lo);
        let mut big_c = SPLIT * hi;
        let mut hx = big_c - hi;
        let mut c = SPLIT * yhi;
        hx = big_c - hx;
        let tx = hi - hx;
        let mut hy = c - yhi;
        big_c = hi * yhi;
        hy = c - hy;
        let ty = yhi - hy;
        c = ((((hx * hy - big_c) + hx * ty) + tx * hy) + tx * ty) + (hi * ylo + lo * yhi);
        let zhi = big_c + c;
        hx = big_c - zhi;
        let zlo = c + hx;
        Self { hi: zhi, lo: zlo }
    }

    /// `DD::selfDivide(double yhi, double ylo)`.
    fn div(self, y: Self) -> Self {
        let (hi, lo, yhi, ylo) = (self.hi, self.lo, y.hi, y.lo);
        let big_c = hi / yhi;
        let mut c = SPLIT * big_c;
        let mut hc = c - big_c;
        let mut u = SPLIT * yhi;
        hc = c - hc;
        let tc = big_c - hc;
        let mut hy = u - yhi;
        let big_u = big_c * yhi;
        hy = u - hy;
        let ty = yhi - hy;
        u = (((hc * hy - big_u) + hc * ty) + tc * hy) + tc * ty;
        c = ((((hi - big_u) - u) + lo) - big_c * ylo) / yhi;
        u = big_c + c;
        Self {
            hi: u,
            lo: (big_c - u) + c,
        }
    }
}

/// `CGAlgorithmsDD::intersection`: the intersection of the lines through
/// `p1-p2` and `q1-q2`, or `None` when it is not finite (parallel lines).
fn dd_intersection(p1: Coord, p2: Coord, q1: Coord, q2: Coord) -> Option<Coord> {
    let (q1x, q1y, q2x, q2y) = (DD::new(q1.x), DD::new(q1.y), DD::new(q2.x), DD::new(q2.y));
    let (p1x, p1y, p2x, p2y) = (DD::new(p1.x), DD::new(p1.y), DD::new(p2.x), DD::new(p2.y));

    let px = p1y.sub(p2y);
    let py = p2x.sub(p1x);
    let pw = p1x.mul(p2y).sub(p2x.mul(p1y));

    let qx = q1y.sub(q2y);
    let qy = q2x.sub(q1x);
    let qw = q1x.mul(q2y).sub(q2x.mul(q1y));

    let x = py.mul(qw).sub(qy.mul(pw));
    let y = qx.mul(pw).sub(px.mul(qw));
    let w = px.mul(qy).sub(qx.mul(py));

    let x_int = x.div(w).to_f64();
    let y_int = y.div(w).to_f64();
    (x_int.is_finite() && y_int.is_finite()).then_some(Coord { x: x_int, y: y_int })
}

fn in_envelope(l: Line, c: Coord) -> bool {
    c.x >= l.start.x.min(l.end.x)
        && c.x <= l.start.x.max(l.end.x)
        && c.y >= l.start.y.min(l.end.y)
        && c.y <= l.start.y.max(l.end.y)
}

/// The point where `p` and `q` *properly* cross (each segment's interior
/// meets the other's, at a single point that is not an endpoint), computed as
/// GEOS's `LineIntersector` computes it. Returns `None` for disjoint,
/// collinear, and endpoint-touching segments; in those cases GEOS copies an
/// input vertex, so there is no new coordinate to compute.
pub(crate) fn proper_crossing(p: Line, q: Line) -> Option<Coord> {
    let (p1, p2, q1, q2) = (p.start, p.end, q.start, q.end);
    if p1.x.max(p2.x) < q1.x.min(q2.x)
        || q1.x.max(q2.x) < p1.x.min(p2.x)
        || p1.y.max(p2.y) < q1.y.min(q2.y)
        || q1.y.max(q2.y) < p1.y.min(p2.y)
    {
        return None;
    }
    let orient = |a: Coord, b: Coord, c: Coord| <f64 as geo::GeoNum>::Ker::orient2d(a, b, c);
    let (pq1, pq2) = (orient(p1, p2, q1), orient(p1, p2, q2));
    let (qp1, qp2) = (orient(q1, q2, p1), orient(q1, q2, p2));
    let strictly_opposite = |a: Orientation, b: Orientation| {
        matches!(
            (a, b),
            (Orientation::Clockwise, Orientation::CounterClockwise)
                | (Orientation::CounterClockwise, Orientation::Clockwise)
        )
    };
    if !(strictly_opposite(pq1, pq2) && strictly_opposite(qp1, qp2)) {
        return None;
    }
    // `LineIntersector::intersection`: an out-of-envelope (round-off) or
    // non-finite result falls back to the nearest endpoint.
    match dd_intersection(p1, p2, q1, q2) {
        Some(c) if in_envelope(p, c) && in_envelope(q, c) => Some(c),
        _ => Some(nearest_endpoint(p, q)),
    }
}

/// `LineIntersector::nearestEndpoint`.
fn nearest_endpoint(p: Line, q: Line) -> Coord {
    use geo::{Distance, Euclidean, Point};
    let d = |c: Coord, l: Line| Euclidean.distance(&Point::from(c), &l);
    let mut best = (p.start, d(p.start, q));
    for (c, l) in [(p.end, q), (q.start, p), (q.end, p)] {
        let dist = d(c, l);
        if dist < best.1 {
            best = (c, dist);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(a: (f64, f64), b: (f64, f64)) -> Line {
        Line::new(Coord { x: a.0, y: a.1 }, Coord { x: b.0, y: b.1 })
    }

    #[test]
    fn axis_aligned_crossing_is_exact() {
        let h = line((-77.0073, 38.9006), (-77.0067, 38.9006));
        let v = line((-77.007, 38.9), (-77.007, 38.901));
        assert_eq!(
            proper_crossing(h, v),
            Some(Coord {
                x: -77.007,
                y: 38.9006
            })
        );
    }

    #[test]
    fn crossing_ignores_operand_order_and_direction() {
        let p = line((0.1, 0.2), (7.3, 5.9));
        let q = line((0.3, 6.1), (6.7, 0.7));
        let rev = |l: Line| Line::new(l.end, l.start);
        let want = proper_crossing(p, q).unwrap();
        for (a, b) in [(q, p), (rev(p), q), (p, rev(q)), (rev(q), rev(p))] {
            let got = proper_crossing(a, b).unwrap();
            assert_eq!(
                (got.x.to_bits(), got.y.to_bits()),
                (want.x.to_bits(), want.y.to_bits())
            );
        }
    }

    #[test]
    fn endpoint_touch_and_collinear_are_not_proper() {
        let p = line((0.0, 0.0), (2.0, 0.0));
        assert_eq!(proper_crossing(p, line((1.0, 0.0), (1.0, 1.0))), None);
        assert_eq!(proper_crossing(p, line((1.0, 0.0), (3.0, 0.0))), None);
        assert_eq!(proper_crossing(p, line((0.0, 1.0), (2.0, 1.0))), None);
    }
}
