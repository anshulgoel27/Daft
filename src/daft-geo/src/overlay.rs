//! Shared overlay engine for `st_union`, `st_intersection`, `st_difference`
//! and `st_symdifference`.
//!
//! - **Areal × areal** goes through `geo::BooleanOps` (i_overlay), followed by
//!   snapping so that output vertices which came from an input are
//!   bit-identical to it (i_overlay rounds everything to an integer grid sized
//!   by the inputs' extent).
//! - **Lines** are noded against the other input's lines (split at crossings
//!   and at the ends of shared segments), so line × line results follow GEOS:
//!   shared parts are found exactly, and results are split at the nodes.
//! - **Mixed dimensions** are combined PostGIS-style: lines are clipped by
//!   polygons, points are kept or dropped by an intersects test, and
//!   lower-dimensional parts covered by a higher-dimensional part of the result
//!   are absorbed into it.
//! - **Output type**: a single polygon is a `POLYGON`, several a `MULTIPOLYGON`
//!   (an empty result of two areal inputs is `MULTIPOLYGON EMPTY`); likewise
//!   `LINESTRING` / `MULTILINESTRING` and `POINT` / `MULTIPOINT`; mixed results
//!   are a `GEOMETRYCOLLECTION` (`GEOMETRYCOLLECTION EMPTY` when empty).
//!
//! Inputs are wrapped in [`Prepared`], so a broadcast literal (e.g. clipping
//! every row by one region) is decomposed and indexed once, not per row.
//!
//! Polygons that touch intersect in their shared boundary edges and touch
//! points (alongside any areal part), as in GEOS.

use geo::{
    BooleanOps, BoundingRect, Coord, CoordsIter, Geometry, GeometryCollection, Intersects, Line,
    LineString, MapCoords, MultiLineString, MultiPoint, MultiPolygon, Point, Rect,
    line_intersection::{LineIntersection, line_intersection},
    unary_union,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverlayOp {
    Union,
    Intersection,
    Difference,
    SymDifference,
}

/// Areal results: one part → `Polygon`, otherwise `MultiPolygon` (incl. empty).
pub(crate) fn areal_geometry(mp: MultiPolygon) -> Geometry {
    if mp.0.len() == 1 {
        Geometry::Polygon(mp.0.into_iter().next().expect("len checked"))
    } else {
        Geometry::MultiPolygon(mp)
    }
}

/// An overlay operand, decomposed and indexed once.
pub(crate) struct Prepared {
    geom: Geometry,
    parts: Parts,
    rect: Option<Rect>,
    /// The input's vertices sorted by x, for snapping output vertices back.
    coords: Vec<Coord>,
    areal: bool,
    /// R-tree of the polygon rings, for large polygonal operands only (built
    /// once per broadcast literal); enables the disjoint/contained fast path.
    rings: Option<Segs>,
}

/// Polygonal operands with more ring segments than this get a ring index.
const RING_INDEX_MIN_SEGMENTS: usize = 64;

impl Prepared {
    pub(crate) fn new(geom: Geometry) -> Self {
        let mut parts = Parts::of(&geom);
        let rect = geom.bounding_rect();
        let mut coords: Vec<Coord> = geom.coords_iter().collect();
        coords.sort_by(|a, b| a.x.total_cmp(&b.x));
        // Polygons gathered from separate GeometryCollection members may
        // overlap; the even-odd boolean ops would treat their overlap as
        // outside. Dissolve them first (unary_union uses the non-zero rule). A
        // MultiPolygon is kept as-is: an overlapping one is invalid input.
        if matches!(geom, Geometry::GeometryCollection(_)) && parts.areas.0.len() > 1 {
            let snap = Snapper::new(rect, None, &coords, &[]);
            parts.areas = snap.snap(unary_union(parts.areas.0.iter()));
        }
        let areal = is_areal_type(&geom);
        let only_areas = parts.points.is_empty() && parts.lines.is_empty();
        let rings = (only_areas && coords.len() > RING_INDEX_MIN_SEGMENTS)
            .then(|| segments(&boundary_lines(&parts.areas)));
        Self {
            geom,
            parts,
            rect,
            coords,
            areal,
            rings,
        }
    }
}

/// Run `op` on two prepared geometries. Returns `None` if the engine panics.
pub(crate) fn overlay(a: &Prepared, b: &Prepared, op: OverlayOp) -> Option<Geometry> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| overlay_inner(a, b, op))).ok()
}

/// True for polygonal geometry types (and collections made only of them).
fn is_areal_type(g: &Geometry) -> bool {
    match g {
        Geometry::Polygon(_)
        | Geometry::MultiPolygon(_)
        | Geometry::Rect(_)
        | Geometry::Triangle(_) => true,
        Geometry::GeometryCollection(gc) => !gc.0.is_empty() && gc.iter().all(is_areal_type),
        _ => false,
    }
}

struct Parts {
    points: Vec<Point>,
    lines: Vec<LineString>,
    areas: MultiPolygon,
}

impl Parts {
    fn empty() -> Self {
        Self {
            points: vec![],
            lines: vec![],
            areas: MultiPolygon::empty(),
        }
    }

    fn of(g: &Geometry) -> Self {
        let mut parts = Self::empty();
        parts.add(g);
        parts
    }

    fn add(&mut self, g: &Geometry) {
        match g {
            Geometry::Point(p) => {
                if !p.x().is_nan() && !p.y().is_nan() {
                    self.points.push(*p);
                }
            }
            Geometry::MultiPoint(mp) => self.points.extend(mp.iter().copied()),
            Geometry::Line(l) => self.lines.push(LineString::new(vec![l.start, l.end])),
            Geometry::LineString(ls) => {
                if ls.0.len() >= 2 {
                    self.lines.push(ls.clone());
                }
            }
            Geometry::MultiLineString(m) => {
                self.lines
                    .extend(m.iter().filter(|ls| ls.0.len() >= 2).cloned());
            }
            Geometry::Polygon(p) => {
                if !p.exterior().0.is_empty() {
                    self.areas.0.push(p.clone());
                }
            }
            Geometry::MultiPolygon(mp) => {
                self.areas
                    .0
                    .extend(mp.iter().filter(|p| !p.exterior().0.is_empty()).cloned());
            }
            Geometry::Rect(r) => self.areas.0.push(r.to_polygon()),
            Geometry::Triangle(t) => self.areas.0.push(t.to_polygon()),
            Geometry::GeometryCollection(gc) => gc.iter().for_each(|m| self.add(m)),
        }
    }

    fn into_geometry(mut self, areal_inputs: bool) -> Geometry {
        dedupe_points(&mut self.points);
        dedupe_lines(&mut self.lines);
        let mut members: Vec<Geometry> = Vec::new();
        if !self.areas.0.is_empty() {
            members.push(areal_geometry(self.areas));
        }
        match self.lines.len() {
            0 => {}
            1 => members.push(Geometry::LineString(self.lines.pop().expect("len checked"))),
            _ => members.push(Geometry::MultiLineString(MultiLineString::new(self.lines))),
        }
        match self.points.len() {
            0 => {}
            1 => members.push(Geometry::Point(self.points[0])),
            _ => members.push(Geometry::MultiPoint(MultiPoint::new(self.points))),
        }
        match members.len() {
            0 if areal_inputs => Geometry::MultiPolygon(MultiPolygon::empty()),
            1 => members.pop().expect("len checked"),
            _ => Geometry::GeometryCollection(GeometryCollection::new_from(members)),
        }
    }
}

/// Where an operand lies relative to a large polygonal one whose rings do not
/// come near it: entirely inside, or entirely outside.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    Inside,
    Outside,
}

/// If `big`'s rings do not meet `small`'s bbox, `small` (polygons only) lies in
/// one face of `big`; one point-in-polygon test says which.
fn placement(big: &Prepared, small: &Prepared) -> Option<Placement> {
    use geo::coordinate_position::{CoordPos, CoordinatePosition};
    let rings = big.rings.as_ref()?;
    let rect = small.rect?;
    if !small.parts.points.is_empty()
        || !small.parts.lines.is_empty()
        || small.parts.areas.0.is_empty()
    {
        return None;
    }
    if rings.near(rect).next().is_some() {
        return None;
    }
    let probe = *small.parts.areas.0[0].exterior().0.first()?;
    match big.parts.areas.coordinate_position(&probe) {
        CoordPos::Inside => Some(Placement::Inside),
        CoordPos::Outside => Some(Placement::Outside),
        CoordPos::OnBoundary => None,
    }
}

/// Overlay of two polygon sets when one lies wholly inside or outside the
/// other: no boolean op is needed, and every vertex stays exact. Returns `None`
/// when the case needs a real overlay (e.g. cutting a hole).
fn overlay_fast_path(a: &Prepared, b: &Prepared, op: OverlayOp) -> Option<Geometry> {
    if !(a.areal && b.areal) {
        return None;
    }
    let both = |x: &MultiPolygon, y: &MultiPolygon| {
        MultiPolygon::new(x.0.iter().chain(&y.0).cloned().collect())
    };
    let (ma, mb) = (&a.parts.areas, &b.parts.areas);
    let empty = MultiPolygon::empty;
    let out = if let Some(place) = placement(a, b) {
        // b is inside or outside a.
        match (op, place) {
            (OverlayOp::Intersection, Placement::Inside) => mb.clone(),
            (OverlayOp::Intersection, Placement::Outside) => empty(),
            (OverlayOp::Union, Placement::Inside) => ma.clone(),
            (OverlayOp::Union | OverlayOp::SymDifference, Placement::Outside) => both(ma, mb),
            (OverlayOp::Difference, Placement::Outside) => ma.clone(),
            _ => return None,
        }
    } else {
        // a is inside or outside b.
        let place = placement(b, a)?;
        match (op, place) {
            (OverlayOp::Intersection, Placement::Inside) => ma.clone(),
            (OverlayOp::Intersection, Placement::Outside) => empty(),
            (OverlayOp::Difference, Placement::Inside) => empty(),
            (OverlayOp::Difference, Placement::Outside) => ma.clone(),
            (OverlayOp::Union, Placement::Inside) => mb.clone(),
            (OverlayOp::Union | OverlayOp::SymDifference, Placement::Outside) => both(ma, mb),
            _ => return None,
        }
    };
    Some(areal_geometry(out))
}

fn overlay_inner(a: &Prepared, b: &Prepared, op: OverlayOp) -> Geometry {
    if let Some(fast) = overlay_fast_path(a, b, op) {
        return fast;
    }
    let (pa, pb) = (&a.parts, &b.parts);
    let snap = Snapper::new(a.rect, b.rect, &a.coords, &b.coords);
    let areal_inputs = a.areal && b.areal;

    let bool_op = |x: &MultiPolygon, y: &MultiPolygon| -> MultiPolygon {
        let raw = match op {
            OverlayOp::Union => x.union(y),
            OverlayOp::Intersection => x.intersection(y),
            OverlayOp::Difference => x.difference(y),
            OverlayOp::SymDifference => x.xor(y),
        };
        snap.snap_areal(raw)
    };
    // Parts of `lines` outside (`invert`) or inside `areas`.
    let clip = |lines: Vec<LineString>, areas: &MultiPolygon, invert: bool| -> Vec<LineString> {
        if lines.is_empty() || areas.0.is_empty() {
            return if invert { lines } else { vec![] };
        }
        let clipped = areas.clip(&MultiLineString::new(lines), invert);
        let clipped: Vec<LineString> = snap
            .snap(clipped)
            .0
            .into_iter()
            .filter(|ls| ls.0.len() >= 2)
            .collect();
        // Split where the lines touch the polygon boundary, as GEOS does.
        chain(&node(&clipped, &segments(&boundary_lines(areas))), |_| true)
    };
    let (sa, sb) = (segments(&pa.lines), segments(&pb.lines));
    let (noded_a, noded_b) = (node(&pa.lines, &sb), node(&pb.lines, &sa));

    let out = match op {
        OverlayOp::Intersection => {
            let areas = if pa.areas.0.is_empty() || pb.areas.0.is_empty() {
                MultiPolygon::empty()
            } else {
                bool_op(&pa.areas, &pb.areas)
            };
            // Shared parts of the two line sets, then their isolated crossings.
            let mut lines = chain(&noded_a, |piece| piece.shared);
            let mut points = crossing_points(&sa.lines, &sb);
            for (l, other) in [(&pa.lines, &pb.areas), (&pb.lines, &pa.areas)] {
                lines.extend(clip(l.clone(), other, false));
                // A line touching a polygon only at isolated points: those points.
                points.extend(crossing_points(
                    &segment_list(l),
                    &segments(&boundary_lines(other)),
                ));
            }
            points.extend(pa.points.iter().filter(|p| b.geom.intersects(*p)).copied());
            points.extend(pb.points.iter().filter(|p| a.geom.intersects(*p)).copied());
            // Polygons that touch: their shared boundary edges and touch points, as
            // GEOS returns them. Parts on the areal result's boundary are absorbed below.
            if !pa.areas.0.is_empty() && !pb.areas.0.is_empty() {
                let (touch_lines, touch_points) = boundary_touches(&pa.areas, &pb.areas);
                lines.extend(touch_lines);
                points.extend(touch_points);
            }
            // Lower-dimensional parts covered by a higher-dimensional part are absorbed.
            absorb(points, lines, areas, &clip, snap.tolerance)
        }
        OverlayOp::Difference => {
            let areas = if pb.areas.0.is_empty() || pa.areas.0.is_empty() {
                pa.areas.clone()
            } else {
                bool_op(&pa.areas, &pb.areas)
            };
            let lines = chain(&noded_a, |piece| !piece.shared);
            let lines = clip(lines, &pb.areas, true);
            let points = pa
                .points
                .iter()
                .filter(|p| !b.geom.intersects(*p))
                .copied()
                .collect();
            Parts {
                points,
                lines,
                areas,
            }
        }
        OverlayOp::Union => {
            let areas = match (pa.areas.0.is_empty(), pb.areas.0.is_empty()) {
                (true, _) => pb.areas.clone(),
                (_, true) => pa.areas.clone(),
                _ => bool_op(&pa.areas, &pb.areas),
            };
            let mut lines = chain(&noded_a, |_| true);
            lines.extend(chain(&noded_b, |piece| !piece.shared));
            let points = pa.points.iter().chain(&pb.points).copied().collect();
            absorb(points, lines, areas, &clip, snap.tolerance)
        }
        OverlayOp::SymDifference => {
            let areas = match (pa.areas.0.is_empty(), pb.areas.0.is_empty()) {
                (true, _) => pb.areas.clone(),
                (_, true) => pa.areas.clone(),
                _ => bool_op(&pa.areas, &pb.areas),
            };
            let mut lines = clip(chain(&noded_a, |piece| !piece.shared), &pb.areas, true);
            lines.extend(clip(
                chain(&noded_b, |piece| !piece.shared),
                &pa.areas,
                true,
            ));
            let mut points: Vec<Point> = pa
                .points
                .iter()
                .filter(|p| !b.geom.intersects(*p))
                .copied()
                .collect();
            points.extend(pb.points.iter().filter(|p| !a.geom.intersects(*p)));
            absorb(points, lines, areas, &clip, snap.tolerance)
        }
    };
    out.into_geometry(areal_inputs)
}

/// Clips lines by polygons: `(lines, areas, invert)` → the parts outside
/// (`invert`) or inside `areas`.
type ClipFn<'a> = dyn Fn(Vec<LineString>, &MultiPolygon, bool) -> Vec<LineString> + 'a;

/// Absorption: drop line parts inside or on `areas`, and points covered by
/// `areas` or by the remaining lines.
///
/// `areas`' vertices sit on i_overlay's grid (only input vertices are snapped
/// back exactly), so a part lying on its boundary — e.g. a crossing point of the
/// two inputs' boundaries — may miss it by up to `tol`; "on the boundary" is
/// therefore tested within `tol`, against an R-tree of the boundary segments.
fn absorb(
    points: Vec<Point>,
    lines: Vec<LineString>,
    areas: MultiPolygon,
    clip: &ClipFn<'_>,
    tol: f64,
) -> Parts {
    use geo::{Distance, Euclidean};
    let lines = clip(lines, &areas, true);
    if lines.is_empty() && points.is_empty() {
        return Parts {
            points,
            lines,
            areas,
        };
    }
    // Few candidates (the usual case: a couple of boundary crossings) are tested
    // by a linear scan; many (e.g. a long shared edge) through an R-tree.
    let ring_segs = segment_list(&boundary_lines(&areas));
    let candidates = points.len() + lines.iter().map(|l| l.0.len()).sum::<usize>();
    let boundary = (candidates.saturating_mul(ring_segs.len()) > 200_000)
        .then(|| Segs::new(ring_segs.clone()));
    let on_boundary = |c: Coord| {
        let Some(boundary) = &boundary else {
            return ring_segs
                .iter()
                .any(|l| Euclidean.distance(&Point::from(c), l) <= tol);
        };
        let probe = Rect::new(
            Coord {
                x: c.x - tol,
                y: c.y - tol,
            },
            Coord {
                x: c.x + tol,
                y: c.y + tol,
            },
        );
        boundary
            .near(probe)
            .any(|l| Euclidean.distance(&Point::from(c), l) <= tol)
    };
    let midpoint = |l: Line| Coord {
        x: f64::midpoint(l.start.x, l.end.x),
        y: f64::midpoint(l.start.y, l.end.y),
    };
    let lines: Vec<LineString> = lines
        .into_iter()
        .filter(|ls| !ls.lines().all(|l| on_boundary(midpoint(l))))
        .collect();
    let line_segs = segments(&lines);
    let points = points
        .into_iter()
        .filter(|p| !on_boundary(p.0) && !is_on(p.0, &line_segs) && !areas.intersects(p))
        .collect();
    Parts {
        points,
        lines,
        areas,
    }
}

/// Where two polygon sets' boundaries touch: their shared edges (split at
/// nodes, as GEOS does) and isolated touch/crossing points. One R-tree query
/// per boundary segment of `a`; pieces are only built near `b`'s boundary.
fn boundary_touches(a: &MultiPolygon, b: &MultiPolygon) -> (Vec<LineString>, Vec<Point>) {
    let (Some(ra), Some(rb)) = (a.bounding_rect(), b.bounding_rect()) else {
        return (vec![], vec![]);
    };
    // Only segments inside the other side's bbox can touch it. This keeps the
    // common "small feature vs. large region" case proportional to the feature.
    let rings = |m: &MultiPolygon| -> Vec<Line> {
        m.iter()
            .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()))
            .flat_map(|r| r.lines())
            .filter(|l| l.start != l.end)
            .collect()
    };
    let sb = Segs::new(
        rings(b)
            .into_iter()
            .filter(|l| l.bounding_rect().intersects(&ra))
            .collect(),
    );
    if sb.lines.is_empty() {
        return (vec![], vec![]);
    }
    let (mut lines, mut points, mut shared_runs) = (vec![], vec![], vec![]);
    for seg in rings(a)
        .into_iter()
        .filter(|l| l.bounding_rect().intersects(&rb))
    {
        let near: Vec<&Line> = sb.near(seg.bounding_rect()).collect();
        if near.is_empty() {
            continue;
        }
        let mut cuts: Vec<(f64, Coord)> = vec![];
        for o in near {
            match line_intersection(seg, *o) {
                Some(LineIntersection::SinglePoint { intersection, .. }) => {
                    points.push(Point::from(intersection));
                    cuts.push((param(seg, intersection), intersection));
                }
                Some(LineIntersection::Collinear { intersection }) => {
                    shared_runs.push(intersection);
                    cuts.push((param(seg, intersection.start), intersection.start));
                    cuts.push((param(seg, intersection.end), intersection.end));
                }
                None => {}
            }
        }
        cuts.retain(|(t, c)| *t > 0.0 && *t < 1.0 && *c != seg.start && *c != seg.end);
        cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
        cuts.dedup_by(|x, y| x.1 == y.1);
        let mut start = seg.start;
        for c in cuts
            .into_iter()
            .map(|(_, c)| c)
            .chain(std::iter::once(seg.end))
        {
            let piece = Line::new(start, c);
            // Every vertex of a shared run lies on `b`, so it is a node: each
            // shared piece is its own line (GEOS splits there too).
            if is_shared(piece, &sb) {
                lines.push(LineString::new(vec![piece.start, piece.end]));
            }
            start = c;
        }
    }
    // A touch point lying on a shared edge is part of that edge.
    if !shared_runs.is_empty() {
        let shared = Segs::new(shared_runs);
        points.retain(|p| !is_on(p.0, &shared));
    }
    (lines, points)
}

/// The rings of `areas` as lines.
fn boundary_lines(areas: &MultiPolygon) -> Vec<LineString> {
    areas
        .iter()
        .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()))
        .cloned()
        .collect()
}

type SegTree =
    rstar::RTree<rstar::primitives::GeomWithData<rstar::primitives::Rectangle<[f64; 2]>, usize>>;

/// The non-degenerate segments of a set of lines, R-tree indexed so that
/// noding two polygon boundaries is ~O((n + m) log n), not O(n·m).
struct Segs {
    lines: Vec<Line>,
    tree: SegTree,
}

impl Segs {
    fn near(&self, r: Rect) -> impl Iterator<Item = &Line> + '_ {
        let env = rstar::AABB::from_corners([r.min().x, r.min().y], [r.max().x, r.max().y]);
        self.tree
            .locate_in_envelope_intersecting(env)
            .map(|item| &self.lines[item.data])
    }
}

/// Every non-degenerate segment of `lines`, unindexed.
fn segment_list(lines: &[LineString]) -> Vec<Line> {
    lines
        .iter()
        .flat_map(|ls| ls.lines())
        .filter(|l| l.start != l.end)
        .collect()
}

/// Every non-degenerate segment of `lines`.
fn segments(lines: &[LineString]) -> Segs {
    Segs::new(
        lines
            .iter()
            .flat_map(|ls| ls.lines())
            .filter(|l| l.start != l.end)
            .collect(),
    )
}

impl Segs {
    fn new(lines: Vec<Line>) -> Self {
        let items = lines
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let r = l.bounding_rect();
                rstar::primitives::GeomWithData::new(
                    rstar::primitives::Rectangle::from_corners(
                        [r.min().x, r.min().y],
                        [r.max().x, r.max().y],
                    ),
                    i,
                )
            })
            .collect();
        Self {
            lines,
            tree: rstar::RTree::bulk_load(items),
        }
    }
}

/// Isolated crossing/touching points between two segment sets (not points
/// that lie on a shared segment).
fn crossing_points(a: &[Line], b: &Segs) -> Vec<Point> {
    let mut points = vec![];
    let mut shared = vec![];
    for la in a {
        for lb in b.near(la.bounding_rect()) {
            match line_intersection(*la, *lb) {
                Some(LineIntersection::SinglePoint { intersection, .. }) => {
                    points.push(Point::from(intersection));
                }
                Some(LineIntersection::Collinear { intersection }) => shared.push(intersection),
                None => {}
            }
        }
    }
    if !shared.is_empty() {
        let shared = Segs::new(shared);
        points.retain(|p| !is_on(p.0, &shared));
    }
    points
}

/// A piece of a noded input line, and whether it lies on the other input.
#[derive(Clone, Copy)]
struct Piece {
    line: Line,
    shared: bool,
}

/// Noded lines of one input against the other.
struct Noded {
    /// Per input line, its pieces in order.
    lines: Vec<Vec<Piece>>,
    /// Nodes: crossing/touching points with the other input, and input
    /// vertices that lie on it (as bit patterns). Results are split here.
    nodes: std::collections::HashSet<(u64, u64)>,
}

/// Split every segment of `lines` at the points where it meets `other`
/// (crossings, touches, and the ends of collinear overlaps), and mark each
/// piece as shared or not. After noding, a piece is either entirely on one of
/// `other`'s segments or shares at most points with it.
fn node(lines: &[LineString], other: &Segs) -> Noded {
    let mut nodes = std::collections::HashSet::new();
    let lines = lines
        .iter()
        .map(|ls| {
            let mut pieces = vec![];
            for seg in ls.lines().filter(|l| l.start != l.end) {
                let near: Vec<&Line> = other.near(seg.bounding_rect()).collect();
                if near.is_empty() {
                    // Nothing of `other` near this segment: one unshared piece, no node.
                    pieces.push(Piece {
                        line: seg,
                        shared: false,
                    });
                    continue;
                }
                let mut cuts: Vec<(f64, Coord)> = vec![];
                for o in near {
                    match line_intersection(seg, *o) {
                        Some(LineIntersection::SinglePoint { intersection, .. }) => {
                            cuts.push((param(seg, intersection), intersection));
                            nodes.insert(bits(intersection));
                        }
                        Some(LineIntersection::Collinear { intersection }) => {
                            cuts.push((param(seg, intersection.start), intersection.start));
                            cuts.push((param(seg, intersection.end), intersection.end));
                        }
                        None => {}
                    }
                }
                cuts.retain(|(t, c)| *t > 0.0 && *t < 1.0 && *c != seg.start && *c != seg.end);
                cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
                cuts.dedup_by(|x, y| x.1 == y.1);
                if !pieces.is_empty() && is_on(seg.start, other) {
                    nodes.insert(bits(seg.start));
                }
                let mut start = seg.start;
                for c in cuts
                    .into_iter()
                    .map(|(_, c)| c)
                    .chain(std::iter::once(seg.end))
                {
                    let line = Line::new(start, c);
                    pieces.push(Piece {
                        line,
                        shared: is_shared(line, other),
                    });
                    start = c;
                }
            }
            pieces
        })
        .collect();
    Noded { lines, nodes }
}

/// True when `c` lies on one of `other`'s segments.
fn is_on(c: Coord, other: &Segs) -> bool {
    let p = Point::from(c);
    other.near(Rect::new(c, c)).any(|o| o.intersects(&p))
}

fn bits(c: Coord) -> (u64, u64) {
    (c.x.to_bits(), c.y.to_bits())
}

/// Position of `c` along `seg`, as a fraction of its length.
fn param(seg: Line, c: Coord) -> f64 {
    let d = seg.delta();
    if d.x.abs() >= d.y.abs() {
        (c.x - seg.start.x) / d.x
    } else {
        (c.y - seg.start.y) / d.y
    }
}

/// True when `line` lies on one of `other`'s segments.
fn is_shared(line: Line, other: &Segs) -> bool {
    other.near(line.bounding_rect()).any(|o| {
        matches!(
            line_intersection(line, *o),
            Some(LineIntersection::Collinear { intersection })
                if (intersection.start == line.start && intersection.end == line.end)
                    || (intersection.start == line.end && intersection.end == line.start)
        )
    })
}

/// Join the kept pieces of each noded line into linestrings. Consecutive kept
/// pieces are joined unless the point between them is a node — where the
/// shared/unshared state changes, a crossing with the other input, or an input
/// vertex lying on it — so results are split at nodes as GEOS's are.
fn chain(noded: &Noded, keep: impl Fn(&Piece) -> bool) -> Vec<LineString> {
    let mut out = vec![];
    for pieces in &noded.lines {
        let mut current: Vec<Coord> = vec![];
        let mut prev: Option<&Piece> = None;
        for piece in pieces {
            if !keep(piece) {
                if current.len() >= 2 {
                    out.push(LineString::new(std::mem::take(&mut current)));
                }
                current.clear();
                prev = None;
                continue;
            }
            let joins = prev.is_some_and(|p| {
                p.line.end == piece.line.start
                    && p.shared == piece.shared
                    && !noded.nodes.contains(&bits(piece.line.start))
            });
            if !joins {
                if current.len() >= 2 {
                    out.push(LineString::new(std::mem::take(&mut current)));
                }
                current.clear();
                current.push(piece.line.start);
            }
            current.push(piece.line.end);
            prev = Some(piece);
        }
        if current.len() >= 2 {
            out.push(LineString::new(current));
        }
    }
    out
}

/// Drop lines identical to an earlier one (in either direction).
fn dedupe_lines(lines: &mut Vec<LineString>) {
    let mut seen: std::collections::HashSet<Vec<(u64, u64)>> = std::collections::HashSet::new();
    lines.retain(|ls| {
        let fwd: Vec<(u64, u64)> =
            ls.0.iter()
                .map(|c| (c.x.to_bits(), c.y.to_bits()))
                .collect();
        let mut rev = fwd.clone();
        rev.reverse();
        seen.insert(fwd.min(rev))
    });
}

fn dedupe_points(points: &mut Vec<Point>) {
    points.sort_by(|p, q| p.x().total_cmp(&q.x()).then(p.y().total_cmp(&q.y())));
    points.dedup();
}

/// Snaps overlay output vertices back onto the exact input vertex they were
/// rounded from.
///
/// i_overlay converts coordinates to integers with a power-of-two scale of
/// `2^(29 - floor(log2(half_extent)))` (`i_float::adapter::FloatPointAdapter`),
/// so every input vertex moves by up to half a grid step — ~3 mm for a parcel
/// intersected with a 60°-wide region. Any output vertex within two grid steps
/// of an input vertex is replaced by that input vertex; new (intersection)
/// vertices further away are left alone.
struct Snapper<'a> {
    tolerance: f64,
    a: &'a [Coord],
    b: &'a [Coord],
}

impl<'a> Snapper<'a> {
    /// `a` and `b` must be sorted by x.
    fn new(ra: Option<Rect>, rb: Option<Rect>, a: &'a [Coord], b: &'a [Coord]) -> Self {
        let rect = match (ra, rb) {
            (Some(ra), Some(rb)) => Some(Rect::new(
                Coord {
                    x: ra.min().x.min(rb.min().x),
                    y: ra.min().y.min(rb.min().y),
                },
                Coord {
                    x: ra.max().x.max(rb.max().x),
                    y: ra.max().y.max(rb.max().y),
                },
            )),
            (r, None) | (None, r) => r,
        };
        let half = rect.map_or(0.0, |r| r.width().max(r.height()) / 2.0);
        let tolerance = if half.is_finite() && half > 0.0 {
            // Twice the grid step: covers the half-step rounding with margin.
            2f64.powi(half.log2().floor() as i32 - 29) * 2.0
        } else {
            0.0
        };
        Self { tolerance, a, b }
    }

    fn snap_coord(&self, c: Coord) -> Coord {
        if self.tolerance == 0.0 {
            return c;
        }
        let tol = self.tolerance;
        let mut best: Option<(f64, Coord)> = None;
        for sorted in [self.a, self.b] {
            let from = sorted.partition_point(|v| v.x < c.x - tol);
            for &cand in sorted[from..].iter().take_while(|v| v.x <= c.x + tol) {
                let d = (cand.x - c.x).hypot(cand.y - c.y);
                if d <= tol && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, cand));
                }
            }
        }
        best.map_or(c, |(_, cand)| cand)
    }

    fn snap<G: MapCoords<f64, f64, Output = G>>(&self, g: G) -> G {
        g.map_coords(|c| self.snap_coord(c))
    }

    /// Snap an areal result. Vertices move by at most two grid steps (~1e-9 of
    /// the inputs' extent), the same scale as i_overlay's own rounding, so a full
    /// validity re-check (super-linear, ~30x the overlay itself on large
    /// polygons) is not run. A ring whose winding flips or area vanishes is the
    /// one cheap-to-detect failure; then the grid result is kept.
    fn snap_areal(&self, mp: MultiPolygon) -> MultiPolygon {
        let snapped = self.snap(mp.clone());
        let twice_area = |r: &LineString| -> f64 { r.lines().map(|l| l.determinant()).sum() };
        let rings_ok = mp.iter().zip(snapped.iter()).all(|(raw, new)| {
            std::iter::once((raw.exterior(), new.exterior()))
                .chain(raw.interiors().iter().zip(new.interiors()))
                .all(|(r, n)| {
                    let (ra, na) = (twice_area(r), twice_area(n));
                    na != 0.0 && ra.signum() == na.signum()
                })
        });
        if rings_ok { snapped } else { mp }
    }
}

#[cfg(test)]
mod tests {
    use geo::{Area, Polygon, polygon, wkt};

    use super::*;

    fn run(a: Geometry, b: Geometry, op: OverlayOp) -> Geometry {
        overlay(&Prepared::new(a), &Prepared::new(b), op).unwrap()
    }

    #[test]
    fn intersection_keeps_input_vertices_exact() {
        let parcel: Polygon = polygon![
            (x: -77.03, y: 38.90), (x: -77.02994, y: 38.90),
            (x: -77.02994, y: 38.90006), (x: -77.03, y: 38.90006),
        ];
        let region: Polygon = polygon![
            (x: -100.0, y: 10.0), (x: -40.0, y: 10.0), (x: -40.0, y: 60.0), (x: -100.0, y: 60.0),
        ];
        let out = run(
            Geometry::Polygon(parcel.clone()),
            Geometry::Polygon(region),
            OverlayOp::Intersection,
        );
        let Geometry::Polygon(p) = out else {
            panic!("expected a single Polygon")
        };
        let mut got: Vec<_> = p.exterior().0.iter().map(|c| (c.x, c.y)).collect();
        let mut want: Vec<_> = parcel.exterior().0.iter().map(|c| (c.x, c.y)).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        got.dedup();
        want.dedup();
        assert_eq!(got, want);
        assert_eq!(p.unsigned_area(), parcel.unsigned_area());
    }

    fn lines(g: Geometry) -> Vec<Vec<(f64, f64)>> {
        let mut v: Vec<Vec<(f64, f64)>> = match g {
            Geometry::LineString(ls) => vec![ls],
            Geometry::MultiLineString(m) => m.0,
            Geometry::GeometryCollection(gc) if gc.0.is_empty() => vec![],
            other => panic!("expected lines, got {other:?}"),
        }
        .into_iter()
        .map(|ls| ls.0.into_iter().map(|c| (c.x, c.y)).collect())
        .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    #[test]
    fn line_difference_subtracts_shared_part_and_splits_at_crossing() {
        let a: Geometry = wkt! { LINESTRING(0. 0.,10. 0.) }.into();
        let shared: Geometry = wkt! { LINESTRING(2. 0.,4. 0.) }.into();
        assert_eq!(
            lines(run(a.clone(), shared, OverlayOp::Difference)),
            vec![vec![(0., 0.), (2., 0.)], vec![(4., 0.), (10., 0.)]]
        );
        let crossing: Geometry = wkt! { LINESTRING(5. -1.,5. 1.) }.into();
        assert_eq!(
            lines(run(a.clone(), crossing, OverlayOp::Difference)),
            vec![vec![(0., 0.), (5., 0.)], vec![(5., 0.), (10., 0.)]]
        );
        assert_eq!(
            lines(run(a.clone(), a, OverlayOp::SymDifference)),
            Vec::<Vec<(f64, f64)>>::new()
        );
    }

    #[test]
    fn line_intersection_splits_shared_run_at_vertices_like_geos() {
        let a: Geometry = wkt! { LINESTRING(0. 0.,5. 0.,10. 0.) }.into();
        let b: Geometry = wkt! { LINESTRING(1. 0.,9. 0.) }.into();
        assert_eq!(
            lines(run(a, b, OverlayOp::Intersection)),
            vec![vec![(1., 0.), (5., 0.)], vec![(5., 0.), (9., 0.)]]
        );
    }

    #[test]
    fn fast_path_matches_full_overlay() {
        use geo::Area;
        // A 200-gon (indexed) with a square hole, and small squares inside it,
        // outside it, and inside its hole.
        let ring: Vec<Coord> = (0..=200)
            .map(|i| {
                let t = std::f64::consts::TAU * f64::from(i % 200) / 200.0;
                Coord {
                    x: 10.0 * t.cos(),
                    y: 10.0 * t.sin(),
                }
            })
            .collect();
        let hole: LineString = wkt! { LINESTRING(-2. -2.,2. -2.,2. 2.,-2. 2.,-2. -2.) };
        let big = Geometry::Polygon(Polygon::new(LineString::new(ring), vec![hole]));
        let squares: [Geometry; 3] = [
            wkt! { POLYGON((5. 0.,6. 0.,6. 1.,5. 1.,5. 0.)) }.into(),
            wkt! { POLYGON((20. 0.,21. 0.,21. 1.,20. 1.,20. 0.)) }.into(),
            wkt! { POLYGON((0. 0.,1. 0.,1. 1.,0. 1.,0. 0.)) }.into(),
        ];
        let ops = [
            OverlayOp::Union,
            OverlayOp::Intersection,
            OverlayOp::Difference,
            OverlayOp::SymDifference,
        ];
        for small in squares {
            for (x, y) in [(big.clone(), small.clone()), (small.clone(), big.clone())] {
                for op in ops {
                    let (px, py) = (Prepared::new(x.clone()), Prepared::new(y.clone()));
                    assert!(placement(&px, &py).or(placement(&py, &px)).is_some());
                    let fast = overlay(&px, &py, op).unwrap();
                    let (mut sx, mut sy) = (Prepared::new(x.clone()), Prepared::new(y.clone()));
                    sx.rings = None;
                    sy.rings = None;
                    let slow = overlay(&sx, &sy, op).unwrap();
                    assert!(
                        (fast.unsigned_area() - slow.unsigned_area()).abs() < 1e-9,
                        "{op:?}: fast {} vs full {}",
                        fast.unsigned_area(),
                        slow.unsigned_area()
                    );
                }
            }
        }
    }

    #[test]
    fn touching_polygons_intersect_in_shared_boundary() {
        let a: Geometry = wkt! { POLYGON((0. 0.,1. 0.,1. 1.,0. 1.,0. 0.)) }.into();
        let edge: Geometry = wkt! { POLYGON((1. 0.,2. 0.,2. 1.,1. 1.,1. 0.)) }.into();
        assert_eq!(
            lines(run(a.clone(), edge, OverlayOp::Intersection)),
            vec![vec![(1., 0.), (1., 1.)]]
        );
        let corner: Geometry = wkt! { POLYGON((1. 1.,2. 1.,2. 2.,1. 2.,1. 1.)) }.into();
        assert_eq!(
            run(a.clone(), corner, OverlayOp::Intersection),
            Geometry::Point(Point::new(1., 1.))
        );
        // Overlapping polygons: boundary crossings are absorbed into the area.
        let overlap: Geometry = wkt! { POLYGON((0.5 0.5,2. 0.5,2. 2.,0.5 2.,0.5 0.5)) }.into();
        assert!(matches!(
            run(a, overlap, OverlayOp::Intersection),
            Geometry::Polygon(_)
        ));
    }

    #[test]
    fn line_union_is_noded_and_dissolved() {
        let a: Geometry = wkt! { LINESTRING(0. 0.,10. 0.) }.into();
        let b: Geometry = wkt! { LINESTRING(5. -5.,5. 5.) }.into();
        assert_eq!(
            lines(run(a.clone(), b, OverlayOp::Union)),
            vec![
                vec![(0., 0.), (5., 0.)],
                vec![(5., -5.), (5., 0.)],
                vec![(5., 0.), (5., 5.)],
                vec![(5., 0.), (10., 0.)],
            ]
        );
        assert_eq!(
            lines(run(a.clone(), a, OverlayOp::Union)),
            vec![vec![(0., 0.), (10., 0.)]]
        );
    }
}
