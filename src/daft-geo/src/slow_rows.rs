//! Opt-in watchdog that reports geometry rows taking too long.
//!
//! Set `DAFT_GEO_SLOW_ROW_SECS` (e.g. `60`) to turn it on. Every row of a
//! geometry function (`st_intersection`, `st_dump`, `st_normalize`, ...) is
//! then registered while it runs, and a background thread reports each row
//! still running past the threshold, once, to stderr (the Ray worker's `.err`
//! log): the function, how long it has run, and each input's vertex count and
//! bounding box. The input WKB is also written to `DAFT_GEO_SLOW_ROW_DIR`
//! (default `/tmp/daft_geo_slow`) as `<pid>_<row id>_<a|b>.wkb`, so the row can
//! be replayed offline. A reported row logs again when it finishes.
//!
//! The report comes from the watchdog, not from the row itself, because the
//! rows this is for may never finish. Unset, it costs one cached check per row.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use geo::{BoundingRect, CoordsIter};

use crate::utils::parse_wkb;

/// A borrowed input row (WKB bytes). Only read while its row is registered.
#[derive(Clone, Copy)]
struct RawWkb(*const u8, usize);

// SAFETY: a `RawWkb` is only dereferenced by the watchdog while it holds the
// registry lock and the entry is registered; a row deregisters (taking the same
// lock) before its borrow of the input ends, so the bytes are alive and unchanged.
unsafe impl Send for RawWkb {}

impl RawWkb {
    fn of(bytes: &[u8]) -> Self {
        Self(bytes.as_ptr(), bytes.len())
    }

    /// # Safety
    /// The row that registered these bytes must still be registered.
    unsafe fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0, self.1) }
    }
}

struct Entry {
    op: String,
    start: Instant,
    inputs: [Option<RawWkb>; 2],
    reported: bool,
}

struct Watch {
    threshold: Duration,
    dir: PathBuf,
    rows: Mutex<HashMap<u64, Entry>>,
    next_id: AtomicU64,
}

static WATCH: OnceLock<Option<&'static Watch>> = OnceLock::new();

fn watch_from_env() -> Option<&'static Watch> {
    let secs: f64 = std::env::var("DAFT_GEO_SLOW_ROW_SECS").ok()?.trim().parse().ok()?;
    if !(secs.is_finite() && secs >= 0.0) {
        return None;
    }
    let dir = std::env::var("DAFT_GEO_SLOW_ROW_DIR").unwrap_or_else(|_| "/tmp/daft_geo_slow".into());
    Some(start_watch(Duration::from_secs_f64(secs), PathBuf::from(dir)))
}

fn start_watch(threshold: Duration, dir: PathBuf) -> &'static Watch {
    let watch: &'static Watch = Box::leak(Box::new(Watch {
        threshold,
        dir,
        rows: Mutex::new(HashMap::new()),
        next_id: AtomicU64::new(0),
    }));
    let poll = (threshold / 2).clamp(Duration::from_millis(20), Duration::from_secs(5));
    std::thread::Builder::new()
        .name("daft-geo-slow-rows".into())
        .spawn(move || {
            loop {
                std::thread::sleep(poll);
                watch.report_slow();
            }
        })
        .expect("spawn the daft-geo slow-row watchdog");
    watch
}

/// Registered for as long as it lives; see the module docs.
pub(crate) struct RowGuard {
    watch: &'static Watch,
    id: u64,
}

/// Register one row of `op` over inputs `a` (and `b`), if the watchdog is on.
///
/// The returned guard must not outlive `a` / `b`.
pub(crate) fn watch_row(op: &str, a: Option<&[u8]>, b: Option<&[u8]>) -> Option<RowGuard> {
    let watch = (*WATCH.get_or_init(watch_from_env))?;
    Some(watch.register(op, a, b))
}

impl Watch {
    fn register(&'static self, op: &str, a: Option<&[u8]>, b: Option<&[u8]>) -> RowGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = Entry {
            op: op.to_string(),
            start: Instant::now(),
            inputs: [a.map(RawWkb::of), b.map(RawWkb::of)],
            reported: false,
        };
        self.rows.lock().unwrap_or_else(|e| e.into_inner()).insert(id, entry);
        RowGuard { watch: self, id }
    }

    fn report_slow(&self) {
        let mut rows = self.rows.lock().unwrap_or_else(|e| e.into_inner());
        for (id, entry) in rows.iter_mut() {
            let elapsed = entry.start.elapsed();
            if entry.reported || elapsed < self.threshold {
                continue;
            }
            entry.reported = true;
            let mut parts = Vec::new();
            for (side, input) in ["a", "b"].iter().zip(entry.inputs.iter()) {
                let Some(raw) = input else { continue };
                // SAFETY: the entry is registered and we hold the lock (see `RawWkb`).
                let bytes = unsafe { raw.bytes() };
                parts.push(format!("{side}: {}", describe(bytes)));
                let path = self.dir.join(format!("{}_{id}_{side}.wkb", std::process::id()));
                let saved = std::fs::create_dir_all(&self.dir).and_then(|()| std::fs::write(&path, bytes));
                parts.push(match saved {
                    Ok(()) => format!("{side} saved to {}", path.display()),
                    Err(e) => format!("{side} not saved ({e})"),
                });
            }
            eprintln!(
                "daft-geo slow row {id}: {} running {:.0}s (pid {}); {}",
                entry.op,
                elapsed.as_secs_f64(),
                std::process::id(),
                parts.join("; ")
            );
        }
    }
}

fn describe(bytes: &[u8]) -> String {
    match parse_wkb(bytes) {
        Ok(g) => {
            let vertices = g.coords_iter().count();
            let bbox = g
                .bounding_rect()
                .map(|r| format!("[{} {}, {} {}]", r.min().x, r.min().y, r.max().x, r.max().y))
                .unwrap_or_else(|| "empty".into());
            format!("{} WKB bytes, {vertices} vertices, bbox {bbox}", bytes.len())
        }
        Err(e) => format!("{} WKB bytes, unparsable ({e})", bytes.len()),
    }
}

impl Drop for RowGuard {
    fn drop(&mut self) {
        let removed = self.watch.rows.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.id);
        if let Some(entry) = removed.filter(|e| e.reported) {
            eprintln!(
                "daft-geo slow row {}: {} finished after {:.0}s",
                self.id,
                entry.op,
                entry.start.elapsed().as_secs_f64()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::geom_to_wkb;

    fn square_wkb() -> Vec<u8> {
        let ring = geo::LineString::from(vec![(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 0.0)]);
        geom_to_wkb(&geo::Geometry::Polygon(geo::Polygon::new(ring, vec![]))).unwrap()
    }

    #[test]
    fn a_row_past_the_threshold_is_saved_and_a_fast_one_is_not() {
        let dir = std::env::temp_dir().join(format!("daft_geo_slow_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let watch = start_watch(Duration::from_millis(100), dir.clone());
        let (a, b) = (square_wkb(), square_wkb());

        let fast = watch.register("st_fast", Some(&a), None);
        let fast_id = fast.id;
        drop(fast);

        let slow = watch.register("st_slow", Some(&a), Some(&b));
        let slow_id = slow.id;
        std::thread::sleep(Duration::from_millis(500));
        drop(slow);

        let pid = std::process::id();
        for side in ["a", "b"] {
            let saved = std::fs::read(dir.join(format!("{pid}_{slow_id}_{side}.wkb"))).unwrap();
            assert_eq!(saved, a);
        }
        assert!(!dir.join(format!("{pid}_{fast_id}_a.wkb")).exists());
        assert!(watch.rows.lock().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn describe_reports_vertices_and_bbox() {
        let text = describe(&square_wkb());
        assert!(text.contains("4 vertices"), "{text}");
        assert!(text.contains("bbox [0 0, 2 1]"), "{text}");
    }
}
