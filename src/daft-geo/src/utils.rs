use std::sync::Arc;

use arrow_buffer::NullBufferBuilder;
use common_error::{DaftError, DaftResult};
use daft_core::{
    prelude::{
        BinaryArray, BooleanArray, DataType, Field, Float64Array, FullNull, Int64Array, IntoSeries,
        ListArray, Schema, StructArray, Utf8Array,
    },
    series::Series,
};
use daft_dsl::{functions::FunctionArgs, ExprRef};
use geo::Geometry;
use geo_traits::{to_geo::ToGeoGeometry, GeometryTrait};

/// Rewrite the geo-only variants WKB has no type for (`Rect`, `Triangle`, `Line`)
/// as `Polygon` / `LineString`. wkb 0.9.3 writes a `Rect` without its ring's point
/// count, producing corrupt bytes that read back as NULL or `POLYGON EMPTY`.
fn wkb_compatible(geom: &Geometry) -> std::borrow::Cow<'_, Geometry> {
    use std::borrow::Cow;
    match geom {
        Geometry::Rect(r) => Cow::Owned(Geometry::Polygon(r.to_polygon())),
        Geometry::Triangle(t) => Cow::Owned(Geometry::Polygon(t.to_polygon())),
        Geometry::Line(l) => Cow::Owned(Geometry::LineString(geo::LineString::new(vec![l.start, l.end]))),
        Geometry::GeometryCollection(gc)
            if gc.iter().any(|m| matches!(wkb_compatible(m), Cow::Owned(_))) =>
        {
            Cow::Owned(Geometry::GeometryCollection(
                gc.iter().map(|m| wkb_compatible(m).into_owned()).collect(),
            ))
        }
        _ => Cow::Borrowed(geom),
    }
}

/// True for an empty geometry, including `POINT EMPTY`, which `parse_wkb`
/// represents as a Point with NaN coordinates (geo's `is_empty` says false).
pub(crate) fn is_empty_geom(g: &Geometry) -> bool {
    use geo::HasDimensions;
    match g {
        Geometry::Point(p) => p.x().is_nan() && p.y().is_nan(),
        Geometry::GeometryCollection(gc) => gc.iter().all(is_empty_geom),
        other => other.is_empty(),
    }
}

/// Encode a `geo::Geometry` into standard (ISO) little-endian WKB bytes.
pub fn geom_to_wkb(geom: &Geometry) -> DaftResult<Vec<u8>> {
    let mut buf = Vec::new();
    let geom = wkb_compatible(geom);
    wkb::writer::write_geometry(&mut buf, geom.as_ref(), &wkb::writer::WriteOptions::default())
        .map_err(|e| DaftError::ComputeError(format!("WKB write error: {e:?}")))?;
    Ok(buf)
}

/// Encode any borrowed geo geometry object into standard (ISO) little-endian WKB bytes.
pub fn geom_ref_to_wkb<G>(geom: &G) -> DaftResult<Vec<u8>>
where
    G: GeometryTrait<T = f64>,
{
    let mut buf = Vec::new();
    wkb::writer::write_geometry(&mut buf, geom, &wkb::writer::WriteOptions::default())
        .map_err(|e| DaftError::ComputeError(format!("WKB write error: {e:?}")))?;
    Ok(buf)
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared f64 literal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Extract a required f64 arg from `FunctionArgs<Series>` by positional index or name.
///
/// Used by `st_buffer` to read its `distance` argument at eval time.
pub fn read_f64_arg(
    inputs: &FunctionArgs<Series>,
    pos: usize,
    name: &'static str,
    fn_name: &'static str,
) -> DaftResult<f64> {
    let s = inputs.required(pos)?;
    if s.len() == 1 {
        // Cast to Float64 and extract the scalar
        let casted = s.cast(&DataType::Float64)?;
        let arr = casted.f64()?;
        if arr.get(0).is_none() && arr.len() == 1 {
            // The value is present but null (e.g. lit(None))
            return Err(DaftError::ValueError(format!(
                "{fn_name}: {name} must not be null"
            )));
        }
        if let Some(v) = arr.get(0) {
            return Ok(v);
        }
    }
    Err(DaftError::ValueError(format!(
        "{fn_name}: {name} must be a numeric literal"
    )))
}

/// Extract a required f64 arg from `FunctionArgs<ExprRef>` by positional index or name.
///
/// Used at planning time (`get_return_field`) by `st_buffer`.
pub fn read_f64_arg_expr(
    inputs: &FunctionArgs<ExprRef>,
    pos: usize,
    name: &'static str,
    fn_name: &'static str,
) -> DaftResult<f64> {
    let expr = inputs.required(pos)?;
    expr.as_literal()
        .and_then(|l| l.as_f64().or_else(|| l.as_i64().map(|v| v as f64)))
        .ok_or_else(|| {
            DaftError::ValueError(format!("{fn_name}: {name} must be a numeric literal"))
        })
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared use_spheroid helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Extract an optional boolean arg from `FunctionArgs<Series>` by positional index or name.
///
/// Used by `st_area`, `st_length`, and `st_distance` to read their `use_spheroid` argument.
pub fn read_bool_arg(
    inputs: &FunctionArgs<Series>,
    pos: usize,
    name: &'static str,
    fn_name: &'static str,
) -> DaftResult<bool> {
    let opt = inputs.optional((pos, name))?;
    match opt {
        None => Ok(false),
        Some(s) => {
            if s.data_type().is_boolean() && s.len() == 1 {
                Ok(s.bool().unwrap().get(0).unwrap_or(false))
            } else {
                Err(DaftError::ValueError(format!(
                    "{fn_name}: {name} must be a boolean literal"
                )))
            }
        }
    }
}

/// Extract an optional boolean arg from `FunctionArgs<ExprRef>` by positional index or name.
///
/// Used at planning time (`get_return_field`) by `st_area`, `st_length`, and `st_distance`.
pub fn read_bool_arg_expr(
    inputs: &FunctionArgs<ExprRef>,
    pos: usize,
    name: &'static str,
    fn_name: &'static str,
) -> DaftResult<bool> {
    let opt = inputs.optional((pos, name))?;
    match opt {
        None => Ok(false),
        Some(expr) => expr.as_literal().and_then(|l| l.as_bool()).ok_or_else(|| {
            DaftError::ValueError(format!("{fn_name}: {name} must be a boolean literal"))
        }),
    }
}

/// Strip EWKB SRID prefix so the standard `wkb` crate can parse it.
///
/// If the WKB type word has the PostGIS SRID flag (0x20000000) set, return a
/// new buffer with that flag cleared and the 4-byte SRID removed.  Otherwise
/// return a slice of the original bytes unchanged.
fn strip_ewkb_srid<'a>(bytes: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
    if bytes.len() < 5 {
        return std::borrow::Cow::Borrowed(bytes);
    }
    let le = bytes[0] == 1;
    let raw_type = if le {
        u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]])
    } else {
        u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]])
    };
    if raw_type & 0x20000000 == 0 {
        return std::borrow::Cow::Borrowed(bytes);
    }
    // Clear the SRID flag and remove the 4-byte SRID field
    let clean_type = raw_type & !0x20000000;
    let clean_type_bytes = if le {
        clean_type.to_le_bytes()
    } else {
        clean_type.to_be_bytes()
    };
    let mut out = Vec::with_capacity(bytes.len() - 4);
    out.push(bytes[0]);
    out.extend_from_slice(&clean_type_bytes);
    // Skip bytes[5..9] (the SRID), then copy the rest
    if bytes.len() > 9 {
        out.extend_from_slice(&bytes[9..]);
    }
    std::borrow::Cow::Owned(out)
}

/// Parse WKB bytes into a `geo::Geometry`.
///
/// Accepts both standard ISO WKB and PostGIS EWKB (with SRID).
pub fn parse_wkb(bytes: &[u8]) -> DaftResult<Geometry> {
    let stripped = strip_ewkb_srid(bytes);
    let wkb = wkb::reader::read_wkb(stripped.as_ref())
        .map_err(|e| DaftError::ComputeError(format!("WKB parse error: {e:?}")))?;
    wkb.try_to_geometry()
        .ok_or_else(|| DaftError::ComputeError("WKB parse error: empty geometry".to_string()))
}

/// Get the physical BinaryArray backing a Geometry or Binary Series.
pub fn get_geometry_binary(series: &Series) -> DaftResult<&BinaryArray> {
    match series.data_type() {
        DataType::Geometry => Ok(&series.geometry()?.physical),
        DataType::Binary => series.binary(),
        _ => Err(DaftError::TypeError(format!(
            "Expected Geometry or Binary column, got {}",
            series.data_type()
        ))),
    }
}

/// Validate that a function argument at `arg_pos` is Geometry or Binary.
pub fn validate_geometry_field(
    inputs: &FunctionArgs<ExprRef>,
    schema: &Schema,
    arg_pos: usize,
    arg_name: &str,
    fn_name: &str,
) -> DaftResult<Field> {
    let f = inputs.required(arg_pos)?.to_field(schema)?;
    if !matches!(f.dtype, DataType::Geometry | DataType::Binary) {
        return Err(DaftError::TypeError(format!(
            "{fn_name}: {arg_name} must be Geometry or Binary (WKB), got {}",
            f.dtype
        )));
    }
    Ok(f)
}

// ──────────────────────────────────────────────────────────────────────────────
// Generic computation helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Apply a unary mapping over a geometry column → Float64 Series.
pub fn unary_geom_to_f64(
    series: &Series,
    out_name: &str,
    f: impl Fn(&Geometry) -> f64,
) -> DaftResult<Series> {
    let binary = get_geometry_binary(series)?;
    let len = binary.len();
    let mut values = Vec::with_capacity(len);
    let mut validity = NullBufferBuilder::new(len);

    for opt in binary.into_iter() {
        match opt.and_then(|b| parse_wkb(b).ok()) {
            Some(geom) => {
                values.push(f(&geom));
                validity.append_non_null();
            }
            None => {
                values.push(0.0);
                validity.append_null();
            }
        }
    }

    let field = Arc::new(Field::new(out_name, DataType::Float64));
    let arr = Float64Array::from_field_and_values(field, values).with_nulls(validity.finish())?;
    Ok(arr.into_series())
}

/// Apply a unary mapping over a geometry column → Boolean Series.
pub fn unary_geom_to_bool(
    series: &Series,
    out_name: &str,
    f: impl Fn(&Geometry) -> bool,
) -> DaftResult<Series> {
    let binary = get_geometry_binary(series)?;
    let len = binary.len();

    if len == 0 {
        return Ok(BooleanArray::empty(out_name, &DataType::Boolean).into_series());
    }

    let mut values: Vec<Option<bool>> = Vec::with_capacity(len);
    for opt in binary.into_iter() {
        values.push(opt.and_then(|b| parse_wkb(b).ok()).map(|g| f(&g)));
    }

    Ok(BooleanArray::from_iter(out_name, values.into_iter()).into_series())
}

/// Apply a unary mapping over a geometry column → Utf8 Series.
pub fn unary_geom_to_utf8(
    series: &Series,
    out_name: &str,
    f: impl Fn(&Geometry) -> String,
) -> DaftResult<Series> {
    let binary = get_geometry_binary(series)?;
    let values: Vec<Option<String>> = binary
        .into_iter()
        .map(|opt| opt.and_then(|b| parse_wkb(b).ok()).map(|g| f(&g)))
        .collect();

    Ok(Utf8Array::from_iter(out_name, values.iter().map(|v| v.as_deref())).into_series())
}

/// Wrap a `Vec<Option<Vec<u8>>>` of WKB bytes into a Geometry logical Series.
///
/// This is the shared tail used by `unary_geom_to_geom`, `binary_geom_to_geom`,
/// and `st_point` — any function that produces WKB-encoded geometries row-by-row.
pub(crate) fn wkb_opts_to_geometry_series(
    out_name: &str,
    wkb_values: Vec<Option<Vec<u8>>>,
) -> DaftResult<Series> {
    use daft_core::prelude::{GeometryType, LogicalArray};

    let len = wkb_values.len();
    let field = Field::new(out_name, DataType::Geometry);
    let mut builder = arrow_buffer::NullBufferBuilder::new(len);
    let phys_values: Vec<Option<&[u8]>> = wkb_values
        .iter()
        .map(|v| {
            if v.is_some() {
                builder.append_non_null();
            } else {
                builder.append_null();
            }
            v.as_deref()
        })
        .collect();

    let phys_field = Field::new(out_name, DataType::Binary);
    let phys_arr = BinaryArray::from_iter(&phys_field.name, phys_values.into_iter());
    let logical = LogicalArray::<GeometryType>::new(field, phys_arr.with_nulls(builder.finish())?);
    Ok(logical.into_series())
}

pub(crate) fn dump_item_struct_fields() -> Vec<Field> {
    vec![
        Field::new("path", DataType::List(Box::new(DataType::Int64))),
        Field::new("geom", DataType::Geometry),
    ]
}

/// Wrap flattened dump items and list offsets into a `List[Struct{path, geom}]` Series.
pub(crate) fn dump_items_to_list_struct_series(
    out_name: &str,
    child_paths: Vec<Vec<i64>>,
    child_wkb_values: Vec<Option<Vec<u8>>>,
    offsets: Vec<i64>,
    outer_validity: Vec<bool>,
) -> DaftResult<Series> {
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};

    let mut flat_path_values: Vec<i64> = Vec::new();
    let mut path_offsets: Vec<i64> = Vec::with_capacity(child_paths.len() + 1);
    path_offsets.push(0);
    for path in &child_paths {
        flat_path_values.extend(path.iter().copied());
        path_offsets.push(flat_path_values.len() as i64);
    }

    let path_child = Int64Array::from_iter(
        Field::new("item", DataType::Int64),
        flat_path_values.into_iter().map(Some),
    )
    .into_series();
    let path_offsets = OffsetBuffer::new(ScalarBuffer::from(path_offsets));
    let path_series = ListArray::new(
        Field::new("path", DataType::List(Box::new(DataType::Int64))),
        path_child,
        path_offsets,
        None,
    )
    .into_series();

    let geom_series = wkb_opts_to_geometry_series("geom", child_wkb_values)?;

    let struct_dtype = DataType::Struct(dump_item_struct_fields());
    let struct_series = StructArray::new(
        Field::new("item", struct_dtype.clone()),
        vec![path_series, geom_series],
        None,
    )
    .into_series();

    let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let nulls = NullBuffer::from_iter(outer_validity);
    Ok(ListArray::new(
        Field::new(out_name, DataType::List(Box::new(struct_dtype))),
        struct_series,
        offsets,
        Some(nulls),
    )
    .into_series())
}

/// Apply a unary mapping over a geometry column → Geometry (WKB Binary) Series.
pub fn unary_geom_to_geom(
    series: &Series,
    out_name: &str,
    f: impl Fn(&Geometry) -> Option<Geometry>,
) -> DaftResult<Series> {
    let binary = get_geometry_binary(series)?;
    let len = binary.len();
    let mut wkb_values: Vec<Option<Vec<u8>>> = Vec::with_capacity(len);

    for opt in binary.into_iter() {
        let result = opt
            .and_then(|b| parse_wkb(b).ok())
            .and_then(|g| f(&g))
            .and_then(|out_geom| geom_to_wkb(&out_geom).ok());
        wkb_values.push(result);
    }

    wkb_opts_to_geometry_series(out_name, wkb_values)
}

/// Apply a unary mapping over a geometry column -> `List[Struct{path, geom}]` Series.
pub fn unary_geom_to_list_dump_item(
    series: &Series,
    out_name: &str,
    f: impl Fn(Geometry) -> Option<Vec<(Vec<i64>, Geometry)>>,
) -> DaftResult<Series> {
    let binary = get_geometry_binary(series)?;
    let len = binary.len();
    let mut child_paths: Vec<Vec<i64>> = Vec::new();
    let mut child_wkb_values: Vec<Option<Vec<u8>>> = Vec::new();
    let mut offsets: Vec<i64> = Vec::with_capacity(len + 1);
    let mut outer_validity: Vec<bool> = Vec::with_capacity(len);

    offsets.push(0);

    for opt in binary.into_iter() {
        let row_result = opt.and_then(|b| parse_wkb(b).ok()).and_then(&f);
        match row_result {
            Some(items) => {
                outer_validity.push(true);
                for (path, geom) in items {
                    child_paths.push(path);
                    child_wkb_values.push(geom_to_wkb(&geom).ok());
                }
            }
            None => outer_validity.push(false),
        }
        offsets.push(child_wkb_values.len() as i64);
    }

    dump_items_to_list_struct_series(
        out_name,
        child_paths,
        child_wkb_values,
        offsets,
        outer_validity,
    )
}

/// Output length of a broadcasting binary op over columns of length `l` and `r`.
///
/// Either side may be length 1 (a literal) and is then broadcast over the
/// other side; otherwise the lengths must match.
pub(crate) fn broadcast_len(l: usize, r: usize, out_name: &str) -> DaftResult<usize> {
    match (l, r) {
        (l, r) if l == r => Ok(l),
        (1, r) => Ok(r),
        (l, 1) => Ok(l),
        (l, r) => Err(DaftError::ValueError(format!(
            "{out_name}: expected inputs of equal length or length 1, got {l} and {r}"
        ))),
    }
}

/// Call `f` for every row of two geometry columns, with broadcasting.
///
/// Each geometry is parsed and passed through `prepare`; a length-1 side is
/// parsed and prepared once and reused for every row. Null or unparseable
/// geometries are passed to `f` as `None`. Returns the row count.
fn for_each_prepared_pair<T>(
    lhs: &Series,
    rhs: &Series,
    out_name: &str,
    prepare: impl Fn(Geometry) -> T,
    mut f: impl FnMut(Option<&T>, Option<&T>),
) -> DaftResult<usize> {
    let lhs_bin = get_geometry_binary(lhs)?;
    let rhs_bin = get_geometry_binary(rhs)?;
    let len = broadcast_len(lhs_bin.len(), rhs_bin.len(), out_name)?;
    let parse = |b: Option<&[u8]>| b.and_then(|b| parse_wkb(b).ok()).map(&prepare);
    let scalar = |arr: &BinaryArray| -> Option<Option<T>> {
        (arr.len() == 1 && len != 1).then(|| parse(arr.into_iter().next().flatten()))
    };
    match (scalar(lhs_bin), scalar(rhs_bin)) {
        (Some(lg), _) => {
            for ropt in rhs_bin {
                f(lg.as_ref(), parse(ropt).as_ref());
            }
        }
        (None, Some(rg)) => {
            for lopt in lhs_bin {
                f(parse(lopt).as_ref(), rg.as_ref());
            }
        }
        (None, None) => {
            for (lopt, ropt) in lhs_bin.into_iter().zip(rhs_bin) {
                f(parse(lopt).as_ref(), parse(ropt).as_ref());
            }
        }
    }
    Ok(len)
}

fn for_each_geom_pair(
    lhs: &Series,
    rhs: &Series,
    out_name: &str,
    f: impl FnMut(Option<&Geometry>, Option<&Geometry>),
) -> DaftResult<usize> {
    for_each_prepared_pair(lhs, rhs, out_name, |g| g, f)
}

/// Like [`binary_geom_to_geom`], but each operand is first converted with
/// `prepare` (e.g. decomposed and indexed) — once for a broadcast literal.
pub(crate) fn binary_prepared_to_geom<T>(
    lhs: &Series,
    rhs: &Series,
    out_name: &str,
    prepare: impl Fn(Geometry) -> T,
    f: impl Fn(&T, &T) -> Option<Geometry>,
) -> DaftResult<Series> {
    let mut wkb_values: Vec<Option<Vec<u8>>> = Vec::new();
    for_each_prepared_pair(lhs, rhs, out_name, prepare, |l, r| {
        wkb_values.push(
            l.zip(r)
                .and_then(|(l, r)| f(l, r))
                .and_then(|g| geom_to_wkb(&g).ok()),
        );
    })?;
    wkb_opts_to_geometry_series(out_name, wkb_values)
}

/// Apply a binary predicate over two geometry columns → Boolean Series.
///
/// `f` returns `None` when the predicate cannot be evaluated (e.g. the
/// geometry engine panicked); that row is null rather than a wrong answer.
pub fn binary_geom_to_bool(
    lhs: &Series,
    rhs: &Series,
    out_name: &str,
    f: impl Fn(&Geometry, &Geometry) -> Option<bool>,
) -> DaftResult<Series> {
    let mut values: Vec<Option<bool>> = Vec::new();
    for_each_geom_pair(lhs, rhs, out_name, |l, r| {
        values.push(l.zip(r).and_then(|(l, r)| f(l, r)));
    })?;
    Ok(BooleanArray::from_iter(out_name, values.into_iter()).into_series())
}

/// Apply a binary mapping over two geometry columns → Geometry (WKB) Series.
pub fn binary_geom_to_geom(
    lhs: &Series,
    rhs: &Series,
    out_name: &str,
    f: impl Fn(&Geometry, &Geometry) -> Option<Geometry>,
) -> DaftResult<Series> {
    let mut wkb_values: Vec<Option<Vec<u8>>> = Vec::new();
    for_each_geom_pair(lhs, rhs, out_name, |l, r| {
        wkb_values.push(
            l.zip(r)
                .and_then(|(l, r)| f(l, r))
                .and_then(|g| geom_to_wkb(&g).ok()),
        );
    })?;
    wkb_opts_to_geometry_series(out_name, wkb_values)
}

/// Apply a binary geometry → Float64 function (e.g. distance).
///
/// `f` returns `None` when the value is undefined (e.g. an empty geometry);
/// that row is null.
pub fn binary_geom_to_f64(
    lhs: &Series,
    rhs: &Series,
    out_name: &str,
    f: impl Fn(&Geometry, &Geometry) -> Option<f64>,
) -> DaftResult<Series> {
    let mut values = Vec::new();
    let mut validity = NullBufferBuilder::new(0);
    for_each_geom_pair(lhs, rhs, out_name, |l, r| {
        match l.zip(r).and_then(|(l, r)| f(l, r)) {
            Some(v) => {
                values.push(v);
                validity.append_non_null();
            }
            None => {
                values.push(0.0);
                validity.append_null();
            }
        }
    })?;
    let field = Arc::new(Field::new(out_name, DataType::Float64));
    let arr = Float64Array::from_field_and_values(field, values).with_nulls(validity.finish())?;
    Ok(arr.into_series())
}
