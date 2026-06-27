use std::{marker::PhantomData, sync::Arc};

use arrow::{
    array::{ArrayData, BooleanBufferBuilder, NullBufferBuilder, make_array},
    buffer::{Buffer, MutableBuffer, NullBuffer, ScalarBuffer},
};
use common_error::DaftResult;

use super::Growable;
use crate::{
    array::prelude::*,
    datatypes::prelude::*,
    series::{IntoSeries, Series},
};

/// One source array's buffers/offset/validity, precomputed once so `extend` does no per-call
/// `ArrayData::buffers()/offset()/nulls()` lookups. All fields are owned Arc-backed clones,
/// so there is no borrow into the growable itself.
struct SourceCache {
    offset: usize,
    nulls: Option<NullBuffer>,
    buf0: Buffer,         // fixed-width/boolean: values; var-len: offsets
    buf1: Option<Buffer>, // var-len: values
}

/// Handles three physical buffer layouts for direct buffer manipulation.
/// Selected at construction time based on Daft `DataType`.
enum ValueGrower {
    /// Fixed-width types: primitives, decimal, interval, temporals, and fixed-size binary.
    /// Single contiguous buffer of elements with known byte width.
    FixedWidth {
        buffer: MutableBuffer,
        byte_width: usize,
    },
    /// Bit-packed boolean values.
    Boolean { builder: BooleanBufferBuilder },
    /// Variable-length types (LargeUtf8, LargeBinary): i64 offsets + value bytes.
    VarLen {
        offsets: Vec<i64>,
        values: MutableBuffer,
    },
}

/// Select the appropriate ValueGrower variant based on the Daft DataType.
/// Temporal types (Date, Timestamp, etc.) are included for the Extension recursive case.
fn grower_from_dtype(dtype: &DataType, capacity: usize) -> ValueGrower {
    let fixed_width = |bw: usize| ValueGrower::FixedWidth {
        buffer: MutableBuffer::new(capacity * bw),
        byte_width: bw,
    };
    match dtype {
        DataType::Boolean => ValueGrower::Boolean {
            builder: BooleanBufferBuilder::new(capacity),
        },
        DataType::Int8 | DataType::UInt8 => fixed_width(1),
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => fixed_width(2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date => fixed_width(4),
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Timestamp(..)
        | DataType::Duration(..)
        | DataType::Time(..) => fixed_width(8),
        DataType::Decimal128(..) | DataType::Interval => fixed_width(16),
        DataType::Utf8 | DataType::Binary => ValueGrower::VarLen {
            offsets: {
                let mut v = Vec::with_capacity(capacity + 1);
                v.push(0i64);
                v
            },
            values: MutableBuffer::new(0),
        },
        DataType::FixedSizeBinary(n) => fixed_width(*n),
        DataType::Extension(_, inner, _) => grower_from_dtype(inner, capacity),
        other => panic!("Unsupported DataType for ArrowGrowable: {other:?}"),
    }
}

/// High-performance growable for all `DaftArrowBackedType` variants.
///
/// Instead of deferring operations to `MutableArrayData` (which uses internal
/// function-pointer dispatch), this copies directly into raw buffers using
/// `extend_from_slice` - about 2x faster than arrow2's growable.
pub struct ArrowGrowable<'a, T: DaftArrowBackedType> {
    name: String,
    dtype: DataType,
    arrow_dtype: arrow::datatypes::DataType,
    sources: Vec<SourceCache>,
    grower: ValueGrower,
    validity: Option<NullBufferBuilder>,
    len: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: DaftArrowBackedType> ArrowGrowable<'a, T> {
    pub fn new(
        name: &str,
        dtype: &DataType,
        arrays: Vec<&'a DataArray<T>>,
        use_validity: bool,
        capacity: usize,
    ) -> Self {
        let mut sources = Vec::with_capacity(arrays.len());
        let mut needs_validity = use_validity;
        let mut arrow_dtype: Option<arrow::datatypes::DataType> = None;
        for a in &arrays {
            let data = a.to_data();
            if arrow_dtype.is_none() {
                arrow_dtype = Some(data.data_type().clone());
            }
            let bufs = data.buffers();
            let nulls = data.nulls().cloned();
            if nulls.is_some() {
                needs_validity = true;
            }
            sources.push(SourceCache {
                offset: data.offset(),
                nulls,
                buf0: bufs[0].clone(),
                buf1: bufs.get(1).cloned(),
            });
        }
        // Get arrow dtype from first source (handles extension types correctly,
        // since Extension dtype cannot go through DataType::to_arrow()).
        let arrow_dtype = arrow_dtype
            .unwrap_or_else(|| dtype.to_arrow().unwrap_or(arrow::datatypes::DataType::Null));

        let grower = grower_from_dtype(dtype, capacity);
        let validity = if needs_validity {
            Some(NullBufferBuilder::new(capacity))
        } else {
            None
        };

        Self {
            name: name.to_string(),
            dtype: dtype.clone(),
            arrow_dtype,
            sources,
            grower,
            validity,
            len: 0,
            _phantom: PhantomData,
        }
    }
}

impl<T: DaftArrowBackedType> Growable for ArrowGrowable<'_, T>
where
    DataArray<T>: IntoSeries,
{
    #[inline]
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        let src = &self.sources[index];

        // Extend validity bitmap.
        // NullBuffer from ArrayData::nulls() has offset baked in, so use logical indices.
        if let Some(ref mut validity) = self.validity {
            if let Some(nulls) = &src.nulls {
                validity.append_buffer(&nulls.slice(start, len));
            } else {
                validity.append_n_non_nulls(len);
            }
        }

        // Extend value buffer(s).
        // Raw buffers from ArrayData::buffers() are NOT offset-adjusted,
        // so we must add source.offset for physical byte access.
        let offset = src.offset;
        match &mut self.grower {
            ValueGrower::FixedWidth { buffer, byte_width } => {
                let bw = *byte_width;
                let s = src.buf0.as_slice();
                let byte_start = (offset + start) * bw;
                buffer.extend_from_slice(&s[byte_start..byte_start + len * bw]);
            }
            ValueGrower::Boolean { builder } => {
                let s = src.buf0.as_slice();
                let base = offset + start;
                for i in 0..len {
                    let bit_idx = base + i;
                    let is_set = (s[bit_idx / 8] >> (bit_idx % 8)) & 1 == 1;
                    builder.append(is_set);
                }
            }
            ValueGrower::VarLen { offsets, values } => {
                let src_offsets: &[i64] = src.buf0.typed_data();
                let src_values = src.buf1.as_ref().expect("var-len source has values buffer").as_slice();
                let base = offset + start;
                for i in 0..len {
                    let idx = base + i;
                    let val_start = src_offsets[idx] as usize;
                    let val_end = src_offsets[idx + 1] as usize;
                    values.extend_from_slice(&src_values[val_start..val_end]);
                    offsets.push(offsets.last().unwrap() + (val_end - val_start) as i64);
                }
            }
        }

        self.len += len;
    }

    #[inline]
    fn add_nulls(&mut self, additional: usize) {
        if let Some(ref mut validity) = self.validity {
            validity.append_n_nulls(additional);
        }

        match &mut self.grower {
            ValueGrower::FixedWidth { buffer, byte_width } => {
                buffer.resize(buffer.len() + additional * *byte_width, 0);
            }
            ValueGrower::Boolean { builder } => {
                builder.append_n(additional, false);
            }
            ValueGrower::VarLen { offsets, .. } => {
                let last = *offsets.last().unwrap();
                offsets.resize(offsets.len() + additional, last);
            }
        }

        self.len += additional;
    }

    #[inline(never)]
    fn build(&mut self) -> DaftResult<Series> {
        let null_buffer = self.validity.as_mut().and_then(|v| v.finish());

        // Collect buffers from the grower, then build ArrayData.
        let buffers: Vec<Buffer> = match &mut self.grower {
            ValueGrower::FixedWidth { buffer, .. } => {
                vec![std::mem::replace(buffer, MutableBuffer::new(0)).into()]
            }
            ValueGrower::Boolean { builder } => {
                vec![builder.finish().into_inner()]
            }
            ValueGrower::VarLen { offsets, values } => {
                let offsets_vec = std::mem::replace(offsets, vec![0i64]);
                vec![
                    ScalarBuffer::from(offsets_vec).into_inner(),
                    std::mem::replace(values, MutableBuffer::new(0)).into(),
                ]
            }
        };

        // SAFETY: buffers are constructed correctly by extend/add_nulls.
        let mut builder = ArrayData::builder(self.arrow_dtype.clone())
            .len(self.len)
            .nulls(null_buffer);
        for buf in buffers {
            builder = builder.add_buffer(buf);
        }
        let data = unsafe { builder.build_unchecked() };

        self.len = 0;

        let arrow_array = make_array(data);
        let field = Arc::new(Field::new(self.name.clone(), self.dtype.clone()));
        Ok(DataArray::<T>::from_arrow(field, arrow_array)?.into_series())
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Simplified null growable — just tracks a length counter.
/// No sources needed since every element is null.
pub struct ArrowNullGrowable {
    name: String,
    dtype: DataType,
    len: usize,
}

impl ArrowNullGrowable {
    pub fn new(name: &str, dtype: &DataType) -> Self {
        Self {
            name: name.to_string(),
            dtype: dtype.clone(),
            len: 0,
        }
    }
}

impl Growable for ArrowNullGrowable {
    #[inline]
    fn extend(&mut self, _index: usize, _start: usize, len: usize) {
        self.len += len;
    }

    #[inline]
    fn add_nulls(&mut self, additional: usize) {
        self.len += additional;
    }

    #[inline]
    fn build(&mut self) -> DaftResult<Series> {
        let len = self.len;
        self.len = 0;
        Ok(NullArray::full_null(&self.name, &self.dtype, len).into_series())
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that extend with a non-zero start correctly offsets into the source buffer.
    #[test]
    fn test_extend_from_nonzero_start() {
        let field = Field::new("test", DataType::Int32);
        let src = Int32Array::from_iter(
            field,
            vec![Some(10), Some(20), Some(30), Some(40), Some(50)],
        );
        let mut growable =
            ArrowGrowable::<Int32Type>::new("test", &DataType::Int32, vec![&src], false, 0);
        // Take elements at indices 2..4 → [30, 40]
        growable.extend(0, 2, 2);
        let result = growable.build().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.i32().unwrap().get(0), Some(30));
        assert_eq!(result.i32().unwrap().get(1), Some(40));
    }

    #[test]
    fn extend_multi_source_with_nulls_and_offset() {
        let field = Field::new("v", DataType::Int32);
        // src0 (no nulls), then slice it to offset=1 -> logical [20,30,40,50]
        let src0_full =
            Int32Array::from_iter(field.clone(), vec![Some(10), Some(20), Some(30), Some(40), Some(50)]);
        let src0 = src0_full.slice(1, 5).unwrap(); // offset=1, values [20,30,40,50]
        // src1 has a null
        let src1 = Int32Array::from_iter(field.clone(), vec![Some(1), None, Some(3)]);

        let mut g = ArrowGrowable::<Int32Type>::new(
            "v", &DataType::Int32, vec![&src0, &src1], false, 0,
        );
        // from src0 (offset=1): take logical idx 1..3 -> [30,40]
        g.extend(0, 1, 2);
        // from src1: take 0..3 -> [1, null, 3]
        g.extend(1, 0, 3);
        // from src0: take logical idx 3 -> [50]
        g.extend(0, 3, 1);

        let out = g.build().unwrap();
        let arr = out.i32().unwrap();
        let got: Vec<Option<i32>> = (0..arr.len()).map(|i| arr.get(i)).collect();
        assert_eq!(
            got,
            vec![Some(30), Some(40), Some(1), None, Some(3), Some(50)]
        );
    }

    #[test]
    fn extend_varlen_utf8_multi_source() {
        let a = Utf8Array::from_iter("s", vec![Some("a"), Some("bb"), Some("ccc")].into_iter());
        let b = Utf8Array::from_iter("s", vec![Some("dddd"), Some("e")].into_iter());
        let mut g = ArrowGrowable::<Utf8Type>::new("s", &DataType::Utf8, vec![&a, &b], false, 0);
        g.extend(0, 1, 2); // ["bb","ccc"]
        g.extend(1, 0, 1); // ["dddd"]
        g.extend(0, 0, 1); // ["a"]
        let out = g.build().unwrap();
        let arr = out.utf8().unwrap();
        let got: Vec<Option<&str>> = (0..arr.len()).map(|i| arr.get(i)).collect();
        assert_eq!(got, vec![Some("bb"), Some("ccc"), Some("dddd"), Some("a")]);
    }
}
