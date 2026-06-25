use common_error::DaftResult;
use daft_micropartition::MicroPartitionRef;
use daft_recordbatch::RecordBatch;

/// Daft RecordBatches -> arrow-rs RecordBatches (uses Daft's TryFrom impl).
pub fn daft_batches_to_arrow(batches: &[RecordBatch]) -> DaftResult<Vec<arrow_array::RecordBatch>> {
    batches
        .iter()
        .map(|b| Ok(arrow_array::RecordBatch::try_from(b.clone())?))
        .collect()
}

/// Flatten a source's MicroPartitions into arrow-rs RecordBatches (arrow_array v59).
pub fn source_to_arrow(mps: &[MicroPartitionRef]) -> DaftResult<Vec<arrow_array::RecordBatch>> {
    let mut out = Vec::new();
    for mp in mps {
        out.extend(daft_batches_to_arrow(mp.record_batches())?);
    }
    Ok(out)
}

/// arrow-rs RecordBatches -> Daft RecordBatches (uses Daft's TryFrom impl).
pub fn arrow_to_daft_batches(
    arrow_batches: Vec<arrow_array::RecordBatch>,
) -> DaftResult<Vec<RecordBatch>> {
    arrow_batches
        .iter()
        .map(|b| Ok(RecordBatch::try_from(b)?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daft_core::prelude::*;
    use daft_recordbatch::RecordBatch;

    fn sample_batch() -> RecordBatch {
        let a = Int64Array::from_vec("a", vec![1i64, 2, 3]).into_series();
        let b = Utf8Array::from_slice("b", vec!["x", "y", "z"].as_slice()).into_series();
        RecordBatch::from_nonempty_columns(vec![a, b]).unwrap()
    }

    #[test]
    fn round_trips_daft_arrow_daft() {
        let original = sample_batch();
        let arrow = daft_batches_to_arrow(&[original.clone()]).unwrap();
        assert_eq!(arrow.len(), 1);
        assert_eq!(arrow[0].num_rows(), 3);
        let back = arrow_to_daft_batches(arrow).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].len(), original.len());
        assert_eq!(back[0].schema, original.schema);
    }
}
