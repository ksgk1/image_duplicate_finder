use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

/// Lock-free progress tracker using atomic operations
#[derive(Debug)]
pub struct LockFreeProgress {
    processed: AtomicUsize,
    total: AtomicUsize,
    start_time: Instant,
}

impl LockFreeProgress {
    /// Create a new progress tracker with the given total number of items
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self { processed: AtomicUsize::new(0), total: AtomicUsize::new(total), start_time: Instant::now() }
    }

    /// Increment the processed counter by 1 and return current progress info
    /// Returns: (`processed_count`, `progress_percentage`, `elapsed_time`)
    pub fn increment(&self) -> (usize, f64, std::time::Duration) {
        self.increment_batch(1)
    }

    /// Increment the processed counter by the specified amount and return current progress info
    /// Returns: (`processed_count`, `progress_percentage`, `elapsed_time`)
    pub fn increment_batch(&self, count: usize) -> (usize, f64, std::time::Duration) {
        let processed = self.processed.fetch_add(count, Ordering::Relaxed) + count;
        let total = self.total.load(Ordering::Relaxed);
        let progress = if total > 0 { processed as f64 / total as f64 } else { 0.0 };
        let elapsed = self.start_time.elapsed();

        (processed, progress, elapsed)
    }

    /// Get current progress without incrementing
    /// Returns: (`processed_count`, `total_count`, `progress_percentage`)
    pub fn get_progress(&self) -> (usize, usize, f64) {
        let processed = self.processed.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        let progress = if total > 0 { processed as f64 / total as f64 } else { 0.0 };

        (processed, total, progress)
    }

    /// Get elapsed time since creation
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Calculate estimated time remaining based on current progress
    /// Returns `None` if progress is 0% or if the calculation overflows
    /// Returns `Some(Duration::ZERO)` when complete (100%)
    pub fn estimated_remaining(&self) -> Option<std::time::Duration> {
        let (processed, total, _) = self.get_progress();

        if processed == 0 {
            return None;
        }

        if processed >= total {
            return Some(std::time::Duration::ZERO);
        }

        let elapsed_ns = u64::try_from(self.elapsed().as_nanos()).ok()?;
        let processed = u64::try_from(processed).ok()?;
        let remaining_items = u64::try_from(total).ok()?.checked_sub(processed)?;
        let nanos_per_item = elapsed_ns / processed;
        let remaining_ns = nanos_per_item.checked_mul(remaining_items)?;
        Some(std::time::Duration::from_nanos(remaining_ns))
    }

    /// Reset the progress tracker with a new total
    pub fn reset(&self, new_total: usize) {
        self.processed.store(0, Ordering::Relaxed);
        self.total.store(new_total, Ordering::Relaxed);
        // Reset start time for accurate timing
        // Note: This doesn't actually reset start_time, but we could add that if needed
    }
}
