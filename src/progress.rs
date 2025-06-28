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

    /// Increment the processed counter and return current progress info
    /// Returns: (`processed_count`, `progress_percentage`, `elapsed_time`)
    pub fn increment(&self) -> (usize, f64, std::time::Duration) {
        let processed = self.processed.fetch_add(1, Ordering::Relaxed) + 1;
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

    /// # Panics
    /// Will panic if number of items processed exceeds `u32::MAX`
    /// Calculate estimated time remaining based on current progress
    pub fn estimated_remaining(&self) -> Option<std::time::Duration> {
        let (processed, total, _) = self.get_progress();

        if processed == 0 || processed >= total {
            return None;
        }

        let elapsed = self.elapsed();
        let avg_time_per_item = elapsed / u32::try_from(processed).expect("Processed items exceeds u32::MAX");
        let remaining_items = total - processed;

        Some(avg_time_per_item * u32::try_from(remaining_items).expect("Remaining items exceeds u32::MAX"))
    }

    /// Reset the progress tracker with a new total
    pub fn reset(&self, new_total: usize) {
        self.processed.store(0, Ordering::Relaxed);
        self.total.store(new_total, Ordering::Relaxed);
    }
}
