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
    /// Returns `None` if progress is 0% or if calculation would overflow
    /// Returns `Some(Duration::ZERO)` when complete (100%)
    pub fn estimated_remaining(&self) -> Option<std::time::Duration> {
        let (processed, total, _) = self.get_progress();

        if processed == 0 {
            return None;
        }

        if processed >= total {
            return Some(std::time::Duration::ZERO);
        }

        let elapsed = self.elapsed();

        // Use f64 for more precise division to avoid integer division truncation
        let avg_time_per_item_ns = elapsed.as_nanos() as f64 / processed as f64;
        let remaining_items = total - processed;
        let remaining_ns = avg_time_per_item_ns * remaining_items as f64;

        // Convert back to Duration, ensuring we don't overflow
        if remaining_ns > 0.0 && remaining_ns < u64::MAX as f64 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Some(std::time::Duration::from_nanos(remaining_ns as u64))
        } else {
            None
        }
    }

    /// Reset the progress tracker with a new total
    pub fn reset(&self, new_total: usize) {
        self.processed.store(0, Ordering::Relaxed);
        self.total.store(new_total, Ordering::Relaxed);
        // Reset start time for accurate timing
        // Note: This doesn't actually reset start_time, but we could add that if needed
    }

    /// Get current progress as a percentage (0.0 to 1.0)
    #[allow(dead_code)]
    pub fn get_progress_percentage(&self) -> f64 {
        let (_, _, progress) = self.get_progress();
        progress
    }

    /// Check if the operation is complete
    #[allow(dead_code)]
    pub fn is_complete(&self) -> bool {
        let (processed, total, _) = self.get_progress();
        processed >= total
    }

    /// Get remaining items count
    #[allow(dead_code)]
    pub fn remaining(&self) -> usize {
        let (processed, total, _) = self.get_progress();
        total.saturating_sub(processed)
    }
}
