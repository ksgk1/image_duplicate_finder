use itertools::Itertools;

pub struct ChunkedCombinations<'a, T> {
    pub(crate) data: &'a [T],
    pub(crate) k: usize,
    pub(crate) chunk_size: usize,
    pub(crate) start_index: usize,
}

impl<'a, T> ChunkedCombinations<'a, T> {
    pub(crate) const fn new(data: &'a [T], k: usize, chunk_size: usize) -> Self {
        ChunkedCombinations { data, k, chunk_size, start_index: 0 }
    }
}

impl<'a, T> Iterator for ChunkedCombinations<'a, T>
where
    T: Clone,
{
    type Item = Vec<Vec<&'a T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut current_chunk = Vec::with_capacity(self.chunk_size);
        let combinations_iter = self.data.iter().combinations(self.k).skip(self.start_index);

        for combination in combinations_iter.take(self.chunk_size) {
            current_chunk.push(combination);
        }

        if current_chunk.is_empty() {
            None
        } else {
            self.start_index += self.chunk_size;
            Some(current_chunk)
        }
    }
}
