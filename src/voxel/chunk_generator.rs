use super::chunk::Chunk;
use super::erosion::ErosionMap;
use super::terrain;
use crossbeam_channel::{Receiver, Sender};
use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

const WORKER_COUNT: usize = 4;

pub struct GeneratedColumn {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub chunks: Vec<Chunk>,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
struct ColumnKey {
    chunk_x: i32,
    chunk_z: i32,
}

/// Manages background threads that generate terrain columns.
/// Main thread sends requests, workers generate, main thread receives results.
pub struct ChunkGenerator {
    request_sender: Sender<ColumnKey>,
    result_receiver: Receiver<GeneratedColumn>,
    pending: HashSet<ColumnKey>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl ChunkGenerator {
    pub fn new(seed: u32, erosion_map: Option<Arc<ErosionMap>>) -> Self {
        let (request_sender, request_receiver) = crossbeam_channel::unbounded::<ColumnKey>();
        let (result_sender, result_receiver) = crossbeam_channel::unbounded::<GeneratedColumn>();

        let mut workers = Vec::with_capacity(WORKER_COUNT);
        for _ in 0..WORKER_COUNT {
            let request = request_receiver.clone();
            let result = result_sender.clone();
            let e_map = erosion_map.clone();
            workers.push(thread::spawn(move || {
                while let Ok(ColumnKey { chunk_x, chunk_z }) = request.recv() {
                    let chunks = terrain::generate_column(chunk_x, chunk_z, seed, e_map.as_deref());
                    if result.send(GeneratedColumn { chunk_x, chunk_z, chunks }).is_err() {
                        break;
                    }
                }
            }));
        }

        Self {
            request_sender,
            result_receiver,
            pending: HashSet::new(),
            _workers: workers,
        }
    }

    /// Requests generation of a column if not already pending.
    pub fn request(&mut self, chunk_x: i32, chunk_z: i32) {
        let key = ColumnKey { chunk_x, chunk_z };
        if self.pending.insert(key) {
            let _ = self.request_sender.send(key);
        }
    }

    /// Drains all completed columns (non-blocking). Returns them for the caller to insert.
    pub fn receive(&mut self) -> Vec<GeneratedColumn> {
        let mut results = Vec::new();
        while let Ok(col) = self.result_receiver.try_recv() {
            self.pending.remove(&ColumnKey {
                chunk_x: col.chunk_x,
                chunk_z: col.chunk_z,
            });
            results.push(col);
        }
        results
    }

    /// Returns true if a column is currently queued or being generated.
    pub fn is_pending(&self, chunk_x: i32, chunk_z: i32) -> bool {
        self.pending.contains(&ColumnKey { chunk_x, chunk_z })
    }
}

impl Drop for ChunkGenerator {
    fn drop(&mut self) {
        for worker in self._workers.drain(..) {
            if let Err(panic) = worker.join() {
                log::error!("Chunk generator worker panicked: {panic:?}");
            }
        }
    }
}
