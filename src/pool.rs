use futures::{
    executor::{
        block_on,
        ThreadPool,
    },
    future::{
        try_join_all,
        Future,
        RemoteHandle,
    },
    task::SpawnExt,
};

pub(crate) fn pool_execute<E, I, F, V>(pool: &ThreadPool, iter: I) -> anyhow::Result<Vec<V>>
where
    I: ExactSizeIterator + Iterator<Item = F>,
    F: Future<Output = Result<V, E>> + Send + 'static,
    <F as Future>::Output: Send,
    E: Send + Sync + 'static,
    anyhow::Error: From<E>,
    V: 'static,
{
    let mut all_jobs: Vec<RemoteHandle<Result<V, E>>> = Vec::with_capacity(iter.len());

    for job in iter {
        all_jobs.push(pool.spawn_with_handle(job)?);
    }

    Ok(block_on(try_join_all(all_jobs))?)
}
