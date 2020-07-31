use futures::{
    executor::{
        block_on,
        ThreadPool,
    },
    future::{
        try_join_all,
        Future,
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
    let jobs = iter
        .map(|job| pool.spawn_with_handle(job))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(block_on(try_join_all(jobs))?)
}
