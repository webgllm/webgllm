/**
 * A queue for work that must never overlap, such as the image tool's model
 * swap: each task starts after the one before has ended.
 */

/** The reason an aborted signal gives, as an error. */
const abortReason = (signal: AbortSignal) =>
  signal.reason instanceof Error ? signal.reason : new DOMException('The operation was aborted.', 'AbortError');

/**
 * A function that runs each task given to it after every task given before.
 * A caller whose `signal` aborts while its task waits rejects at once, and
 * its task never runs: the queue moves straight on to the next. Once a task
 * has started, its caller waits for it to end, so the next task can't start
 * while it is still cleaning up.
 */
export function oneAtATime() {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(task: () => Promise<T>, signal?: AbortSignal): Promise<T> => {
    if (signal?.aborted) return Promise.reject(abortReason(signal));
    let started = false;
    const run = tail.then(() => {
      signal?.throwIfAborted();
      started = true;
      return task();
    });
    tail = run.catch(() => undefined);
    if (!signal) return run;
    return new Promise<T>((resolve, reject) => {
      const onAbort = () => {
        if (!started) reject(abortReason(signal));
      };
      signal.addEventListener('abort', onAbort, { once: true });
      run.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
    });
  };
}
