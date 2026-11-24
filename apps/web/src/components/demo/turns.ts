/**
 * Turns at the page's models. Replies share a turn: any number run together.
 * The image tool's model swap takes a turn alone: it starts once every reply
 * that was running has ended, and no reply starts until it has ended (the
 * swap unloads every chat model, which would end a running reply, and a
 * reply started mid-swap would load its model beside the image model).
 */

const aborted = () => new DOMException('The operation was aborted.', 'AbortError');

/**
 * A function that waits for a turn, shared or alone, and resolves with the
 * function that ends it. Turns are given in the order they were asked for,
 * so a swap that is waiting isn't passed by replies that come after it. A
 * caller whose `signal` aborts while it waits rejects at once and never gets
 * its turn: the ones behind it move up.
 */
export function turns() {
  let sharing = 0;
  let alone = false;
  const waiting: Array<{ alone: boolean; enter: () => void }> = [];
  const admit = () => {
    for (let next = waiting[0]; next && !alone && !(next.alone && sharing > 0); next = waiting[0]) {
      waiting.shift();
      if (next.alone) alone = true;
      else sharing++;
      next.enter();
    }
  };
  return (kind: 'shared' | 'alone', signal?: AbortSignal): Promise<() => void> =>
    new Promise((resolve, reject) => {
      if (signal?.aborted) return reject(aborted());
      const entry = {
        alone: kind === 'alone',
        enter: () => {
          signal?.removeEventListener('abort', onAbort);
          let ended = false;
          resolve(() => {
            if (ended) return;
            ended = true;
            if (entry.alone) alone = false;
            else sharing--;
            admit();
          });
        },
      };
      const onAbort = () => {
        const at = waiting.indexOf(entry);
        if (at < 0) return;
        waiting.splice(at, 1);
        reject(aborted());
        admit();
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      waiting.push(entry);
      admit();
    });
}
