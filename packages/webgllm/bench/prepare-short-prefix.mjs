/** Freeze a baseline engine source snapshot for short-prefix.html; never runs inference or reads environment files.
 * Usage: node packages/webgllm/bench/prepare-short-prefix.mjs <before-commit> | --clean
 * Then serve the bench (server.mjs) and open /short-prefix.html: "before" is the snapshot, "after" this checkout. */
import {readFile,writeFile,mkdir,rm,access} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
const repo=resolve(fileURLToPath(new URL('../../..',import.meta.url))),pkg=join(repo,'packages/webgllm'),qa=join(pkg,'.short-prefix-qa');
const marker='webgl-short-prefix-native-qa-v1';
const exists=async path=>{try{await access(path);return true}catch{return false}};
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const git=(...args)=>execFileSync('git',args,{cwd:repo});
if(process.argv[2]==='--clean'){
 if(await exists(qa)){if((await readFile(join(qa,'.generated'),'utf8')).trim()!==marker)throw Error('Refusing to delete unowned QA source directory');await rm(qa,{recursive:true})}
 console.log('Removed only the owned source snapshot. Browser model storage is unchanged.');
}else{
 const requested=process.argv[2];if(!requested||requested.startsWith('-'))throw Error('Usage: node packages/webgllm/bench/prepare-short-prefix.mjs <before-commit> | --clean');
 const base=git('rev-parse','--verify',requested+'^{commit}').toString().trim();
 const source='packages/webgllm/src';
 const paths=git('ls-tree','-r','--name-only',base,'--',source).toString().trim().split('\n').filter(path=>path.endsWith('.ts')&&!path.endsWith('.test.ts')&&!path.split('/').some(part=>part.startsWith('.env')));
 const before={};
 if(await exists(qa)&&!(await exists(join(qa,'.generated')))&&!(await exists(join(qa,'before-sha.json'))))throw Error('Unowned QA directory exists');
 await mkdir(qa,{recursive:true});await writeFile(join(qa,'.gitignore'),'*\n');
 for(const path of paths){const relative=path.slice(source.length+1),bytes=git('show',base+':'+path),dest=join(qa,'before',relative);before[relative]=sha(bytes);if(await exists(dest)){if(sha(await readFile(dest))!==before[relative])throw Error('Different baseline already prepared: '+relative)}else{await mkdir(resolve(dest,'..'),{recursive:true});await writeFile(dest,bytes)}}
 // The worker imports each variant's engine and prompt modules as one bundle.
 await writeFile(join(qa,'before','runtime.ts'),'export * from "./engine.js"\nexport * from "./prompt.js"\n');
 await writeFile(join(qa,'.generated'),marker+'\n');await writeFile(join(qa,'before-sha.json'),JSON.stringify(before,null,2)+'\n');
 const after={};for(const path of git('ls-files','--','packages/webgllm/src').toString().trim().split('\n').filter(path=>path.endsWith('.ts')&&!path.endsWith('.test.ts'))){after[path]=sha(await readFile(join(repo,path)))}
 const benchmarks={};for(const name of ['short-prefix-worker.ts','short-prefix-runtime.ts','short-prefix.html'])benchmarks[name]=sha(await readFile(join(pkg,'bench',name)));
 const manifest={baseline:base,head:git('rev-parse','HEAD').toString().trim(),lockSha256:sha(await readFile(join(repo,'pnpm-lock.yaml'))),before,after,benchmarks};
 await writeFile(join(qa,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
 console.log('Prepared .short-prefix-qa/before. Open /short-prefix.html on the bench server; explicit Run required, no inference started.');
}
