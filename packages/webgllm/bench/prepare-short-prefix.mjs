/** Prepare explicit native Chrome checks; never runs inference or reads environment files. */
import {readFile,writeFile,mkdir,rm,access} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
const repo=resolve(fileURLToPath(new URL('../../..',import.meta.url))),pkg=join(repo,'packages/webgl'),qa=join(pkg,'.short-prefix-qa');
const marker='webgl-short-prefix-native-qa-v1';
const pages=[['short-prefix-template.html','apps/checkers/__short-prefix-qa.html','<!-- Generated short-prefix native benchmark; remove after QA. -->'],['short-prefix-sdk-template.html','apps/checkers/__short-prefix-sdk-qa.html','<!-- Generated short-prefix SDK benchmark; remove after QA. -->']];
const exists=async path=>{try{await access(path);return true}catch{return false}};
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const git=(...args)=>execFileSync('git',args,{cwd:repo});
if(process.argv[2]==='--clean'){
 for(const[,file,tag]of pages){const path=join(repo,file);if(await exists(path)){if(!(await readFile(path,'utf8')).startsWith(tag))throw Error('Refusing to delete unowned page '+file);await rm(path)}}
 if(await exists(qa)){if((await readFile(join(qa,'.generated'),'utf8')).trim()!==marker)throw Error('Refusing to delete unowned QA source directory');await rm(qa,{recursive:true})}
 console.log('Removed only the owned generated pages/source snapshot. Browser model storage is unchanged.');
}else{
 const requested=process.argv[2];if(!requested||requested.startsWith('-'))throw Error('Usage: node packages/webgl/bench/prepare-short-prefix.mjs <before-commit> | --clean');
 const base=git('rev-parse','--verify',requested+'^{commit}').toString().trim();
 const paths=git('ls-tree','-r','--name-only',base,'--','packages/webgl/src').toString().trim().split('\n').filter(path=>path.endsWith('.ts')&&!path.endsWith('.test.ts')&&!path.split('/').some(part=>part.startsWith('.env')));
 const before={};
 if(await exists(qa)&&!(await exists(join(qa,'.generated')))&&!(await exists(join(qa,'before-sha.json'))))throw Error('Unowned QA directory exists');
 await mkdir(qa,{recursive:true});await writeFile(join(qa,'.gitignore'),'*\n');
 for(const path of paths){const relative=path.slice('packages/webgl/src/'.length),bytes=git('show',base+':'+path),dest=join(qa,'before',relative);before[relative]=sha(bytes);if(await exists(dest)){if(sha(await readFile(dest))!==before[relative])throw Error('Different baseline already prepared: '+relative)}else{await mkdir(resolve(dest,'..'),{recursive:true});await writeFile(dest,bytes)}}
 await writeFile(join(qa,'.generated'),marker+'\n');await writeFile(join(qa,'before-sha.json'),JSON.stringify(before,null,2)+'\n');
 const after={};for(const path of paths){after[path]=sha(await readFile(join(repo,path)))}
 const benchmarks={};for(const name of ['short-prefix-worker.ts','short-prefix-template.html','short-prefix-sdk-template.html'])benchmarks[name]=sha(await readFile(join(pkg,'bench',name)));
 const manifest={baseline:base,head:git('rev-parse','HEAD').toString().trim(),lockSha256:sha(await readFile(join(repo,'pnpm-lock.yaml'))),before,after,benchmarks};
 await writeFile(join(qa,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
 for(const[template,file,tag]of pages){const path=join(repo,file);if(await exists(path)&&!(await readFile(path,'utf8')).startsWith(tag))throw Error('Refusing to replace unowned page '+file);const source=tag+'\n'+(await readFile(join(pkg,'bench',template),'utf8')).replace('__ROOT_URL__',JSON.stringify('/@fs/'+repo+'/').replaceAll('<','\\u003c'));if(!(await exists(path))||await readFile(path,'utf8')!==source)await writeFile(path,source)}
 console.log('Prepared Chrome-only pages /__short-prefix-qa.html and /__short-prefix-sdk-qa.html. Explicit Run required; no inference started.');
}
