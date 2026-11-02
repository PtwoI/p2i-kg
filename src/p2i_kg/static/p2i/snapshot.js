/** Adapter for the unmodified P2I React viewer and genuine runtime snapshots. */
const nativeFetch=globalThis.fetch.bind(globalThis);
const id=new URLSearchParams(location.search).get('model');
try{
 if(!id||!/^[a-z0-9_-]+$/.test(id))throw Error('지도에서 모델을 선택하세요.');
 const response=await nativeFetch(new URL('../snapshots/'+id+'.json?v='+(new URLSearchParams(location.search).get('v')??'neon-v1'),import.meta.url));
 if(!response.ok)throw Error('모델 관측 결과를 불러오지 못했습니다.');
 const snapshot=await response.json();if(!snapshot.observed?.modules?.length)throw Error('P2I 관측 결과가 없는 모델입니다.');
 const ir=snapshot.observed;
 document.title=ir.metadata.name+' · P2I';
 globalThis.fetch=async(input,options)=>{
  const url=new URL(input instanceof Request?input.url:input,location.href);
  if(!url.pathname.startsWith('/api/'))return nativeFetch(input,options);
  const respond=(body,status=200)=>Promise.resolve(new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}}));
  if(options?.signal?.aborted)throw new DOMException('Aborted','AbortError');
  if(options?.method&&options.method!=='GET')return respond({detail:'웹에서는 관측된 구조를 탐색합니다. 실행·편집은 로컬 P2I를 사용하세요.'},405);
  switch(url.pathname){
   case '/api/model/manifest':return respond({},404);
   case '/api/model':return respond(ir);
   case '/api/harness':return respond({enabled:false,observed_configuration:snapshot.configuration??{}});
   case '/api/inspection':return respond(snapshot.inspection??{tensors:{},operations:{}});
   case '/api/skills':return respond({skills:[]});
   case '/api/comparison':return respond({available:false});
   default:return respond({detail:'Snapshot endpoint unavailable'},404);
  }
 };
 await import('./assets/index-DlHueU6R.js');
 // The original explorer can also serve local live sessions. This embedding is a snapshot.
 const observer=new MutationObserver(()=>{const badge=document.querySelector('header .local');if(badge&&badge.textContent!=='● P2I · 관측 결과')badge.textContent='● P2I · 관측 결과';});
 observer.observe(document.getElementById('root'),{childList:true,subtree:true});
}catch(error){const root=document.getElementById('root');root.replaceChildren();const box=document.createElement('main');box.className='fatal';box.textContent=error.message;root.append(box);}
