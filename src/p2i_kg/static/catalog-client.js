/** The hosted viewer reads a catalog snapshot; local mode uses the live API. */
export const hosted=document.documentElement.dataset.mode==='static';
let snapshot;
export async function catalog(){
 if(!snapshot){const response=await fetch(new URL('./catalog.json',import.meta.url));if(!response.ok)throw Error('카탈로그를 불러오지 못했습니다.');snapshot=await response.json();}
 return snapshot;
}
function effective(graph,id){
 const result=new Map(),seen=new Set(),pending=[[id,[]]];
 while(pending.length){const [key,path]=pending.shift();if(seen.has(key))continue;seen.add(key);
  for(const edge of graph.edges){if(edge.source!==key)continue;
   if(['uses','subconcept_of'].includes(edge.relation)){const trail=[...path,edge];if(!result.has(edge.target))result.set(edge.target,trail);pending.push([edge.target,trail]);}
   else if(edge.relation==='has_component')pending.push([edge.target,path.concat(edge)]);
  }
 }return result;
}
export async function read(url,options){
 if(!hosted){const response=await fetch(url,options);if(!response.ok){let body;try{body=await response.json()}catch{body={detail:`요청 실패 (${response.status})`}}throw Error(body.detail??'요청을 완료하지 못했습니다.')}return response.json();}
 const parsed=new URL(url,location.origin),path=parsed.pathname,params=parsed.searchParams,graph=await catalog(),nodes=new Map(graph.nodes.map(n=>[n.id,n]));
 if(options?.method&&options.method!=='GET')throw Error('실제 모델 편집과 카탈로그 변경은 로컬 P2I에서 사용할 수 있습니다.');
 if(path==='/api/kg')return graph;
 if(path==='/api/kg/search'){const q=(params.get('q')??'').toLocaleLowerCase(),kind=params.get('kind');return {nodes:graph.nodes.filter(n=>(!kind||n.kind===kind)&&[n.label,n.summary,...(n.aliases??[])].join(' ').toLocaleLowerCase().includes(q))};}
 if(path==='/api/kg/compare'){
  const left=nodes.get(params.get('left')),right=nodes.get(params.get('right'));if(left?.kind!=='model'||right?.kind!=='model')throw Error('모델 두 개를 선택하세요.');
  const a=effective(graph,left.id),b=effective(graph,right.id),select=keys=>[...keys].sort().map(k=>nodes.get(k));
  return {left,right,common:select([...a.keys()].filter(k=>b.has(k))),left_only:select([...a.keys()].filter(k=>!b.has(k))),right_only:select([...b.keys()].filter(k=>!a.has(k))),
   facets:Object.fromEntries(['has_component','trained_with','applied_to'].map(relation=>[relation,Object.fromEntries([['left',left],['right',right]].map(([side,node])=>[side,graph.edges.filter(e=>e.source===node.id&&e.relation===relation).map(e=>e.target).sort()]))])),note:'현재 카탈로그의 구성 개념을 비교합니다. 학습 성능이나 직접적인 역사적 계승을 뜻하지 않습니다.'};
 }
 if(path==='/api/kg/query'){
  const concept=params.get('concept'),objective=params.get('objective'),task=params.get('task'),family=params.get('family');
  return {matches:graph.nodes.filter(n=>n.kind==='model').flatMap(node=>{const proof=effective(graph,node.id).get(concept)??[];
   if(family&&node.family!==family||concept&&!proof.length||objective&&!graph.edges.some(e=>e.source===node.id&&e.relation==='trained_with'&&e.target===objective)||task&&!graph.edges.some(e=>e.source===node.id&&e.relation==='applied_to'&&e.target===task))return [];
   return [{node,proof:proof.map((_,i)=>i),proof_edges:proof}];})};
 }
 if(path.startsWith('/api/kg/neighborhood/')){
  const id=decodeURIComponent(path.split('/').at(-1)),limit=Math.max(1,Math.min(200,Number(params.get('limit')??40))),hops=Math.max(1,Math.min(3,Number(params.get('hops')??1))),relations=params.get('relations')?.split(',');
  if(!nodes.has(id))throw Error('노드를 찾을 수 없습니다.');
  let frontier=new Set([id]),selected=new Set([id]),truncated=false;
  for(let i=0;i<hops;i++){const next=new Set();for(const edge of graph.edges){if(relations&&!relations.includes(edge.relation))continue;if(frontier.has(edge.source)||frontier.has(edge.target)){for(const key of [edge.source,edge.target])if(!selected.has(key)){if(selected.size>=limit)truncated=true;else{selected.add(key);next.add(key);}}}}frontier=next;}
  return {nodes:[...selected].sort().map(id=>nodes.get(id)),edges:graph.edges.filter(e=>selected.has(e.source)&&selected.has(e.target)&&(!relations||relations.includes(e.relation))),hops,truncated};
 }
 if(path.startsWith('/api/kg/inspect/')){
  const node=nodes.get(decodeURIComponent(path.split('/').at(-1)));if(!node?.runtime_id)throw Error('실행 예제가 없는 모델입니다.');
  const response=await fetch(new URL('./snapshots/'+node.runtime_id+'.json',import.meta.url));if(!response.ok)throw Error('이 모델은 로컬 P2I에서 구조를 열어보세요.');return response.json();
 }
 throw Error('이 기능은 로컬 P2I에서 사용할 수 있습니다.');
}
