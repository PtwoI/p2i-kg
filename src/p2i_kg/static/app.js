import {mindmap,neighborhood,connectionLayout,edgePath,NODE_WIDTH,NODE_HEIGHT} from './layout.js?v=20261103-dependency-flow';
import {read,hosted} from './catalog-client.js?v=neon-v1';
const $=selector=>document.querySelector(selector);
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let graph,nodes,selected='model:transformer',zoom=1,searchTimer,mode='overview',world={width:1400,height:720},camera={x:0,y:0,width:1400,height:720},pan=null;
const sessions=new Map();
const initialModel=new URLSearchParams(location.search).get('model');if(initialModel)selected=initialModel.startsWith('model:')?initialModel:'model:'+initialModel;
let mapRequest=0;
const relations={extends:'확장',alternative_to:'대안',documented_in:'대표 문헌',uses:'사용 개념',has_component:'구성 요소',subconcept_of:'상위 개념',trained_with:'학습 목적',applied_to:'적용 과제'};
const kinds={model:'모델',concept:'개념',paper:'논문',component:'구성 요소',objective:'학습 목적',task:'과제'};
const bases={documented:'문헌 관계',structural:'구조적 관계',pedagogical:'교육적 연결'};
const families={feedforward:['Feedforward','#c6ff6b'],sequence:['Sequence','#b99aff'],vision:['Vision','#5ccfff'],attention:['Attention','#39f6cf'],generative:['Generative','#ffba69'],graph:['Graph','#d8ec6a'],multimodal:['Multimodal','#ff87ca'],set:['Set models','#73d9c7'],'3d':['3D vision','#8db9ff'],'self-supervised':['Self-supervised','#e8a3ff'],'state-space':['State space','#d5b6ff'],'reinforcement-learning':['Reinforcement learning','#ffcf70']};
function select(id,{closePanels=true}={}){
 if(!nodes.has(id))return;selected=id;
 const previousMode=mode;if(nodes.get(id).kind!=='model')mode='neighborhood';
 if(nodes.get(id).kind==='model'&&[...$('#model-select').options].some(option=>option.value===id))$('#model-select').value=id;
 if(closePanels)document.querySelectorAll('dialog[open]').forEach(dialog=>dialog.close());$('#search-results').innerHTML='';
 renderMap(zoom===1||mode!==previousMode||mode==='neighborhood');renderDetail();
}
function chip(id){const node=nodes.get(id);return `<button class="chip" data-node="${esc(id)}">${esc(node?.label??id)}</button>`}
function applyCamera(){ $('#map').setAttribute('viewBox',`${camera.x} ${camera.y} ${camera.width} ${camera.height}`);$('#zoom-reset').textContent=zoom===1?'전체':Math.round(zoom*100)+'%'; }
function fit(){zoom=1;camera={x:0,y:0,width:world.width,height:world.height};applyCamera();}
function scaleMap(factor){const next=Math.max(.3,Math.min(16,zoom*factor)),ratio=zoom/next;camera={x:camera.x+camera.width*(1-ratio)/2,y:camera.y+camera.height*(1-ratio)/2,width:camera.width*ratio,height:camera.height*ratio};zoom=next;applyCamera();}
function wrapLabel(label,max=24){
 const chars=Array.from(label);if(chars.length<=max)return [label];
 let split=label.lastIndexOf(' ',max);if(split<max/2)split=max;
 const tail=Array.from(chars.slice(split).join('').trim());
 return [chars.slice(0,split).join(''),tail.length>max?tail.slice(0,max-1).join('')+'…':tail.join('')];
}
async function renderMap(reset=true){
 const request=++mapRequest,id=selected,family=$('#family').value;
 $('#overview').setAttribute('aria-pressed',String(mode==='overview'));$('#neighborhood').setAttribute('aria-pressed',String(mode==='neighborhood'));$('.focus-controls').hidden=mode==='overview';
 try{
  const hops=Number($('#hops').value),relation=$('#relation').value;
  const result=mode==='neighborhood'?neighborhood(graph,id,hops,relation):null;
  const layout=mode==='overview'?mindmap(graph,family):connectionLayout(result,id);
  if(request!==mapRequest)return;
  const items=new Map(layout.items.map(item=>[item.key,item]));
  world={width:layout.width,height:layout.height};
  const adjacent=new Set(graph.edges.filter(e=>e.source===id||e.target===id).flatMap(e=>[e.source,e.target]));
  let svg='<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L7,4 L1,7" fill="none" stroke="context-stroke" stroke-width="1.5"/></marker></defs>';
  for(const guide of layout.guides)svg+=`<g class="direction-guide ${guide.side}" transform="translate(${guide.x},${guide.y})"><text text-anchor="middle">${esc(guide.label)}</text><path d="M-65,14 H65"/></g>`;
  for(const branch of layout.branches){
   const a=items.get(branch.source),b=items.get(branch.target),color=families[b.node.family]?.[1]??'#9cadc5';
   svg+=`<path class="mindmap-branch" style="--branch-color:${color}" d="${edgePath(a,b)}"/>`;
  }
  const links=mode==='overview'?graph.edges.filter(e=>[e.source,e.target].includes(id)&&items.has(e.source)&&items.has(e.target)).map(edge=>({edge,source:edge.source,target:edge.target})):layout.edges;
  const parallel=new Map();
  for(const {edge,source,target}of links){
   const a=items.get(source),b=items.get(target);if(!a||!b)continue;
   const pair=source+'|'+target,count=parallel.get(pair)??0;parallel.set(pair,count+1);
   const path=edgePath(a,b,count*7),index=graph.edges.indexOf(edge),active=[edge.source,edge.target].includes(id);
   const label=`${nodes.get(edge.source)?.label} → ${nodes.get(edge.target)?.label} · ${relations[edge.relation]??edge.relation}`;
   const labelX=(a.x+b.x)/2,labelY=(a.y+b.y)/2-9+count*7;
   svg+=`<g class="relation-edge ${active?'is-active':''}" data-edge="${index}" role="button" tabindex="0" aria-label="${esc(label)}"><title>${esc(label)}${edge.explanation?' — '+esc(edge.explanation):''}</title><path class="edge-hit" d="${path}"/><path class="map-edge ${edge.basis==='pedagogical'||edge.relation==='alternative_to'?'dotted':''} ${active?'active':'faded'}" d="${path}" marker-end="url(#arrow)"/><text class="edge-label" x="${labelX}" y="${labelY}" text-anchor="middle">${esc(relations[edge.relation]??edge.relation)}</text></g>`;
  }
  for(const item of layout.items){
   const {node,x,y}=item,accent=families[node.family]?.[1]??'#9cadc5';
   if(node.kind==='root'){
    svg+=`<g class="mindmap-root" data-map-home role="button" tabindex="0" aria-label="모든 모델 지도"><rect x="${x-96}" y="${y-42}" width="192" height="84" rx="24"/><text x="${x}" y="${y-13}" text-anchor="middle" class="root-eyebrow">P2I / ARCHITECTURE</text><text x="${x}" y="${y+10}" text-anchor="middle" class="root-label">Neural networks</text><text x="${x}" y="${y+29}" text-anchor="middle" class="root-count">${node.count} models · ${layout.items.filter(n=>n.node.kind==='family').length} families</text></g>`;continue;
   }
   if(node.kind==='family'){
    svg+=`<g class="family-node" style="--node-accent:${accent}" data-family="${esc(node.family)}" role="button" tabindex="0" aria-label="${esc(families[node.family]?.[0]??node.label)} 계열만 보기" transform="translate(${x-96},${y-28})"><rect width="192" height="56" rx="18"/><text x="96" y="24" text-anchor="middle">${esc(families[node.family]?.[0]??node.label)}</text><text class="family-count" x="96" y="43" text-anchor="middle">${node.count} models · 계열 보기</text></g>`;continue;
   }
   const label=wrapLabel(node.label),meta=node.kind==='model'?`${families[node.family]?.[0]??node.family??'MODEL'}${node.reference_year?' · '+node.reference_year:''}`:kinds[node.kind]??node.kind;
   svg+=`<g class="model-node kind-${node.kind} ${node.id===id?'selected':adjacent.has(node.id)?'related':''}" style="--node-accent:${accent}" data-node="${esc(node.id)}" data-placement="${esc(item.key)}" role="button" tabindex="0" aria-label="선택 ${esc(node.label)}${item.duplicate?' · 양방향 연결로 양쪽에 표시':''}" transform="translate(${x-NODE_WIDTH/2},${y-NODE_HEIGHT/2})"><title>${esc(node.label)}</title><rect width="${NODE_WIDTH}" height="${NODE_HEIGHT}" rx="12"/><rect class="node-stripe" x="0" y="16" width="3" height="36" rx="1"/><text x="14" y="18" class="node-index">${esc(meta)}</text><text x="14" y="${label.length===1?45:37}" class="node-label">${label.map((line,i)=>`<tspan x="14" dy="${i?17:0}">${esc(line)}</tspan>`).join('')}</text>${item.duplicate?'<circle class="bidirectional-dot" cx="177" cy="15" r="3"/>':''}</g>`;
  }
  $('#map').innerHTML=svg;$('#map').classList.toggle('mindmap',mode==='overview');if(reset)fit();else applyCamera();
  $('#global-error').textContent='';
  $('#map-message').textContent=mode==='overview'?`${layout.modelCount}개 모델 · 계열별 마인드맵 · 선택한 모델의 실제 관계를 화살표로 표시`:`${result.nodes.length}개 노드 · ${result.edges.length}개 관계 · 기반·구성 ${layout.incomingCount} / 활용·확장 ${layout.outgoingCount} · ${hops}단계 전체 표시`;
  $('#canvas-hint').textContent=mode==='overview'?'계열을 눌러 집중 보기 · 드래그 이동 · 휠 / 두 손가락 확대':'기반·구성은 왼쪽 · 활용·확장은 오른쪽 · 화살표는 저장된 관계 방향';
  $('#family-legend').innerHTML=mode==='overview'?Object.entries(families).filter(([key])=>(family==='all'||family===key)&&graph.nodes.some(n=>n.kind==='model'&&n.family===key)).map(([,value])=>`<span><i style="--node-accent:${value[1]}"></i>${value[0]}</span>`).join(''):'기반 · 구성 / 선택한 항목 / 활용 · 확장';
 }catch(error){if(request===mapRequest)$('#global-error').textContent=error.message}
}

function showEdge(index){
 const edge=graph.edges[index];if(!edge)return;
 $('#detail').innerHTML=`<span class="detail-type">${esc(relations[edge.relation])} / ${esc(bases[edge.basis])}</span><h2>연결의 의미</h2><p class="edge-direction">${esc(nodes.get(edge.source).label)} → ${esc(nodes.get(edge.target).label)}</p><p>${esc(edge.explanation)}</p>${edge.locator?`<p class="muted">${esc(edge.locator)}</p>`:''}<h3>연결된 항목</h3><div class="chips">${chip(edge.source)}${chip(edge.target)}</div><h3>근거 문헌</h3><div class="chips">${edge.evidence.map(chip).join('')}</div>${nodes.get(edge.source).kind==='model'&&nodes.get(edge.target).kind==='model'?`<button class="secondary" data-compare-pair="${esc(edge.source)},${esc(edge.target)}">두 모델 비교</button>`:''}`;
}
function renderDetail(){
 const node=nodes.get(selected);if(!node)return;
 const outgoing=graph.edges.filter(edge=>edge.source===node.id),incoming=graph.edges.filter(edge=>edge.target===node.id);
 const papers=node.kind==='paper'?[node]:outgoing.filter(edge=>edge.relation==='documented_in').map(edge=>nodes.get(edge.target));
 const concepts=outgoing.filter(edge=>edge.relation==='uses');
 const connections=[...outgoing,...incoming].filter(edge=>['extends','alternative_to'].includes(edge.relation));
 let html=`<span class="detail-type">${esc(node.kind)} / ${esc(node.family??'KNOWLEDGE')}</span><h2>${esc(node.label)}</h2>${node.reference_year?`<div class="detail-year">${node.reference_year}</div>`:''}<p>${esc(node.summary)}</p>`;
 html+=`<div class="detail-graph-actions"><button class="primary" id="expand-connections">연결 펼치기</button>${node.kind==='model'?'<button class="secondary" id="compare-selected">이 모델 비교</button>':''}</div>`;
 if(node.kind==='model'){
  html+=`<section class="p2i-entry"><span class="eyebrow">MODEL / P2I</span><h3>P2I 구조 탐색</h3><p>${esc(node.runtime_note??'실행 예제의 모듈과 관측 연산을 탐색합니다.')}</p><button class="primary p2i-open" id="open-p2i">P2I로 보기 <span>↗</span></button><button class="secondary" id="inspect-model">모듈 연결 펼치기</button><div id="bindings"></div>`;
  if(!hosted)html+=`<div class="runtime-actions"><button class="secondary" id="launch">실행·편집 세션 열기 ↗</button><a class="secondary" href="/api/kg/models/${encodeURIComponent(node.id)}/checkpoint">체크포인트 ↓</a></div><div id="runtime-status" aria-live="polite"></div>`;
  html+='</section>';
 }
 if(concepts.length)html+=`<h3>구성 개념</h3><div class="chips">${concepts.map(edge=>chip(edge.target)).join('')}</div>`;
 if(node.kind==='concept')html+=`<button class="secondary" id="find-models">이 개념을 사용하는 모델 찾기</button><div id="reasoned-models"></div>`;
 if(node.kind==='concept')html+=`<h3>이 개념을 사용하는 모델</h3><div class="chips">${incoming.filter(edge=>edge.relation==='uses').map(edge=>chip(edge.source)).join('')}</div>`;
 if(connections.length)html+=`<h3>다른 모델과의 관계</h3>${connections.map(edge=>`<div class="relation-item"><button data-node="${esc(edge.source===node.id?edge.target:edge.source)}">${esc(nodes.get(edge.source).label)} → ${esc(nodes.get(edge.target).label)} · ${esc(relations[edge.relation])}</button><small>${esc(bases[edge.basis])} · ${esc(edge.explanation)}</small>${edge.evidence.map(id=>`<button data-node="${esc(id)}">근거 문헌 ↗</button>`).join('')}</div>`).join('')}`;
 for(const relation of ['has_component','subconcept_of','trained_with','applied_to']){
  const links=outgoing.filter(edge=>edge.relation===relation);
  if(links.length)html+=`<h3>${esc(relations[relation])}</h3><div class="chips">${links.map(edge=>chip(edge.target)).join('')}</div>`;
 }
 if(node.kind!=='model'&&node.kind!=='paper'&&incoming.length)html+=`<h3>연결된 항목</h3><div class="chips">${[...new Set(incoming.map(edge=>edge.source))].map(chip).join('')}</div>`;
 const evidence=[...new Set([...outgoing,...incoming].flatMap(edge=>edge.evidence))];
 if(!papers.length&&evidence.length)html+=`<h3>관계의 근거 문헌</h3><div class="chips">${evidence.map(chip).join('')}</div>`;
 if(papers.length)html+=`<h3>대표 문헌</h3>${papers.map(paper=>`<a class="reference" href="${esc(paper.url)}" target="_blank" rel="noopener noreferrer">${esc(paper.label)} ↗<small>${esc(paper.authors.join(', '))} · ${paper.reference_year}</small></a>`).join('')}`;
 $('#detail').innerHTML=html;$('#detail').scrollTop=0;
 if(!hosted&&node.runtime_id&&sessions.has(node.id))showSession(node.id,sessions.get(node.id));
}
function showSession(id,session){
 if(selected!==id)return;
 $('#runtime-status').innerHTML=`세션이 준비되었습니다.<a class="session-link" href="${esc(session.url)}" target="_blank" rel="noopener noreferrer">P2I 모델 탐색기 열기 ↗</a><button class="close-session" id="close-session">이 세션 닫기</button>`;
}
async function launch(){
 const id=selected,button=$('#launch');button.disabled=true;$('#runtime-status').textContent='모델 구조를 준비하고 있습니다…';
 try{const session=await read(`/api/kg/runtime/${encodeURIComponent(id)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});sessions.set(id,session);showSession(id,session)}
 catch(error){if(selected===id){$('#runtime-status').classList.add('error');$('#runtime-status').textContent=error.message}}
 finally{if(selected===id&&$('#launch'))$('#launch').disabled=false}
}
async function compare(){
 const a=$('#compare-left').value,b=$('#compare-right').value;
 try{const result=await read(`/api/kg/compare?left=${encodeURIComponent(a)}&right=${encodeURIComponent(b)}`);
  $('#comparison').innerHTML=`<div class="compare-columns">${[['공통 개념',result.common],[result.left.label+'에만 연결',result.left_only],[result.right.label+'에만 연결',result.right_only]].map(([title,items])=>`<div><h3>${esc(title)}</h3><div class="chips">${items.length?items.map(node=>chip(node.id)).join(''):'<span class="muted">현재 그래프에 연결된 개념이 없습니다.</span>'}</div></div>`).join('')}</div><p class="compare-note">${esc(result.note)}</p>${Object.entries(result.facets??{}).map(([relation,side])=>`<div class="facet-row"><strong>${esc(relations[relation])}</strong><div>${side.left.map(chip).join('')}</div><div>${side.right.map(chip).join('')}</div></div>`).join('')}`;
 }catch(error){$('#global-error').textContent=error.message}
}
document.addEventListener('click',async event=>{
 if(pan?.moved){pan=null;return}
 const category=event.target.closest('[data-family]');if(category){$('#family').value=category.dataset.family;$('#family').dispatchEvent(new Event('change'));return}
 if(event.target.closest('[data-map-home]')){$('#family').value='all';$('#family').dispatchEvent(new Event('change'));return}
 const target=event.target.closest('[data-node]');if(target){select(target.dataset.node);return;}
 const edge=event.target.closest('[data-edge]');if(edge){showEdge(Number(edge.dataset.edge));return}
 const panel=event.target.closest('[data-panel]');if(panel){$('#'+panel.dataset.panel).showModal();return}
 if(event.target.closest('[data-close]')){event.target.closest('dialog').close();return}
 const pair=event.target.closest('[data-compare-pair]');if(pair){const [left,right]=pair.dataset.comparePair.split(',');$('#compare-left').value=left;$('#compare-right').value=right;await compare();$('#compare-panel').showModal();return}
 if(event.target.id==='expand-connections'){mode='neighborhood';renderMap();return}
 if(event.target.id==='compare-selected'){$('#compare-left').value=selected;await compare();$('#compare-panel').showModal();return}
 if(event.target.closest('#open-p2i')){const node=nodes.get(selected);if(!hosted){await launch();if(!sessions.has(node.id))return;}const url=hosted?new URL('./p2i/?model='+encodeURIComponent(node.runtime_id)+'&v='+(document.documentElement.dataset.build??graph.revision),import.meta.url):new URL(sessions.get(node.id).url);$('#p2i-title').textContent=node.label;$('#p2i-scope').textContent=node.runtime_note??'';$('#p2i-tab').href=url.href;$('#p2i-frame').src=url.href;$('#p2i-panel').showModal();return}
 if(event.target.id==='launch')launch();
 if(event.target.id==='find-models'){
  const id=selected;try{const result=await read(`/api/kg/query?concept=${encodeURIComponent(id)}`);if(selected===id)$('#reasoned-models').innerHTML=result.matches.map(match=>`<div>${chip(match.node.id)} <details><summary>${match.proof.length}단계 연결 보기</summary>${match.proof_edges.map(edge=>`<p>${esc(nodes.get(edge.source)?.label)} → ${esc(relations[edge.relation])} → ${esc(nodes.get(edge.target)?.label)}</p><div class="chips">${edge.evidence.map(chip).join('')}</div>`).join('')}</details></div>`).join('')||'<p>연결된 모델이 없습니다.</p>'}catch(error){$('#global-error').textContent=error.message}
 }
 if(event.target.id==='inspect-model'){
  const id=selected;event.target.disabled=true;$('#bindings').textContent='실제 모듈을 추적하고 있습니다…';
  try{const result=await read(`/api/kg/inspect/${encodeURIComponent(id)}`);if(selected===id)$('#bindings').innerHTML=`<p>${result.modules}개 모듈 · ${result.operations}개 관측 연산</p>`+result.bindings.map(binding=>`<div class="binding">${chip(binding.ontology_id)}<code>${esc(binding.module_path)}</code><small>${esc(binding.module_type)} · ${binding.parameter_count.toLocaleString()} parameters${binding.editable?' · 편집 가능':''}</small></div>`).join('')+`<p class="muted">${hosted?'사전에 추적한 예제 구조':'현재 예제 구조'} · 무작위 초기 가중치</p><div class="module-tree">${result.architecture.nodes.map(node=>`<details style="--depth:${Math.min((node.qualified_name??'').split('.').length-1,5)}"><summary>${esc(node.qualified_name||'(root)')} <code>${esc(node.module_type)}</code></summary><pre>${esc(JSON.stringify(node.parameters??{},null,2))}</pre></details>`).join('')}</div>`}catch(error){if(selected===id)$('#bindings').textContent=error.message}
  finally{if(selected===id&&$('#inspect-model'))$('#inspect-model').disabled=false}
 }
 if(event.target.id==='close-session'){
  const id=selected;try{await read(`/api/kg/runtime/${encodeURIComponent(id)}`,{method:'DELETE'});sessions.delete(id);renderDetail()}catch(error){$('#runtime-status').textContent=error.message}
 }
});
document.addEventListener('keydown',event=>{if((event.key==='Enter'||event.key===' ')&&event.target.closest('[data-family],[data-map-home]')){event.preventDefault();event.target.closest('[data-family],[data-map-home]').dispatchEvent(new MouseEvent('click',{bubbles:true}));return;}const node=event.target.closest('[data-node]'),edge=event.target.closest('[data-edge]');if((node||edge)&&(event.key==='Enter'||event.key===' ')){event.preventDefault();if(node)select(node.dataset.node);else showEdge(Number(edge.dataset.edge))}});
$('#family').addEventListener('change',()=>{
 const family=$('#family').value;$('#model-select').innerHTML=graph.nodes.filter(n=>n.kind==='model'&&(family==='all'||n.family===family)).map(n=>`<option value="${esc(n.id)}">${esc(n.label)}</option>`).join('');
 mode='overview';if(![...$('#model-select').options].some(option=>option.value===selected)&&$('#model-select').value)select($('#model-select').value);else{$('#model-select').value=selected;renderMap();}
});
$('#model-select').addEventListener('change',()=>select($('#model-select').value));
$('#hops').addEventListener('change',()=>renderMap());$('#relation').addEventListener('change',()=>renderMap());
function renderLibrary(){const kind=$('#library-kind').value;$('#concepts').innerHTML=graph.nodes.filter(node=>node.kind===kind).map(node=>`<button class="concept-tile" data-node="${esc(node.id)}"><strong>${esc(node.label)}</strong><p>${esc(node.summary)}</p></button>`).join('')}
$('#library-kind').addEventListener('change',renderLibrary);
$('#patch-file').addEventListener('change',async event=>{
 const file=event.target.files?.[0];if(!file)return;
 try{const patch=JSON.parse(await file.text());const result=await read('/api/kg/extend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_revision:graph.revision,patch})});$('#patch-status').textContent=`추가 완료: ${result.added_nodes}개 노드 · ${result.added_edges}개 관계`;await loadGraph()}
 catch(error){$('#patch-status').textContent=error.message}finally{event.target.value=''}
});
$('#overview').addEventListener('click',()=>{mode='overview';renderMap()});
$('#neighborhood').addEventListener('click',()=>{mode='neighborhood';renderMap()});
$('#zoom-in').addEventListener('click',()=>scaleMap(1.3));
$('#zoom-out').addEventListener('click',()=>scaleMap(1/1.3));
$('#zoom-reset').addEventListener('click',fit);
$('#collapse-detail').addEventListener('click',()=>{const collapsed=$('.inspector').classList.toggle('collapsed');$('.workspace').classList.toggle('detail-collapsed',collapsed);$('#collapse-detail').textContent=collapsed?'펼치기':'접기';$('#collapse-detail').setAttribute('aria-expanded',String(!collapsed));$('#collapse-detail').setAttribute('aria-label',collapsed?'상세 패널 펼치기':'상세 패널 접기')});
const pointers=new Map();let pinch;
function mapPoint(event){const point=$('#map').createSVGPoint();point.x=event.clientX;point.y=event.clientY;return point.matrixTransform($('#map').getScreenCTM().inverse())}
$('#map').addEventListener('pointerdown',event=>{if(event.button!==0)return;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});pan={x:event.clientX,y:event.clientY,moved:false,point:mapPoint(event),camera:{...camera}};if(pointers.size===2){const [a,b]=[...pointers.values()];pinch={distance:Math.hypot(a.x-b.x,a.y-b.y),zoom};pan.moved=true;}});
$('#map').addEventListener('pointermove',event=>{if(!pointers.has(event.pointerId)||!pan)return;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});if(pointers.size===2&&pinch){const[a,b]=[...pointers.values()];scaleMap(pinch.zoom*Math.hypot(a.x-b.x,a.y-b.y)/pinch.distance/zoom);pan.moved=true;return}if(Math.hypot(event.clientX-pan.x,event.clientY-pan.y)<5&&!pan.moved)return;pan.moved=true;$('#map').setPointerCapture(event.pointerId);$('#map').classList.add('dragging');const point=mapPoint(event);camera.x+=pan.point.x-point.x;camera.y+=pan.point.y-point.y;applyCamera();});
function endPointer(event){pointers.delete(event.pointerId);pinch=null;$('#map').classList.remove('dragging');if(!pan?.moved){pan=null}else if(pointers.size){pan=null}else{setTimeout(()=>{pan=null},0)} }
$('#map').addEventListener('pointerup',endPointer);$('#map').addEventListener('pointercancel',event=>{endPointer(event);pan=null});
$('#map').addEventListener('wheel',event=>{event.preventDefault();scaleMap(event.deltaY<0?1.12:1/1.12)},{passive:false});
$('#map').addEventListener('keydown',event=>{if(event.target!==$('#map'))return;const delta=camera.width*.08;if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();if(event.key==='ArrowLeft')camera.x-=delta;if(event.key==='ArrowRight')camera.x+=delta;if(event.key==='ArrowUp')camera.y-=delta;if(event.key==='ArrowDown')camera.y+=delta;applyCamera()}else if(event.key==='+'||event.key==='=')scaleMap(1.3);else if(event.key==='-')scaleMap(1/1.3);else if(event.key==='Home')fit()});
$('#compare').addEventListener('click',compare);
$('#search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(async()=>{
 const q=$('#search').value;if(!q.trim()){$('#search-results').innerHTML='';return;}
 try{const result=await read(`/api/kg/search?q=${encodeURIComponent(q)}`);if($('#search').value!==q)return;$('#search-results').innerHTML=result.nodes.length?result.nodes.map(node=>`<button data-node="${esc(node.id)}">${esc(node.label)} <small>· ${esc(node.kind)}</small></button>`).join(''):'<span class="muted">검색 결과가 없습니다.</span>'}catch(error){$('#global-error').textContent=error.message}
},160)});
async function loadGraph(){
 graph=await read('/api/kg');nodes=new Map(graph.nodes.map(node=>[node.id,node]));
 const models=graph.nodes.filter(node=>node.kind==='model');
 for(const key of [...new Set(models.map(n=>n.family))])if(![...$('#family').options].some(option=>option.value===key)){const option=document.createElement('option');option.value=key;option.textContent=families[key]?.[0]??key;$('#family').append(option);}
 $('#stats').textContent=`${models.length}개 모델 · ${graph.nodes.length}개 노드 · ${graph.edges.length}개 관계`;

 $('#revision').textContent=`CATALOG ${graph.revision.slice(0,7)}`;
 for(const selector of ['#compare-left','#compare-right','#model-select'])$(selector).innerHTML=models.map(node=>`<option value="${esc(node.id)}">${esc(node.label)}</option>`).join('');
 $('#compare-left').value='model:bert';$('#compare-right').value='model:gpt2';
 if(!nodes.has(selected))selected=models[0]?.id;$('#model-select').value=selected;
 if(hosted){$('.extend-card').hidden=true;$('.snapshot-note').textContent='현재 카탈로그의 스냅샷입니다. 모든 모델은 P2I에서 모듈·연산·텐서 연결을 탐색할 수 있습니다. 무작위 초기화 예제의 관측 결과입니다.'}renderLibrary();select(selected,{closePanels:false});await compare();
}
try{await loadGraph()}catch(error){$('#global-error').textContent=error.message}

$('#p2i-panel').addEventListener('close',()=>{$('#p2i-frame').removeAttribute('src')});

