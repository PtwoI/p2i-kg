import {read,hosted} from './catalog-client.js';
const $=selector=>document.querySelector(selector);
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let graph,nodes,selected='model:transformer',zoom=1,searchTimer,mode='overview',world={width:1400,height:720},camera={x:0,y:0,width:1400,height:720},pan=null;
const sessions=new Map();
let mapRequest=0;
const relations={extends:'확장',alternative_to:'대안',documented_in:'대표 문헌',uses:'사용 개념',has_component:'구성 요소',subconcept_of:'상위 개념',trained_with:'학습 목적',applied_to:'적용 과제'};
const kinds={model:'모델',concept:'개념',paper:'논문',component:'구성 요소',objective:'학습 목적',task:'과제'};
const bases={documented:'문헌 관계',structural:'구조적 관계',pedagogical:'교육적 연결'};
const families={feedforward:['Feedforward','#8b6b46'],sequence:['Sequence','#85619a'],vision:['Vision','#317d98'],attention:['Attention','#18745f'],generative:['Generative','#bf8554'],graph:['Graph','#79883a'],multimodal:['Multimodal','#a76078']};
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
function scaleMap(factor){const next=Math.max(.6,Math.min(5,zoom*factor)),ratio=zoom/next;camera={x:camera.x+camera.width*(1-ratio)/2,y:camera.y+camera.height*(1-ratio)/2,width:camera.width*ratio,height:camera.height*ratio};zoom=next;applyCamera();}
async function renderMap(reset=true){
 const request=++mapRequest,id=selected;
 $('#overview').setAttribute('aria-pressed',String(mode==='overview'));$('#neighborhood').setAttribute('aria-pressed',String(mode==='neighborhood'));$('.focus-controls').hidden=mode==='overview';
 try{
  const hops=$('#hops').value,relation=$('#relation').value,family=$('#family').value;
  let result,positions=new Map(),width,height;
  if(mode==='overview'){
   const models=graph.nodes.filter(n=>n.kind==='model'&&(family==='all'||n.family===family)).sort((a,b)=>(a.reference_year??0)-(b.reference_year??0)||a.id.localeCompare(b.id));
   const known=new Set(models.map(n=>n.id));result={nodes:models,edges:graph.edges.filter(e=>known.has(e.source)&&known.has(e.target)&&['extends','alternative_to'].includes(e.relation))};
   const columns=Math.min(6,Math.max(2,Math.ceil(Math.sqrt(models.length*1.5)))),rows=Math.ceil(models.length/columns);
   width=columns*226+70;height=rows*132+85;
   models.forEach((node,i)=>positions.set(node.id,[145+(i%columns)*226,85+Math.floor(i/columns)*132]));
  }else{
   result=await read(`/api/kg/neighborhood/${encodeURIComponent(id)}?hops=${hops}&limit=40${relation?'&relations='+relation:''}`);if(request!==mapRequest)return;
   const neighbors=result.nodes.filter(n=>n.id!==id).sort((a,b)=>a.kind.localeCompare(b.kind)||a.label.localeCompare(b.label));
   const columns=neighbors.length>14?4:3,rows=Math.max(2,Math.ceil(neighbors.length/columns));width=390+columns*226;height=rows*120+70;
   positions.set(id,[140,height/2]);neighbors.forEach((node,i)=>positions.set(node.id,[405+Math.floor(i/rows)*226,70+(i%rows)*120]));
  }
  if(request!==mapRequest)return;
  world={width,height};const adjacent=new Set(result.edges.filter(e=>e.source===id||e.target===id).flatMap(e=>[e.source,e.target]));
  let svg='<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#809d8d"/></marker></defs>';
  if(mode==='overview')svg+=`<text x="45" y="25" class="map-caption">모델 ${result.nodes.length}개 · 대표 문헌 연도순 · 색상은 모델 계열</text>`;
  for(const edge of result.edges){
   const a=positions.get(edge.source),b=positions.get(edge.target);if(!a||!b)continue;
   let path;
   if(Math.abs(a[1]-b[1])<10){const forward=b[0]>a[0],x1=a[0]+(forward?96:-96),x2=b[0]+(forward?-96:96),lift=Math.abs(a[0]-b[0])>230?55:18;path=`M${x1},${a[1]} C${x1},${a[1]-lift} ${x2},${b[1]-lift} ${x2},${b[1]}`;}
   else{const down=b[1]>a[1],y1=a[1]+(down?39:-39),y2=b[1]+(down?-39:39),mid=(y1+y2)/2;path=`M${a[0]},${y1} C${a[0]},${mid} ${b[0]},${mid} ${b[0]},${y2}`;}
   const index=graph.edges.indexOf(edge),key=index<0?graph.edges.findIndex(e=>e.source===edge.source&&e.target===edge.target&&e.relation===edge.relation):index;
   svg+=`<g data-edge="${key}" role="button" tabindex="0" aria-label="관계 ${esc(nodes.get(edge.source)?.label)} → ${esc(nodes.get(edge.target)?.label)} · ${esc(relations[edge.relation])}"><title>${esc(edge.explanation)}</title><path class="edge-hit" d="${path}"/><path class="map-edge ${edge.basis==='pedagogical'||edge.relation==='alternative_to'?'dotted':''} ${[edge.source,edge.target].includes(id)?'active':'faded'}" d="${path}" marker-end="url(#arrow)"/></g>`;
  }
  for(const node of result.nodes){
   const [x,y]=positions.get(node.id),label=Array.from(node.label),short=label.length>21?label.slice(0,20).join('')+'…':node.label,accent=families[node.family]?.[1]??'#6d8a74';
   svg+=`<g class="model-node kind-${node.kind} ${node.id===id?'selected':adjacent.has(node.id)?'related':''}" style="--node-accent:${accent}" data-node="${esc(node.id)}" role="button" tabindex="0" aria-label="선택 ${esc(node.label)}" transform="translate(${x-96},${y-39})"><title>${esc(node.label)}</title><rect width="192" height="78" rx="9"/><rect class="node-stripe" x="0" y="14" width="3" height="49" rx="1"/><text x="15" y="19" class="node-index">${esc(node.kind==='model'?families[node.family]?.[0]??'MODEL':kinds[node.kind])}</text><text x="15" y="44" class="node-label">${esc(short)}</text><text x="15" y="65" class="node-year">${esc(node.reference_year??'ONTOLOGY')}</text></g>`;
  }
  $('#map').innerHTML=svg;if(reset)fit();else applyCamera();
  $('#map-message').textContent=mode==='overview'?`${result.nodes.length}개 모델 · ${result.edges.length}개 모델 간 관계 · 연결 방향은 상세 패널에서 확인`:`${result.nodes.length}개 노드 · ${result.edges.length}개 관계${result.truncated?' · 일부 표시':''}`;
  $('#family-legend').innerHTML=mode==='overview'?Object.entries(families).filter(([key])=>family==='all'||family===key).map(([,value])=>`<span><i style="--node-accent:${value[1]}"></i>${value[0]}</span>`).join(''):'모델 · 개념 · 구성 요소 · 학습 목적 · 과제 · 문헌';
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
 if(node.runtime_id){html+=`<h3>실제 구조 살펴보기</h3><p>${esc(node.runtime_note)}</p><div class="runtime-actions"><button class="primary" id="launch">P2I로 열기 ↗</button><a class="secondary" href="/api/kg/models/${encodeURIComponent(node.id)}/checkpoint">예제 체크포인트 ↓</a></div><p class="muted">편집한 결과는 열린 P2I 세션에서 저장하세요.</p><button class="secondary" id="inspect-model">온톨로지 ↔ 실제 모듈 연결</button><div id="bindings"></div><div id="runtime-status" class="runtime-status" aria-live="polite"></div>`;}
 if(hosted){html=html.replace('<button class="primary" id="launch">P2I로 열기 ↗</button>', '<a class="primary" href="http://127.0.0.1:8010/" target="_blank" rel="noopener noreferrer">로컬 P2I 열기 ↗</a>').replace(/<a class="secondary" href="[^"]+\/checkpoint">예제 체크포인트 ↓<\/a>/,'').replace('편집한 결과는 열린 P2I 세션에서 저장하세요.','실제 실행·편집은 PC에서 p2i-kg serve를 시작한 후 로컬 P2I로 연결하세요.');}
 $('#detail').innerHTML=html;$('#detail').scrollTop=0;
 if(node.runtime_id&&sessions.has(node.id))showSession(node.id,sessions.get(node.id));
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
 const target=event.target.closest('[data-node]');if(target){select(target.dataset.node);return;}
 const edge=event.target.closest('[data-edge]');if(edge){showEdge(Number(edge.dataset.edge));return}
 const panel=event.target.closest('[data-panel]');if(panel){$('#'+panel.dataset.panel).showModal();return}
 if(event.target.closest('[data-close]')){event.target.closest('dialog').close();return}
 const pair=event.target.closest('[data-compare-pair]');if(pair){const [left,right]=pair.dataset.comparePair.split(',');$('#compare-left').value=left;$('#compare-right').value=right;await compare();$('#compare-panel').showModal();return}
 if(event.target.id==='expand-connections'){mode='neighborhood';renderMap();return}
 if(event.target.id==='compare-selected'){$('#compare-left').value=selected;await compare();$('#compare-panel').showModal();return}
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
document.addEventListener('keydown',event=>{const node=event.target.closest('[data-node]'),edge=event.target.closest('[data-edge]');if((node||edge)&&(event.key==='Enter'||event.key===' ')){event.preventDefault();if(node)select(node.dataset.node);else showEdge(Number(edge.dataset.edge))}});
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
 $('#stats').textContent=`${models.length}개 모델 · ${graph.nodes.length}개 노드 · ${graph.edges.length}개 관계`;

 $('#revision').textContent=`CATALOG ${graph.revision.slice(0,7)}`;
 for(const selector of ['#compare-left','#compare-right','#model-select'])$(selector).innerHTML=models.map(node=>`<option value="${esc(node.id)}">${esc(node.label)}</option>`).join('');
 $('#compare-left').value='model:bert';$('#compare-right').value='model:gpt2';
 if(!nodes.has(selected))selected=models[0]?.id;$('#model-select').value=selected;
 if(hosted){$('.extend-card').hidden=true;$('.snapshot-note').textContent='현재 카탈로그의 스냅샷입니다. 실제 실행·편집은 로컬 P2I에서 연결합니다.'}renderLibrary();select(selected,{closePanels:false});await compare();
}
try{await loadGraph()}catch(error){$('#global-error').textContent=error.message}
