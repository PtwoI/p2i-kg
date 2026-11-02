import './study.js';
import {read,hosted} from './catalog-client.js';
const $=selector=>document.querySelector(selector);
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let graph,nodes,selected='model:transformer',zoom=1,searchTimer;
const sessions=new Map();
let mapRequest=0;
const relations={extends:'확장',alternative_to:'대안',documented_in:'대표 문헌',uses:'사용 개념',has_component:'구성 요소',subconcept_of:'상위 개념',trained_with:'학습 목적',applied_to:'적용 과제'};
const kinds={model:'모델',concept:'개념',paper:'논문',component:'구성 요소',objective:'학습 목적',task:'과제'};
const bases={documented:'문헌 관계',structural:'구조적 관계',pedagogical:'교육적 연결'};
function select(id){if(!nodes.has(id))return;selected=id;if(nodes.get(id).kind==='model'&&[...$('#model-select').options].some(option=>option.value===id))$('#model-select').value=id;renderMap();renderDetail();}
window.addEventListener('atlas:select',event=>{select(event.detail.id);if(event.detail.compare){$('#compare-left').value=event.detail.compare[0];$('#compare-right').value=event.detail.compare[1];compare();}});
function chip(id){const node=nodes.get(id);return `<button class="chip" data-node="${esc(id)}">${esc(node?.label??id)}</button>`}
async function renderMap(){
 const request=++mapRequest,id=selected;
 try{
  const hops=$('#hops').value,relation=$('#relation').value;
  const result=await read(`/api/kg/neighborhood/${encodeURIComponent(id)}?hops=${hops}&limit=40${relation?'&relations='+relation:''}`);
  if(request!==mapRequest)return;
  const neighbors=result.nodes.filter(node=>node.id!==id).sort((a,b)=>a.kind.localeCompare(b.kind)||a.label.localeCompare(b.label));
  const rows=Math.max(3,Math.ceil(neighbors.length/3)),height=rows*110+70;
  const positions=new Map([[id,[130,height/2]]]);
  neighbors.forEach((node,index)=>positions.set(node.id,[420+Math.floor(index/rows)*320,65+(index%rows)*110]));
  let svg='<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#809d8d"/></marker></defs>';
  const shownEdges=hops==='1'?result.edges.filter(edge=>edge.source===id||edge.target===id):result.edges;
  for(const edge of shownEdges){
   const a=positions.get(edge.source),b=positions.get(edge.target);if(!a||!b)continue;
   const forward=b[0]>=a[0],x1=a[0]+(forward?94:-94),x2=b[0]+(forward?-94:94),mid=(x1+x2)/2;
   const path=`M${x1},${a[1]} C${mid},${a[1]} ${mid},${b[1]} ${x2},${b[1]}`;
   svg+=`<path class="map-edge ${edge.basis==='pedagogical'?'dotted':''} ${[edge.source,edge.target].includes(id)?'active':''}" d="${path}" marker-end="url(#arrow)"><title>${esc(nodes.get(edge.source)?.label)} → ${esc(nodes.get(edge.target)?.label)} · ${esc(relations[edge.relation])}</title></path>`;
  }
  for(const node of [nodes.get(id),...neighbors]){
   const [x,y]=positions.get(node.id),label=Array.from(node.label);const short=label.length>18?label.slice(0,17).join('')+'…':node.label;
   svg+=`<g class="model-node kind-${node.kind} ${node.id===id?'selected':''}" data-node="${esc(node.id)}" role="button" tabindex="0" aria-label="선택 ${esc(node.label)}" transform="translate(${x-94},${y-40})"><title>${esc(node.label)}</title><rect width="188" height="80" rx="9"/><text x="14" y="18" class="node-index">${esc(kinds[node.kind])}</text><text x="14" y="44" class="node-label">${esc(short)}</text><text x="14" y="65" class="node-year">${esc(node.reference_year??node.family??'ONTOLOGY')}</text></g>`;
  }
  $('#map').setAttribute('viewBox',`0 0 1200 ${height}`);$('#map').innerHTML=svg;$('#map').style.width=`${zoom*100}%`;$('#zoom-reset').textContent=`${Math.round(zoom*100)}%`;
  $('#map-message').textContent=`${result.nodes.length}개 노드 · ${shownEdges.length}개 관계${result.truncated?' · 일부 표시 (검색으로 범위를 좁히세요)':''}`;
 }catch(error){if(request===mapRequest)$('#global-error').textContent=error.message}
}
function renderDetail(){
 const node=nodes.get(selected);if(!node)return;
 const outgoing=graph.edges.filter(edge=>edge.source===node.id),incoming=graph.edges.filter(edge=>edge.target===node.id);
 const papers=node.kind==='paper'?[node]:outgoing.filter(edge=>edge.relation==='documented_in').map(edge=>nodes.get(edge.target));
 const concepts=outgoing.filter(edge=>edge.relation==='uses');
 const connections=[...outgoing,...incoming].filter(edge=>['extends','alternative_to'].includes(edge.relation));
 let html=`<span class="detail-type">${esc(node.kind)} / ${esc(node.family??'KNOWLEDGE')}</span><h2>${esc(node.label)}</h2>${node.reference_year?`<div class="detail-year">${node.reference_year}</div>`:''}<p>${esc(node.summary)}</p>`;
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
 $('#detail').innerHTML=html;
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
 const target=event.target.closest('[data-node]');if(target){select(target.dataset.node);return;}
 if(event.target.id==='launch')launch();
 if(event.target.id==='find-models'){
  const id=selected;try{const result=await read(`/api/kg/query?concept=${encodeURIComponent(id)}`);if(selected===id)$('#reasoned-models').innerHTML=result.matches.map(match=>`<div>${chip(match.node.id)} <details><summary>${match.proof.length}단계 연결 보기</summary>${match.proof_edges.map(edge=>`<p>${esc(nodes.get(edge.source)?.label)} → ${esc(relations[edge.relation])} → ${esc(nodes.get(edge.target)?.label)}</p><div class="chips">${edge.evidence.map(chip).join('')}</div>`).join('')}</details></div>`).join('')||'<p>연결된 모델이 없습니다.</p>'}catch(error){$('#global-error').textContent=error.message}
 }
 if(event.target.id==='inspect-model'){
  const id=selected;event.target.disabled=true;$('#bindings').textContent='실제 모듈을 추적하고 있습니다…';
  try{const result=await read(`/api/kg/inspect/${encodeURIComponent(id)}`);if(selected===id)$('#bindings').innerHTML=`<p>${result.modules}개 모듈 · ${result.operations}개 관측 연산</p>`+result.bindings.map(binding=>`<div class="binding">${chip(binding.ontology_id)}<code>${esc(binding.module_path)}</code><small>${esc(binding.module_type)} · ${binding.parameter_count.toLocaleString()} parameters${binding.editable?' · 편집 가능':''}</small></div>`).join('')}catch(error){if(selected===id)$('#bindings').textContent=error.message}
  finally{if(selected===id&&$('#inspect-model'))$('#inspect-model').disabled=false}
 }
 if(event.target.id==='close-session'){
  const id=selected;try{await read(`/api/kg/runtime/${encodeURIComponent(id)}`,{method:'DELETE'});sessions.delete(id);renderDetail()}catch(error){$('#runtime-status').textContent=error.message}
 }
});
document.addEventListener('keydown',event=>{const node=event.target.closest('[data-node]');if(node&&(event.key==='Enter'||event.key===' ')){event.preventDefault();select(node.dataset.node)}});
$('#family').addEventListener('change',()=>{
 const family=$('#family').value;$('#model-select').innerHTML=graph.nodes.filter(n=>n.kind==='model'&&(family==='all'||n.family===family)).map(n=>`<option value="${esc(n.id)}">${esc(n.label)}</option>`).join('');
 if($('#model-select').value)select($('#model-select').value);
});
$('#model-select').addEventListener('change',()=>select($('#model-select').value));
$('#hops').addEventListener('change',renderMap);$('#relation').addEventListener('change',renderMap);
function renderLibrary(){const kind=$('#library-kind').value;$('#concepts').innerHTML=graph.nodes.filter(node=>node.kind===kind).map(node=>`<button class="concept-tile" data-node="${esc(node.id)}"><strong>${esc(node.label)}</strong><p>${esc(node.summary)}</p></button>`).join('')}
$('#library-kind').addEventListener('change',renderLibrary);
$('#patch-file').addEventListener('change',async event=>{
 const file=event.target.files?.[0];if(!file)return;
 try{const patch=JSON.parse(await file.text());const result=await read('/api/kg/extend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_revision:graph.revision,patch})});$('#patch-status').textContent=`추가 완료: ${result.added_nodes}개 노드 · ${result.added_edges}개 관계`;await loadGraph()}
 catch(error){$('#patch-status').textContent=error.message}finally{event.target.value=''}
});
$('#zoom-in').addEventListener('click',()=>{zoom=Math.min(2,zoom+.2);renderMap()});
$('#zoom-out').addEventListener('click',()=>{zoom=Math.max(.8,zoom-.2);renderMap()});
$('#zoom-reset').addEventListener('click',()=>{zoom=1;renderMap()});
$('#compare').addEventListener('click',compare);
$('#search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(async()=>{
 const q=$('#search').value;if(!q.trim()){$('#search-results').innerHTML='';return;}
 try{const result=await read(`/api/kg/search?q=${encodeURIComponent(q)}`);if($('#search').value!==q)return;$('#search-results').innerHTML=result.nodes.length?result.nodes.map(node=>`<button data-node="${esc(node.id)}">${esc(node.label)} <small>· ${esc(node.kind)}</small></button>`).join(''):'<span class="muted">검색 결과가 없습니다.</span>'}catch(error){$('#global-error').textContent=error.message}
},160)});
async function loadGraph(){
 graph=await read('/api/kg');nodes=new Map(graph.nodes.map(node=>[node.id,node]));window.dispatchEvent(new CustomEvent('atlas:catalog',{detail:graph}));
 const models=graph.nodes.filter(node=>node.kind==='model');
 $('#stats').textContent=`${models.length}개 모델 · ${graph.nodes.length}개 노드 · ${graph.edges.length}개 관계`;
 $('#catalog-range').textContent=`01 — ${String(models.length).padStart(2,'0')}`;
 $('#revision').textContent=`CATALOG ${graph.revision.slice(0,7)}`;
 for(const selector of ['#compare-left','#compare-right','#model-select'])$(selector).innerHTML=models.map(node=>`<option value="${esc(node.id)}">${esc(node.label)}</option>`).join('');
 $('#compare-left').value='model:bert';$('#compare-right').value='model:gpt2';
 if(!nodes.has(selected))selected=models[0]?.id;$('#model-select').value=selected;
 renderLibrary();select(selected);await compare();
}
try{await loadGraph()}catch(error){$('#global-error').textContent=error.message}
