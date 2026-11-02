const $=selector=>document.querySelector(selector);
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let graph,nodes,selected='model:transformer',zoom=1,searchTimer;
const sessions=new Map();
const positions={perceptron:[120,170],mlp:[380,170],cnn:[660,65],resnet:[960,65],rnn:[660,365],lstm:[960,275],gru:[960,445],transformer:[1240,170]};
const relations={extends:'확장',alternative_to:'대안',documented_in:'대표 문헌',uses:'사용 개념'};
const bases={documented:'문헌 관계',structural:'구조적 관계',pedagogical:'교육적 연결'};
async function read(url,options){const response=await fetch(url,options);if(!response.ok){let body;try{body=await response.json()}catch{body={detail:`요청 실패 (${response.status})`}}throw Error(body.detail??'요청을 완료하지 못했습니다.')}return response.json()}
function select(id){if(!nodes.has(id))return;selected=id;renderMap();renderDetail();}
function chip(id){const node=nodes.get(id);return `<button class="chip" data-node="${esc(id)}">${esc(node?.label??id)}</button>`}
function renderMap(){
 const family=$('#family').value;const focus=nodes.get(selected);
 const related=focus?.kind==='concept'?new Set(graph.edges.filter(edge=>edge.target===selected&&edge.relation==='uses').map(edge=>edge.source)):null;
 let svg='<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#809d8d"/></marker></defs>';
 for(const edge of graph.edges.filter(edge=>['extends','alternative_to'].includes(edge.relation))){
  const a=positions[edge.source.split(':')[1]],b=positions[edge.target.split(':')[1]];if(!a||!b)continue;
  const x1=a[0]-94,y1=a[1],x2=b[0]+94,y2=b[1],mid=(x1+x2)/2;
  const path=edge.relation==='alternative_to'?`M${a[0]},${a[1]+42} V505 Q${a[0]},525 ${a[0]-20},525 H${b[0]+20} Q${b[0]},525 ${b[0]},505 V${b[1]+42}`:`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
  svg+=`<path class="map-edge ${edge.basis==='pedagogical'||edge.relation==='alternative_to'?'dotted':''} ${[edge.source,edge.target].includes(selected)?'active':''}" d="${path}" marker-end="url(#arrow)"><title>${esc(nodes.get(edge.source).label)} → ${esc(nodes.get(edge.target).label)} · ${esc(bases[edge.basis])}</title></path>`;
 }
 graph.nodes.filter(node=>node.kind==='model').forEach((node,index)=>{
  const [x,y]=positions[node.id.split(':')[1]]??[120+index*220,170];
  const dimmed=(family!=='all'&&node.family!==family)||(related&&!related.has(node.id));
  svg+=`<g class="model-node ${node.id===selected?'selected':''} ${dimmed?'dimmed':''}" data-node="${esc(node.id)}" role="button" tabindex="0" aria-label="선택 ${esc(node.label)}" transform="translate(${x-94},${y-42})"><rect width="188" height="84" rx="9"/><text x="16" y="19" class="node-index">${String(index+1).padStart(2,'0')} · ${esc(node.family.toUpperCase())}</text><text x="16" y="46" class="node-label">${esc(node.label.split(' · ')[0])}</text><text x="16" y="67" class="node-year">${node.reference_year} · 대표 문헌</text></g>`;
 });
 $('#map').innerHTML=svg;$('#map').style.width=`${zoom*100}%`;$('#zoom-reset').textContent=`${Math.round(zoom*100)}%`;
}
function renderDetail(){
 const node=nodes.get(selected);if(!node)return;
 const outgoing=graph.edges.filter(edge=>edge.source===node.id),incoming=graph.edges.filter(edge=>edge.target===node.id);
 const papers=node.kind==='paper'?[node]:outgoing.filter(edge=>edge.relation==='documented_in').map(edge=>nodes.get(edge.target));
 const concepts=outgoing.filter(edge=>edge.relation==='uses');
 const connections=[...outgoing,...incoming].filter(edge=>['extends','alternative_to'].includes(edge.relation));
 let html=`<span class="detail-type">${esc(node.kind)} / ${esc(node.family??'KNOWLEDGE')}</span><h2>${esc(node.label)}</h2>${node.reference_year?`<div class="detail-year">${node.reference_year}</div>`:''}<p>${esc(node.summary)}</p>`;
 if(concepts.length)html+=`<h3>구성 개념</h3><div class="chips">${concepts.map(edge=>chip(edge.target)).join('')}</div>`;
 if(node.kind==='concept')html+=`<h3>이 개념을 사용하는 모델</h3><div class="chips">${incoming.filter(edge=>edge.relation==='uses').map(edge=>chip(edge.source)).join('')}</div>`;
 if(connections.length)html+=`<h3>다른 모델과의 관계</h3>${connections.map(edge=>`<div class="relation-item"><button data-node="${esc(edge.source===node.id?edge.target:edge.source)}">${esc(nodes.get(edge.source).label)} → ${esc(nodes.get(edge.target).label)} · ${esc(relations[edge.relation])}</button><small>${esc(bases[edge.basis])} · ${esc(edge.explanation)}</small>${edge.evidence.map(id=>`<button data-node="${esc(id)}">근거 문헌 ↗</button>`).join('')}</div>`).join('')}`;
 if(papers.length)html+=`<h3>대표 문헌</h3>${papers.map(paper=>`<a class="reference" href="${esc(paper.url)}" target="_blank" rel="noopener noreferrer">${esc(paper.label)} ↗<small>${esc(paper.authors.join(', '))} · ${paper.reference_year}</small></a>`).join('')}`;
 if(node.runtime_id){html+=`<h3>실제 구조 살펴보기</h3><p>${esc(node.runtime_note)}</p><div class="runtime-actions"><button class="primary" id="launch">P2I로 열기 ↗</button><a class="secondary" href="/api/kg/models/${encodeURIComponent(node.id)}/checkpoint">예제 체크포인트 ↓</a></div><p class="muted">편집한 결과는 열린 P2I 세션에서 저장하세요.</p><div id="runtime-status" class="runtime-status" aria-live="polite"></div>`;}
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
  $('#comparison').innerHTML=`<div class="compare-columns">${[['공통 개념',result.common],[result.left.label+'에만 연결',result.left_only],[result.right.label+'에만 연결',result.right_only]].map(([title,items])=>`<div><h3>${esc(title)}</h3><div class="chips">${items.length?items.map(node=>chip(node.id)).join(''):'<span class="muted">현재 그래프에 연결된 개념이 없습니다.</span>'}</div></div>`).join('')}</div><p class="compare-note">${esc(result.note)}</p>`;
 }catch(error){$('#global-error').textContent=error.message}
}
document.addEventListener('click',async event=>{
 const target=event.target.closest('[data-node]');if(target){select(target.dataset.node);return;}
 if(event.target.id==='launch')launch();
 if(event.target.id==='close-session'){
  const id=selected;try{await read(`/api/kg/runtime/${encodeURIComponent(id)}`,{method:'DELETE'});sessions.delete(id);renderDetail()}catch(error){$('#runtime-status').textContent=error.message}
 }
});
document.addEventListener('keydown',event=>{const node=event.target.closest('[data-node]');if(node&&(event.key==='Enter'||event.key===' ')){event.preventDefault();select(node.dataset.node)}});
$('#family').addEventListener('change',renderMap);
$('#zoom-in').addEventListener('click',()=>{zoom=Math.min(2,zoom+.2);renderMap()});
$('#zoom-out').addEventListener('click',()=>{zoom=Math.max(.8,zoom-.2);renderMap()});
$('#zoom-reset').addEventListener('click',()=>{zoom=1;renderMap()});
$('#compare').addEventListener('click',compare);
$('#search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(async()=>{
 const q=$('#search').value;if(!q.trim()){$('#search-results').innerHTML='';return;}
 try{const result=await read(`/api/kg/search?q=${encodeURIComponent(q)}`);if($('#search').value!==q)return;$('#search-results').innerHTML=result.nodes.length?result.nodes.map(node=>`<button data-node="${esc(node.id)}">${esc(node.label)} <small>· ${esc(node.kind)}</small></button>`).join(''):'<span class="muted">검색 결과가 없습니다.</span>'}catch(error){$('#global-error').textContent=error.message}
},160)});
try{
 graph=await read('/api/kg');nodes=new Map(graph.nodes.map(node=>[node.id,node]));
 const models=graph.nodes.filter(node=>node.kind==='model');
 $('#stats').textContent=`${models.length}개 모델 · ${graph.nodes.filter(node=>node.kind==='concept').length}개 개념 · ${graph.nodes.filter(node=>node.kind==='paper').length}개 문헌`;
 $('#revision').textContent=`CATALOG ${graph.revision.slice(0,7)}`;
 for(const selector of ['#compare-left','#compare-right'])$(selector).innerHTML=models.map(node=>`<option value="${esc(node.id)}">${esc(node.label)}</option>`).join('');
 $('#compare-left').value='model:rnn';$('#compare-right').value='model:transformer';
 $('#concepts').innerHTML=graph.nodes.filter(node=>node.kind==='concept').map(node=>`<button class="concept-tile" data-node="${esc(node.id)}"><strong>${esc(node.label)}</strong><p>${esc(node.summary)}</p></button>`).join('');
 select(selected);await compare();
}catch(error){$('#global-error').textContent=error.message}
