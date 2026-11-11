/** Deterministic layouts; no simulation, network dependency, or node-count cutoff.
 * Mind map: Reingold–Tilford style contour packing, then balanced left/right trees.
 * Reference: https://reingold.co/graph-drawing.shtml
 * The hierarchy is a navigation scaffold, never a new ontology assertion.
 */
export const NODE_WIDTH = 192, NODE_HEIGHT = 68;
const COLUMN = 286, ROW = 94, PAD = 130;
const order = (a,b) => (a.reference_year??0)-(b.reference_year??0)||a.id.localeCompare(b.id);

// Pack child contours at each depth, centering each parent over its children.
function tidy(root) {
  const children = root.children.map(tidy);
  let lower = [], upper = [];
  for (const child of children) {
    let shift = 0;
    for (let d=0;d<child.upper.length;d++) if (lower[d]!==undefined)
      shift = Math.max(shift,lower[d]+ROW-child.upper[d]);
    child.shift = shift;
    child.upper.forEach((y,d)=>{
      upper[d] = Math.min(upper[d]??Infinity,y+shift);
      lower[d] = Math.max(lower[d]??-Infinity,child.lower[d]+shift);
    });
  }
  const center = children.length ? (children[0].shift+children.at(-1).shift)/2 : 0;
  const items = [{node:root.node,depth:0,y:0}], branches = [];
  for (const child of children) {
    const shift = child.shift-center;
    items.push(...child.items.map(item=>({...item,depth:item.depth+1,y:item.y+shift})));
    branches.push({source:root.node.id,target:child.items[0].node.id},...child.branches);
  }
  return {items,branches,upper:[0,...upper.map(y=>y-center)],lower:[0,...lower.map(y=>y-center)]};
}

function bounds(items,branches,edges,guides=[]) {
  const minX=Math.min(0,...items.map(n=>n.x-NODE_WIDTH/2))-PAD;
  const minY=Math.min(0,...items.map(n=>n.y-NODE_HEIGHT/2))-PAD;
  const maxX=Math.max(0,...items.map(n=>n.x+NODE_WIDTH/2))+PAD;
  const maxY=Math.max(0,...items.map(n=>n.y+NODE_HEIGHT/2))+PAD;
  return {items:items.map(n=>({...n,x:n.x-minX,y:n.y-minY})),branches,edges,
    guides:guides.map(g=>({...g,x:g.x-minX,y:g.y-minY})),width:maxX-minX,height:maxY-minY};
}

export function mindmap(graph,family='all') {
  const models=graph.nodes.filter(n=>n.kind==='model'&&(family==='all'||n.family===family)).sort(order);
  const known=new Map(models.map(n=>[n.id,n]));
  const categories=[...new Set(models.map(n=>n.family??'other'))];
  const trees=categories.map(key=>{
    const members=models.filter(n=>(n.family??'other')===key), ids=new Set(members.map(n=>n.id));
    const parent=new Map();
    for (const node of members) {
      // Prefer the closest older documented parent in the same family.
      const candidates=graph.edges.filter(e=>e.source===node.id&&e.relation==='extends'&&ids.has(e.target)&&order(known.get(e.target),node)<0)
        .sort((a,b)=>(a.basis==='pedagogical')-(b.basis==='pedagogical')||order(known.get(b.target),known.get(a.target)));
      if(candidates.length)parent.set(node.id,candidates[0].target);
    }
    const build=node=>({node,children:members.filter(n=>parent.get(n.id)===node.id).map(build)});
    const root={node:{id:'family:'+key,kind:'family',family:key,label:key,count:members.length},children:members.filter(n=>!parent.has(n.id)).map(build)};
    const tree=tidy(root), ys=tree.items.map(n=>n.y);
    return {key,root,tree,height:Math.max(...ys)-Math.min(...ys)+ROW};
  }).sort((a,b)=>b.height-a.height||a.key.localeCompare(b.key));
  const sides=[[],[]], weights=[0,0];
  for(const tree of trees){const side=weights[0]<=weights[1]?0:1;sides[side].push(tree.root);weights[side]+=tree.height+38;}
  const root={id:'atlas:root',kind:'root',label:'Neural architectures',count:models.length};
  const items=[{key:root.id,node:root,x:0,y:0}],branches=[];
  sides.forEach((children,index)=>{
    const side=index===0?-1:1, tree=tidy({node:root,children});
    const ys=tree.items.slice(1).map(item=>item.y),center=ys.length?(Math.min(...ys)+Math.max(...ys))/2:0;
    items.push(...tree.items.slice(1).map(item=>({key:item.node.id,node:item.node,x:side*item.depth*COLUMN,y:item.y-center,side})));
    branches.push(...tree.branches.map(edge=>({...edge,side})));
  });
  return {...bounds(items,branches,[]),modelCount:models.length};
}

export function neighborhood(graph,id,hops=1,relation='') {
  const allowed=relation?new Set(relation.split(',')):null;
  const known=new Map(graph.nodes.map(n=>[n.id,n]));
  const edges=graph.edges.filter(e=>known.has(e.source)&&known.has(e.target)&&(!allowed||allowed.has(e.relation)));
  const distance=new Map([[id,0]]);let frontier=[id];
  for(let hop=0;hop<hops;hop++){
    const active=new Set(frontier), next=[];
    for(const edge of edges)if(active.has(edge.source)||active.has(edge.target)){
      for(const key of [edge.source,edge.target])if(!distance.has(key)){distance.set(key,hop+1);next.push(key);}
    }
    frontier=next;
  }
  return {nodes:graph.nodes.filter(n=>distance.has(n.id)),edges:edges.filter(e=>distance.has(e.source)&&distance.has(e.target)),distance,hops};
}

export function connectionLayout(result,id) {
  const byId=new Map(result.nodes.map(n=>[n.id,n])), copies=new Map([[id,[0]]]);
  const incoming=new Set(result.edges.filter(e=>e.target===id&&e.source!==id).map(e=>e.source));
  const outgoing=new Set(result.edges.filter(e=>e.source===id&&e.target!==id).map(e=>e.target));
  for(const node of result.nodes){
    if(node.id===id)continue;
    const sides=[];if(incoming.has(node.id))sides.push(-1);if(outgoing.has(node.id))sides.push(1);
    if(sides.length)copies.set(node.id,sides);
  }
  // Every mixed-direction second-hop neighbor remains visible, with its own depth.
  for(let depth=2;depth<=result.hops;depth++)for(const node of result.nodes){
    if(result.distance.get(node.id)!==depth)continue;
    const votes=[];
    for(const e of result.edges){const other=e.source===node.id?e.target:e.target===node.id?e.source:null;
      if(other&&result.distance.get(other)===depth-1)votes.push(...(copies.get(other)??[]));}
    copies.set(node.id,[(votes.reduce((a,b)=>a+b,0)<0?-1:1)*depth]);
  }
  const layers=new Map(), placements=new Map();
  for(const [nodeId,levels] of copies)for(const level of levels){
    const item={key:nodeId+'@'+level,node:byId.get(nodeId),level,x:-level*350,y:0,duplicate:levels.length>1};
    if(!layers.has(level))layers.set(level,[]);layers.get(level).push(item);placements.set(item.key,item);
  }
  for(const [level,items] of layers){
    items.sort((a,b)=>a.node.kind.localeCompare(b.node.kind)||a.node.label.localeCompare(b.node.label));
    items.forEach((item,i)=>item.y=(i-(items.length-1)/2)*ROW);
  }
  // Barycentric sweeps reduce crossings between the outer columns.
  for(let pass=0;pass<4;pass++)for(const [level,items] of [...layers].sort((a,b)=>Math.abs(a[0])-Math.abs(b[0]))){
    if(Math.abs(level)<2)continue;
    const score=item=>{
      const linked=result.edges.flatMap(e=>e.source===item.node.id?[e.target]:e.target===item.node.id?[e.source]:[]);
      const ys=linked.flatMap(key=>(copies.get(key)??[]).filter(l=>Math.sign(l)===Math.sign(level)&&Math.abs(l)<Math.abs(level)).map(l=>placements.get(key+'@'+l).y));
      return ys.length?ys.reduce((a,b)=>a+b,0)/ys.length:item.y;
    };
    const scores=new Map(items.map(n=>[n.key,score(n)]));items.sort((a,b)=>scores.get(a.key)-scores.get(b.key)||a.node.id.localeCompare(b.node.id));
    items.forEach((item,i)=>item.y=(i-(items.length-1)/2)*ROW);
  }
  const edges=result.edges.map(edge=>{
    const a=copies.get(edge.source),b=copies.get(edge.target);
    let source=a[0],target=b[0];
    if(edge.source===id){source=0;target=b.includes(1)?1:b[0];}
    else if(edge.target===id){target=0;source=a.includes(-1)?-1:a[0];}
    else{
      const pairs=a.flatMap(x=>b.map(y=>[x,y])).sort((p,q)=>(Math.sign(p[0])!==Math.sign(p[1]))-(Math.sign(q[0])!==Math.sign(q[1]))||Math.abs(p[0]-p[1])-Math.abs(q[0]-q[1]));
      [source,target]=pairs[0];
    }
    return {edge,source:edge.source+'@'+source,target:edge.target+'@'+target};
  });
  // Display dependency flow: referenced foundations left, models that use this item right.
  // Keep source/target identities unchanged so relation inspection remains accurate.
  const guides=[{x:-350,y:-Math.max(1,layers.get(1)?.length??0)*ROW/2-15,label:'기반 · 구성',side:'incoming'},
    {x:0,y:-86,label:'선택한 항목',side:'center'},
    {x:350,y:-Math.max(1,layers.get(-1)?.length??0)*ROW/2-15,label:'활용 · 확장',side:'outgoing'}];
  for(const [level,items]of layers)if(Math.abs(level)>1)guides.push({x:-level*350,y:-items.length*ROW/2-15,label:Math.abs(level)+'단계 연결',side:level>0?'incoming':'outgoing'});
  return {...bounds([...placements.values()],[],edges,guides),uniqueCount:byId.size,incomingCount:outgoing.size,outgoingCount:incoming.size};
}

export function edgePath(a,b,offset=0) {
  if(a.key===b.key){const x=a.x+NODE_WIDTH/2,y=a.y;return `M${x},${y-13} C${x+66},${y-90} ${x+66},${y+90} ${x},${y+13}`;}
  if(Math.abs(a.x-b.x)<1){const x=a.x+NODE_WIDTH/2+12+offset,y1=a.y,y2=b.y;return `M${a.x+NODE_WIDTH/2},${y1} C${x+55},${y1} ${x+55},${y2} ${b.x+NODE_WIDTH/2},${y2}`;}
  const direction=b.x>a.x?1:-1,x1=a.x+direction*NODE_WIDTH/2,x2=b.x-direction*NODE_WIDTH/2;
  const bend=Math.max(45,Math.abs(x2-x1)*.5);
  return `M${x1},${a.y+offset} C${x1+direction*bend},${a.y+offset} ${x2-direction*bend},${b.y+offset} ${x2},${b.y+offset}`;
}
