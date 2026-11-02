/** Small transparent calculations for the learning playgrounds. */
export function attention(queries,keys,values,{causal=false,temperature=1}={}){
 const scale=Math.sqrt(keys[0].length)*temperature;
 const weights=queries.map((query,i)=>{
  const scores=keys.map((key,j)=>causal&&j>i?-Infinity:query.reduce((sum,x,k)=>sum+x*key[k],0)/scale);
  const maximum=Math.max(...scores),exp=scores.map(x=>Number.isFinite(x)?Math.exp(x-maximum):0),total=exp.reduce((a,b)=>a+b,0);
  return exp.map(x=>x/total);
 });
 return {weights,output:weights.map(row=>values[0].map((_,k)=>row.reduce((sum,w,j)=>sum+w*values[j][k],0)))};
}
export function convolve(input,kernel,stride=1){
 const size=Math.floor((input.length-kernel.length)/stride)+1;
 return Array.from({length:size},(_,y)=>Array.from({length:size},(_,x)=>kernel.reduce((sum,row,ky)=>sum+row.reduce((s,w,kx)=>s+w*input[y*stride+ky][x*stride+kx],0),0)));
}
export function recurrence(input,wx=.8,wh=.6){
 const states=[0];for(const x of input)states.push(Math.tanh(wx*x+wh*states.at(-1)));return states;
}
