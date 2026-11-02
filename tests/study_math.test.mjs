import {test} from 'node:test';
import assert from 'node:assert/strict';
import {attention,convolve,recurrence} from '../src/p2i_kg/static/study-math.js';
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`);
test('attention: stable softmax, masked future, normalized output',()=>{
 const q=[[1,0],[0,1]],k=[[1,0],[0,1]],v=[[2,0],[0,4]];
 const plain=attention(q,k,v),expected=Math.exp(1/Math.sqrt(2))/(Math.exp(1/Math.sqrt(2))+1);
 close(plain.weights[0][0],expected);close(plain.output[0][0],2*expected);
 const masked=attention(q,k,v,{causal:true});assert.deepEqual(masked.weights[0],[1,0]);assert.deepEqual(masked.output[0],[2,0]);
 for(const row of masked.weights)close(row.reduce((a,b)=>a+b,0),1);
 const hot=attention(q,k,v,{temperature:2});assert.ok(hot.weights[0][0]<plain.weights[0][0]);
 const huge=attention([[10000]],[[10000]],[[3]]);assert.deepEqual(huge.output,[[3]]);
});
test('cross-correlation: orientation and stride are visible in output',()=>{
 const input=[[1,2,3],[4,5,6],[7,8,9]],kernel=[[1,0],[0,-1]];
 assert.deepEqual(convolve(input,kernel),[[-4,-4],[-4,-4]]);
 assert.deepEqual(convolve(input,kernel,2),[[-4]]);
 assert.deepEqual(convolve(input,[[0,1],[0,0]]),[[2,3],[5,6]]);
});
test('recurrence: previous state is reused, zero recurrence forgets it',()=>{
 const states=recurrence([1,0,-1],.8,.6);close(states[1],Math.tanh(.8));close(states[2],Math.tanh(.6*states[1]));
 close(states[3],Math.tanh(-.8+.6*states[2]));close(recurrence([1,0],.8,0)[2],0);
});
