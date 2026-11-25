// Core: features, CTC decode, word alignment. Works in browser and node.
export const TOKENS = ['<blk>','ء','ا','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي',' '];
const SHIFT=160, LEN=400, FFT=512, NB=80, SR=16000;
const mel=f=>1127*Math.log(1+f/700);
const banks=(()=>{const bins=FFT/2,width=SR/FFT,ml=mel(20),mh=mel(SR/2-400),d=(mh-ml)/(NB+1);const B=[];
 for(let b=0;b<NB;b++){const l=ml+b*d,c=ml+(b+1)*d,r=ml+(b+2)*d,row=new Float64Array(bins);
  for(let i=0;i<bins;i++){const m=mel(width*i);if(m>l&&m<r)row[i]=m<=c?(m-l)/(c-l):(r-m)/(r-c);}B.push(row);}return B;})();
const win=(()=>{const w=new Float64Array(LEN);for(let i=0;i<LEN;i++)w[i]=Math.pow(0.5-0.5*Math.cos(2*Math.PI*i/(LEN-1)),0.85);return w;})();
const cosT=new Float64Array(FFT/2+1),sinT=new Float64Array(FFT/2+1);
// naive real FFT via radix-2 complex
function fft(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
 for(let len=2;len<=n;len<<=1){const ang=-2*Math.PI/len,wr=Math.cos(ang),wi=Math.sin(ang);for(let i=0;i<n;i+=len){let cr=1,ci=0;for(let j=0;j<len/2;j++){const a=i+j,b=i+j+len/2;const tr=re[b]*cr-im[b]*ci,ti=re[b]*ci+im[b]*cr;re[b]=re[a]-tr;im[b]=im[a]-ti;re[a]+=tr;im[a]+=ti;const nr=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nr;}}}}
export function fbank(w){const n=w.length,T=Math.floor((n+SHIFT/2)/SHIFT),out=new Float32Array(T*NB),eps=1.1920929e-7;
 const fr=new Float64Array(LEN),re=new Float64Array(FFT),im=new Float64Array(FFT),pw=new Float64Array(FFT/2);
 for(let t=0;t<T;t++){const s=t*SHIFT+SHIFT/2-LEN/2;let mean=0;
  for(let i=0;i<LEN;i++){let k=s+i;if(k<0)k=-k-1;if(k>=n)k=2*n-1-k;k=Math.min(Math.max(k,0),n-1);fr[i]=w[k];mean+=fr[i];}
  mean/=LEN;for(let i=0;i<LEN;i++)fr[i]-=mean;
  re.fill(0);im.fill(0);re[0]=fr[0]*(1-0.97)*win[0];for(let i=1;i<LEN;i++)re[i]=(fr[i]-0.97*fr[i-1])*win[i];
  fft(re,im);for(let i=0;i<FFT/2;i++)pw[i]=re[i]*re[i]+im[i]*im[i];
  for(let b=0;b<NB;b++){let a=0;const row=banks[b];for(let i=0;i<FFT/2;i++)a+=pw[i]*row[i];out[t*NB+b]=Math.log(Math.max(a,eps));}}
 return {data:out,T};}
// run model: ort = onnxruntime module, sess = InferenceSession. returns per-frame log-probs [T,31]
export async function logprobs(ort,sess,wave){const {data,T}=fbank(wave);
 const o=await sess.run({x:new ort.Tensor('float32',data,[1,T,NB]),x_lens:new ort.Tensor('int64',BigInt64Array.from([BigInt(T)]),[1])});
 const lp=o.log_probs,L=lp.dims[1];return {lp:lp.data,T:L};}
export function greedy(lp,T,V=31){let prev=-1,s='',conf=[];for(let t=0;t<T;t++){let b=0,bv=-1e9;for(let v=0;v<V;v++){const x=lp[t*V+v];if(x>bv){bv=x;b=v;}}
 if(b!==prev&&b!==0){s+=TOKENS[b];conf.push(Math.exp(bv));}prev=b;}return {text:s.replace(/\s+/g,' ').trim(),conf};}
// ---- text normalisation (mirrors G1) ----
const MAP={'ٱ':'ا','أ':'ا','إ':'ا','آ':'ا','ى':'ي','ة':'ه','ؤ':'و','ئ':'ي','ء':'','ٰ':''};
const LET=new Set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي');
export function norm(w){let o='';for(const c0 of w.normalize('NFC')){const c=MAP[c0]!==undefined?MAP[c0]:c0;if(LET.has(c))o+=c;}return o;}
export function words(ayahText){return ayahText.split(/\s+/).map(w=>({raw:w,key:norm(w)})).filter(w=>w.key);}
// ---- similarity ----
export function lev(a,b){const p=Array.from({length:b.length+1},(_,i)=>i);for(let i=1;i<=a.length;i++){let prev=p[0];p[0]=i;for(let j=1;j<=b.length;j++){const t=p[j];p[j]=Math.min(p[j]+1,p[j-1]+1,prev+(a[i-1]!==b[j-1]?1:0));prev=t;}}return p[b.length];}
// best similarity of word w inside hyp string near expected position (substring match, ratio of edits)
export function bestSim(w,hyp,from,to){let best=0;from=Math.max(0,from);to=Math.min(hyp.length,to);
 for(let a=from;a<=to;a++)for(let L=Math.max(1,w.length-2);L<=w.length+2&&a+L<=hyp.length;L++){const d=lev(w,hyp.substr(a,L));const s=1-d/Math.max(w.length,L);if(s>best)best=s;}return best;}
// classify words of an expected span against a hypothesis. returns per word {state,score}
// Positions: words are laid out proportionally over hyp when the span is fully recited.
export function classify(expWords,hyp,opt={}){const hi=opt.hi??0.86,lo=opt.lo??0.5,win=opt.win??6;
 const total=expWords.reduce((a,w)=>a+w.key.length+1,0);let off=0;const out=[];
 for(const w of expWords){const c=Math.round(off*hyp.length/total);off+=w.key.length+1;
  const s=bestSim(w.key,hyp,c-win-w.key.length,c+win+w.key.length);
  out.push({score:s,state:s>=hi?'ok':(s>=lo?'repeat':'mark')});}return out;}
// ---- sequential alignment: expected words (from cursor) vs one utterance's letters ----
// Hyp is a prefix-ish of expected (end free on expected side). Returns per expected word {score,state,covered}.
export function align(expWords,hypRaw,opt={}){const hi=opt.hi??0.8,lo=opt.lo??0.5;const hyp=hypRaw.replace(/ /g,'');
 const exp=[],wid=[];expWords.forEach((w,i)=>{for(const c of w.key){exp.push(c);wid.push(i);}});
 const n=exp.length,m=hyp.length,D=Array.from({length:n+1},()=>new Int32Array(m+1));
 for(let j=0;j<=m;j++)D[0][j]=j;for(let i=1;i<=n;i++){D[i][0]=i;for(let j=1;j<=m;j++)D[i][j]=Math.min(D[i-1][j]+1,D[i][j-1]+1,D[i-1][j-1]+(exp[i-1]!==hyp[j-1]?1:0));}
 // end free on expected: choose i* minimizing D[i][m] (+ small penalty to prefer longer coverage on ties)
 let bi=n,bv=1e9;for(let i=0;i<=n;i++){const v=D[i][m]-i*1e-3;if(v<bv){bv=v;bi=i;}}
 const match=new Uint8Array(n);let i=bi,j=m;
 while(i>0&&j>0){const d=D[i][j];if(d===D[i-1][j-1]+(exp[i-1]!==hyp[j-1]?1:0)){if(exp[i-1]===hyp[j-1])match[i-1]=1;i--;j--;}else if(d===D[i-1][j]+1)i--;else j--;}
 const cnt=expWords.map(()=>0),len=expWords.map(w=>w.key.length);for(let k=0;k<bi;k++)cnt[wid[k]]+=match[k];
 return expWords.map((w,k)=>{const covered=wid.indexOf(k)<bi&&(wid.lastIndexOf(k)<bi||true);const endIdx=wid.lastIndexOf(k);
  const inSpan=(wid.indexOf(k)<bi);const s=inSpan?cnt[k]/len[k]:0;
  return {score:s,covered:inSpan&&endIdx<bi+3,state:!inSpan?'pending':(s>=hi?'ok':(s>=lo?'repeat':'mark'))};});}
