import * as ort from './11-ort.wasm.min.mjs';
import {logprobs,greedy,words,align} from './core.js';
ort.env.wasm.wasmPaths={mjs:new URL('./9-ort-wasm-simd-threaded.mjs',import.meta.url).href,wasm:new URL('./10-ort-wasm-simd-threaded.wasm',import.meta.url).href};ort.env.wasm.numThreads=1;
const $=id=>document.getElementById(id);
const HI=0.99,LO=0.25,WINDOW=30; // thresholds tuned on Husary Qalun, see report
let sess,texts={},W=[],cursor=0,state=[],ctx,stream,proc,listening=false;
const NAMES=['الفاتحة','البقرة','آل عمران','النساء','المائدة','الأنعام','الأعراف','الأنفال','التوبة','يونس','هود','يوسف','الرعد','إبراهيم','الحجر','النحل','الإسراء','الكهف','مريم','طه','الأنبياء','الحج','المؤمنون','النور','الفرقان','الشعراء','النمل','القصص','العنكبوت','الروم','لقمان','السجدة','الأحزاب','سبأ','فاطر','يس','الصافات','ص','الزمر','غافر','فصلت','الشورى','الزخرف','الدخان','الجاثية','الأحقاف','محمد','الفتح','الحجرات','ق','الذاريات','الطور','النجم','القمر','الرحمن','الواقعة','الحديد','المجادلة','الحشر','الممتحنة','الصف','الجمعة','المنافقون','التغابن','الطلاق','التحريم','الملك','القلم','الحاقة','المعارج','نوح','الجن','المزمل','المدثر','القيامة','الإنسان','المرسلات','النبأ','النازعات','عبس','التكوير','الانفطار','المطففين','الانشقاق','البروج','الطارق','الأعلى','الغاشية','الفجر','البلد','الشمس','الليل','الضحى','الشرح','التين','العلق','القدر','البينة','الزلزلة','العاديات','القارعة','التكاثر','العصر','الهمزة','الفيل','قريش','الماعون','الكوثر','الكافرون','النصر','المسد','الإخلاص','الفلق','الناس'];
NAMES.forEach((n,i)=>{const o=document.createElement('option');o.value=i+1;o.textContent=(i+1)+' - '+n;$('sura').appendChild(o);});
$('sura').value=105;
const verdicts=JSON.parse(localStorage.getItem('verdicts')||'[]');const upd=()=>$('count').textContent=verdicts.length+' حكما محفوظا';upd();
function setStatus(t){$('status').textContent=t;}
async function init(){const parts=[];for(const n of ['7-model.part00','8-model.part01']){const r=await fetch(n);if(!r.ok)throw new Error('model '+n);parts.push(new Uint8Array(await r.arrayBuffer()));}
 const model=new Uint8Array(parts.reduce((a,p)=>a+p.length,0));let o=0;parts.forEach(p=>{model.set(p,o);o+=p.length;});
 sess=await ort.InferenceSession.create(model,{executionProviders:['wasm']});
 for(const r of['qal','haf'])texts[r]=await (await fetch({qal:'5-qal.json',haf:'6-haf.json'}[r])).json();setStatus('جاهز. اختر السورة ثم اضغط ابدأ.');load();}
function load(){const r=$('riwaya').value,s=$('sura').value,ay=texts[r][s];const f=Math.max(1,+$('from').value)-1,t=Math.min(ay.length,+$('to').value);
 W=ay.slice(f,t).flatMap(a=>words(a));state=W.map(()=>'hidden');cursor=0;render();}
function render(){const box=$('text');box.innerHTML='';W.forEach((w,i)=>{const s=document.createElement('span');s.className='w '+state[i];s.textContent=state[i]==='hidden'?w.raw:(state[i]==='repeat'?w.raw+' ؟':w.raw);s.onclick=()=>openVerdict(i);box.appendChild(s);});}
async function handle(wave){setStatus('جارٍ التحليل…');const {lp,T}=await logprobs(ort,sess,wave);const hyp=greedy(lp,T).text;
 const span=W.slice(cursor,cursor+WINDOW);if(!span.length)return;const res=align(span,hyp,{hi:HI,lo:LO});
 let firstRepeat=-1,lastCovered=-1;
 res.forEach((r,k)=>{if(r.state==='pending')return;lastCovered=k;const i=cursor+k;
   if(r.state==='ok')state[i]='ok';else if(r.state==='repeat'){state[i]='repeat';if(firstRepeat<0)firstRepeat=k;}else{state[i]='mark';if($('vib').checked&&navigator.vibrate)navigator.vibrate(120);}});
 if(lastCovered>=0)cursor+= (firstRepeat>=0?firstRepeat:lastCovered+1);
 render();setStatus(firstRepeat>=0?'لم أتأكد من بعض الكلمات: أعد قراءتها.':'تمام. تابع.');window.__lastHyp=hyp;}
// --- microphone with simple pause-based segmentation ---
async function startMic(){stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,channelCount:1}});
 ctx=new AudioContext();const src=ctx.createMediaStreamSource(stream);proc=ctx.createScriptProcessor(4096,1,1);
 let buf=[],silent=0,voiced=false;const rate=ctx.sampleRate;
 proc.onaudioprocess=e=>{const x=e.inputBuffer.getChannelData(0);let en=0;for(let i=0;i<x.length;i++)en+=x[i]*x[i];en=Math.sqrt(en/x.length);
  if(en>0.012){voiced=true;silent=0;}else silent+=x.length/rate;if(voiced)buf.push(new Float32Array(x));
  const dur=buf.length*4096/rate;if(voiced&&((silent>0.7&&dur>1)||dur>18)){const all=new Float32Array(buf.length*4096);buf.forEach((b,i)=>all.set(b,i*4096));buf=[];voiced=false;silent=0;handle(resample(all,rate));}};
 src.connect(proc);proc.connect(ctx.destination);listening=true;$('start').textContent='إيقاف';$('start').classList.add('rec');setStatus('أستمع… اقرأ.');}
function stopMic(){proc&&proc.disconnect();stream&&stream.getTracks().forEach(t=>t.stop());ctx&&ctx.close();listening=false;$('start').textContent='ابدأ الاستماع';$('start').classList.remove('rec');setStatus('متوقف.');}
function resample(x,from){if(from===16000)return x;const r=from/16000,n=Math.floor(x.length/r),o=new Float32Array(n);for(let i=0;i<n;i++){const p=i*r,a=Math.floor(p),f=p-a;o[i]=x[a]*(1-f)+(x[a+1]??x[a])*f;}return o;}
$('start').onclick=()=>listening?stopMic():startMic();
$('reset').onclick=load;$('skip').onclick=()=>{if(cursor<W.length){state[cursor]='mark';cursor++;render();}};
['riwaya','sura','from','to'].forEach(id=>$(id).onchange=load);
$('file').onchange=async e=>{const f=e.target.files[0];if(!f)return;const ac=new AudioContext();const ab=await ac.decodeAudioData(await f.arrayBuffer());
 const ch=ab.getChannelData(0);const w=resample(ch,ab.sampleRate);let i=0;while(i<w.length){let j=Math.min(w.length,i+16000*14);if(j<w.length){let best=1e9,bk=j;for(let k=j;k<Math.min(w.length-1600,j+16000*5);k+=800){let e=0;for(let q=k;q<k+1600;q++)e+=Math.abs(w[q]);if(e<best){best=e;bk=k+800;}}j=bk;}await handle(w.subarray(i,j));i=j;}};
let cur=-1;function openVerdict(i){cur=i;$('vword').textContent=W[i].raw+'  ('+state[i]+')';$('verdict').style.display='block';}
$('verdict').onclick=e=>{const v=e.target.dataset&&e.target.dataset.v;if(!v)return;if(v!=='close'&&cur>=0){verdicts.push({t:Date.now(),riwaya:$('riwaya').value,sura:+$('sura').value,word_index:cur,word:W[cur].raw,shown_state:state[cur],muqri_verdict:v});localStorage.setItem('verdicts',JSON.stringify(verdicts));upd();}$('verdict').style.display='none';};
$('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(verdicts,null,1)],{type:'application/json'}));a.download='verdicts.json';a.click();};
window.__ready=init().catch(e=>setStatus('خطأ: '+e));
