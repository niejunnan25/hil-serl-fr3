const $ = id => document.getElementById(id);
const phases = {starting:'正在初始化',waiting_actor:'等待 Actor 策略握手',stopping_learner:'正在请求 Learner 停止',waiting_reset:'等待复位',waiting_controller:'等待 Xbox RB',resetting:'正在复位',collecting:'正在采集',awaiting_label:'等待人工裁决',paused:'Actor 已暂停',fault:'记录或设备需要处理',degraded:'任务已降级',stopped:'任务已停止',legacy_running:'旧入口正在运行'};
let status=null, replayId=null, detail=null, timelines={}, players={}, segments={}, syncing=false, gripperBusy=false, controllerPending=null, learnerPending=false;
const names={wrist_1:'腕部视角',side_policy:'侧面视角'};
function text(id,value){$(id).textContent=value;}
function node(tag,content,cls){const n=document.createElement(tag);if(content!==undefined)n.textContent=content;if(cls)n.className=cls;return n;}
function notice(message,error=false){$('notice').hidden=!message;$('notice').textContent=message;$('notice').classList.toggle('error',error);}
async function get(path){const r=await fetch(path,{cache:'no-store'});const v=await r.json();if(!r.ok)throw Error(v.error||'请求失败');return v;}
async function post(path,data){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-HILSERL-Token':status.token},body:JSON.stringify(data)});const v=await r.json();if(!r.ok)throw Error(v.error||'操作失败');return v;}
function query(path,params){return path+'?'+new URLSearchParams(params);}
function formatStamp(value){return Math.floor(value/60)+':'+(value%60).toFixed(1).padStart(4,'0');}
function formatTime(value){return Math.floor(value/60)+':'+String(Math.floor(value%60)).padStart(2,'0');}
function controllerIsBusy(){return controllerPending==='recover'||['recovering','starting'].includes(status?.controller?.phase);}
function actorStateError(actor){
  for(const value of [actor?.error,actor?.robot_state_error?.message,actor?.robot_state_error]){
    if(typeof value==='string'&&value.trim())return value;
  }
  return '机器人状态反馈暂不可用。';
}
function learnerIdentity(runtime){
  return Number.isInteger(runtime?.pid)&&runtime.pid>0&&typeof runtime.start_time==='string'&&runtime.start_time&&typeof runtime.attempt_id==='string'&&runtime.attempt_id;
}
function learnerDisplay(learner,device,runtime,policy,candidate){
  const phase=runtime.phase||policy.state,step=Number.isInteger(runtime.step)&&runtime.step>=0?'step '+runtime.step:'尚未更新';
  const pauseSaving=phase==='saving_checkpoint'&&!runtime.stop_requested;
  let label='未运行',reason='启动 Learner 会新建训练。';
  if(learner){
    if(phase==='saving_checkpoint')label=pauseSaving?'暂停保存 checkpoint':'停止保存 checkpoint';
    else if(phase==='stop_requested')label='正在停止 Learner';
    else if(phase==='closing')label='正在退出';
    else if(phase==='paused'||policy.state==='paused')label=runtime.manual_paused?'已手动暂停':'已自动暂停';
    else if(phase==='initializing')label='正在初始化';
    else if(phase==='fault'||policy.state==='fault')label='Learner 需要处理';
    else if(!device)label='进程运行中 · 等待 Actor';
    else if(phase==='training')label='正在训练';
    else if(policy.state==='waiting')label='等待策略握手';
    else label='进程运行中';
    reason=phase==='training'&&!device?'Actor 已断开，等待 Learner 确认暂停。':
      (phase==='paused'||pauseSaving?runtime.pause_reason:null)||policy.reason||
      (phase==='initializing'?'正在恢复或初始化模型与 replay。':'等待 Actor 采集并收到在线数据。');
    if(pauseSaving)reason+=' 保存完成后保持暂停，进程和训练状态保留。';
    else if(runtime.manual_paused)reason+=' 点击“继续 Learner”解除手动暂停；仍需 Actor 正在采集。';
    return {label,detail:'PID '+learner.pid+' · '+step+'；'+reason,pauseSaving};
  }
  if(runtime.checkpoint_error)reason='checkpoint 保存失败：'+runtime.checkpoint_error;
  else if(candidate)reason='可续训 step '+candidate.step+'；启动 Learner 会新建训练。';
  else if(runtime.checkpoint_saved)reason='checkpoint 已保存 · '+step;
  return {label,detail:reason,pauseSaving:false};
}
function lockLearnerActions(){for(const id of ['start','start-learner','resume-learner','learner-activity','start-actor','stop-learner'])$(id).disabled=true;}
function resetDetail(reset){
  if(!reset)return '';
  const parts=[reset.phase==='clear'?'垂直抬升':reset.phase==='reset_pose'?'移动到起始位姿':'复位检查'];
  if(Number.isFinite(reset.elapsed_seconds))parts.push('已用 '+reset.elapsed_seconds.toFixed(1)+' 秒');
  const phase=reset.phases?.at(-1);
  if(Number.isFinite(phase?.position_error_m))parts.push('位置误差 '+(phase.position_error_m*1000).toFixed(1)+' mm');
  if(Number.isFinite(phase?.rotation_error_rad))parts.push('姿态误差 '+(phase.rotation_error_rad*180/Math.PI).toFixed(1)+'°');
  return parts.join(' · ');
}
async function act(command,value){
  if(controllerIsBusy())return;
  const a=status?.actor;const expected=a?.attempt_id?Object.fromEntries(['attempt_id','episode_id','phase','gate_id'].map(k=>[k,a[k]??null])):null;
  try{await post('/api/command',{command,value,expected});notice('');await refresh();}catch(e){notice(e.message,true);}
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===button));
  document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===button.dataset.view));
  if(button.dataset.view==='replay')loadRecordings();
}));
$('mode').addEventListener('change',()=>{$('eval-options').hidden=$('mode').value!=='eval';});
$('start').addEventListener('click',async()=>{
  if(controllerIsBusy()||learnerPending)return;
  const data={mode:$('mode').value};if(data.mode==='eval'){data.checkpoint=$('checkpoint').value.trim();data.eval_episodes=Number($('eval-episodes').value);if(!data.checkpoint)return notice('请填写要评估的 checkpoint 路径。',true);}
  $('start').disabled=true;
  try{await post('/api/start',data);notice('启动请求已提交。初始化完成后，需点击“复位并开始”才会开始任务。');await refresh();}catch(e){notice(e.message,true);}
});
async function startTrainProcess(learnerOnly,resumeCheckpoint=null){
  if(!status||controllerIsBusy()||learnerPending)return;
  learnerPending=true;lockLearnerActions();
  try{
    const data={mode:'train',learner_only:learnerOnly};
    if(resumeCheckpoint)data.resume_checkpoint=resumeCheckpoint;
    await post('/api/start',data);
    notice(resumeCheckpoint?'Learner 续训请求已提交；原 step 和 optimizer 将恢复，等待 Actor 采集。':learnerOnly?'已请求新建 Learner；等待 Actor 采集后才更新参数。':'Actor 启动请求已提交，等待策略握手。');
  }catch(e){notice(e.message,true);}
  finally{learnerPending=false;await refresh();}
}
$('start-learner').onclick=()=>startTrainProcess(true);
$('resume-learner').onclick=()=>{if(!$('resume-learner').disabled&&status?.resume_candidate?.checkpoint)return startTrainProcess(true,status.resume_candidate.checkpoint);};
$('start-actor').onclick=()=>startTrainProcess(false);
$('learner-activity').onclick=async()=>{
  const runtime=status?.learner_runtime;
  if(learnerPending||controllerIsBusy()||$('learner-activity').disabled||!learnerIdentity(runtime))return;
  const command=runtime.manual_paused?'resume':'pause',expected=Object.fromEntries(['pid','start_time','attempt_id'].map(key=>[key,runtime[key]]));
  learnerPending=true;lockLearnerActions();
  try{const result=await post('/api/learner-activity',{command,expected});notice(result.message||(command==='pause'?'已请求手动暂停 Learner；Actor 继续由独立控制。':'已请求继续 Learner；Actor 正在采集并收到新数据后，才继续更新。'));}
  catch(e){notice(e.message,true);}
  finally{learnerPending=false;await refresh();}
};
$('label-fail').onclick=()=>act('label',0);$('label-success').onclick=()=>act('label',1);
$('continue').onclick=()=>act('continue');$('pause').onclick=()=>act('pause');$('stop').onclick=()=>act('stop');
$('stop-learner').onclick=async()=>{if(controllerIsBusy())return;try{const result=await post('/api/stop-learner',{});notice(result.message||'停止请求已提交，等待 Learner 确认。');await refresh();}catch(e){notice(e.message,true);}};
async function controlController(operation){
  const button=$('controller-'+(operation==='status'?'refresh':'recover'));
  if(!status||controllerPending||controllerIsBusy()||button.disabled||(operation==='recover'&&gripperBusy))return;
  controllerPending=operation;
  const locked=operation==='recover'?['controller-recover','controller-refresh','start','start-learner','resume-learner','learner-activity','start-actor','stop-learner','continue','pause','stop','label-fail','label-success','gripper-open','gripper-close','gripper-refresh']:['controller-recover','controller-refresh'];
  for(const id of locked)$(id).disabled=true;
  renderController();
  try{
    const result=await post('/api/controller/'+operation,{});
    if(operation==='recover')notice(result.message||'底层服务恢复请求已提交，正在检查控制器与状态更新。');
  }catch(e){notice(e.message,true);}
  finally{controllerPending=null;await refresh();}
}
$('controller-recover').onclick=()=>controlController('recover');
$('controller-refresh').onclick=()=>controlController('status');
function renderController({device,launchBusy=false,gripperPending=false}={}){
  const controller=status?.controller||{};
  const phase=controllerPending==='recover'?'recovering':controllerPending==='status'?'checking':controller.phase||'unknown';
  const busy=controllerIsBusy();
  const labels={unknown:'未检查',checking:'正在检查服务',starting:'正在恢复服务',recovering:'正在恢复服务',ready:'状态时效已确认',failed:'底层服务需要处理'};
  const messages={unknown:'停止 Actor 后可恢复底层服务；Learner 可继续运行。',checking:'正在检查控制器与状态更新。',starting:'正在启动控制器并验证状态更新。',recovering:'正在启动控制器并验证状态更新。',ready:'底层服务已就绪，可以读取夹爪开口或启动 Actor。',failed:'服务恢复未完成，请查看错误原因后重试。'};
  text('controller-state',labels[phase]||labels.unknown);
  const details=[controllerPending?messages[phase]:controller.error||controller.message||messages[phase]||messages.unknown];
  const age=controller.health?.state_age_seconds;
  if(phase==='ready'&&Number.isFinite(age))details.push('最近检查时，状态在 '+age.toFixed(2)+' 秒前更新。');
  if(device)details.push('请先停止 Actor，再恢复底层服务。');
  else if(launchBusy)details.push('请等待当前启动或停止操作完成。');
  else if(gripperBusy||gripperPending)details.push('请等待夹爪操作完成。');
  text('controller-detail',details.join(' '));
  $('controller-controls').dataset.phase=phase;
  $('controller-controls').setAttribute('aria-busy',String(busy));
  $('controller-recover').disabled=!status||status.offline||!!controllerPending||busy||!!device||launchBusy||gripperBusy||gripperPending;
  $('controller-refresh').disabled=!status||status.offline||!!controllerPending||busy;
  text('controller-recover',['recovering','starting'].includes(phase)?'正在恢复服务…':'恢复底层服务');
  text('controller-refresh',phase==='checking'?'正在检查…':'检查服务状态');
}
async function controlGripper(operation){
  if(gripperBusy||controllerIsBusy())return;
  gripperBusy=true;for(const id of ['gripper-open','gripper-close','gripper-refresh'])$(id).disabled=true;
  $('controller-recover').disabled=true;
  const device=status?.processes?.find(p=>p.role!=='learner'),a=device?status.actor:null;
  const expected=a?.attempt_id?Object.fromEntries(['attempt_id','episode_id','phase','gate_id'].map(k=>[k,a[k]??null])):null;
  try{const result=await post('/api/gripper',{operation,expected});notice(result.message||(operation==='status'?'已读取夹爪状态。':'已提交夹爪命令，请检查实际开口。'));}
  catch(e){notice(e.message,true);}
  finally{gripperBusy=false;await refresh();}
}
$('gripper-refresh').onclick=()=>controlGripper('status');
$('gripper-open').onclick=()=>controlGripper('open');
$('gripper-close').onclick=()=>controlGripper('close');
document.addEventListener('keydown',e=>{if(e.repeat||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||!$('workspace').classList.contains('active'))return;if(e.key==='0'&&!$('label-fail').disabled){e.preventDefault();act('label',0);}if(e.key==='1'&&!$('label-success').disabled){e.preventDefault();act('label',1);}});
function settings(config){
  const rows=[['每条上限',config.max_episode_steps+' 步'],['示教基准','20 条 · 中位数 94.5 步 · 约 2 倍'],['控制频率',config.control_hz+' Hz'],['录像','两路 720p · 全程连续记录'],['视频分段',config.video_segment_seconds+' 秒'],['人工裁决','0 失败结束 · 1 成功结束'],['数据目录',status.data_dir],['磁盘保留',config.min_free_gib+' GiB']];
  const table=node('table',undefined,'settings-table');for(const [key,value]of rows){const row=table.insertRow();row.append(node('th',key));row.insertCell().textContent=value;}$('settings-content').replaceChildren(table);
}

function episodeMetricContext(value){
  const rows=value?.processes||[],actor=rows.find(x=>x.role!=='learner'),learner=rows.find(x=>x.role==='learner');
  if(actor&&!actor.run_dir)return {run_id:null,signature:null,reason:'当前 Actor 未提供本轮记录'};
  const directory=actor?.run_dir||learner?.run_dir||value?.launch?.run_dir;
  const run=typeof directory==='string'?directory.split('/').filter(Boolean).at(-1):null;
  if(!run||!/^[A-Za-z0-9_.-]+$/.test(run))return {run_id:null,signature:null,reason:'尚未开始可统计的任务'};
  const a=value.actor||{};
  return {run_id:run,signature:JSON.stringify([run,actor?.pid,actor?.start_time,a.attempt_id,
    value.recording_id,a.completed_episodes,a.latest_episode?.episode_id,a.latest_episode?.outcome,a.latest_episode?.steps,a.latest_episode?.human_steps]),reason:null};
}
function currentEpisodeMetric(value){
  const a=value?.actor,process=(value?.processes||[]).find(x=>x.role!=='learner');
  const empty=reason=>({rate:null,human_steps:null,steps:null,reason});
  if(!process||!process.run_dir||!a||!process.start_time||a.pid!==process.pid||a.start_time!==process.start_time)
    return empty('等待采集');
  if(!['collecting','awaiting_label'].includes(a.phase)||typeof a.episode_id!=='string'||!a.episode_id)
    return empty('等待下一条采集');
  if(a.step===0)return empty('本条尚未执行动作');
  if(!Number.isInteger(a.step)||a.step<1||!Number.isInteger(a.human_steps)||a.human_steps<0||a.human_steps>a.step)
    return empty('接管步数暂不可用');
  return {rate:a.human_steps/a.step,human_steps:a.human_steps,steps:a.step,reason:null};
}
function metricEligibleEpisode(e){
  return e?.complete===true&&e.verdict_source==='human'&&Number.isInteger(e.outcome)&&[0,1].includes(e.outcome)
    &&Number.isInteger(e.steps)&&e.steps>0&&Number.isFinite(e.finalized?.unix_ns)&&e.finalized.unix_ns>0
    &&(e.committed_steps===undefined||e.committed_steps===e.steps);
}
function summarizeEpisodeMetrics(items){
  const unique=new Map();
  for(const [index,e] of items.entries())if(metricEligibleEpisode(e))
    unique.set(typeof e.id==='string'&&e.capture_id?e.capture_id+'/'+e.id:'row-'+index,e);
  const selected=[...unique.values()].sort((a,b)=>b.finalized.unix_ns-a.finalized.unix_ns
    ||String(b.capture_id+'/'+b.id).localeCompare(String(a.capture_id+'/'+a.id))).slice(0,10);
  const steps=selected.reduce((sum,e)=>sum+e.steps,0),successes=selected.reduce((sum,e)=>sum+e.outcome,0);
  const known=selected.every(e=>Number.isInteger(e.human_steps)&&e.human_steps>=0&&e.human_steps<=e.steps);
  const human=known?selected.reduce((sum,e)=>sum+e.human_steps,0):null;
  return {episode_count:selected.length,success_count:successes,success_rate:selected.length?successes/selected.length:null,
    steps,human_steps:human,intervention_rate:selected.length&&known?human/steps:null};
}
class EpisodeMetricHistory{
  constructor(fetchJson,onChange){this.fetchJson=fetchJson;this.onChange=onChange;this.run_id=null;this.signature=null;
    this.pending=false;this.ticket=0;this.retryAt=0;this.state={phase:'empty',...summarizeEpisodeMetrics([])};}
  update(value){
    this.latest=value;const context=episodeMetricContext(value);this.desired=context;
    if(context.run_id!==this.run_id){this.run_id=context.run_id;this.signature=null;this.pending=false;this.retryAt=0;this.ticket++;
      this.state={phase:context.run_id?'loading':'empty',run_id:context.run_id,reason:context.reason,...summarizeEpisodeMetrics([])};this.onChange(this.state);}
    if(!context.run_id){this.state={phase:'empty',run_id:null,reason:context.reason,...summarizeEpisodeMetrics([])};this.onChange(this.state);return Promise.resolve();}
    if(this.pending)return this.promise;
    if(context.signature===this.signature&&(this.state.phase==='ready'||Date.now()<this.retryAt))return Promise.resolve();
    const ticket=++this.ticket;this.signature=context.signature;this.pending=true;
    this.state={phase:'loading',run_id:context.run_id,...summarizeEpisodeMetrics([])};this.onChange(this.state);
    this.promise=this.load(context).then(result=>{
      if(ticket!==this.ticket)return;
      if(this.desired.signature!==context.signature)return;
      this.state={phase:'ready',run_id:context.run_id,...result};this.onChange(this.state);
    }).catch(error=>{
      if(ticket!==this.ticket)return;
      this.retryAt=Date.now()+3000;
      this.state={phase:'error',run_id:context.run_id,reason:error.message||'历史记录读取失败',...summarizeEpisodeMetrics([])};this.onChange(this.state);
    }).finally(()=>{
      if(ticket!==this.ticket)return;
      this.pending=false;
      if(this.desired.signature!==context.signature)this.update(this.latest);
    });return this.promise;
  }
  invalidate(reason){this.ticket++;this.pending=false;this.signature=null;this.retryAt=0;
    this.state={phase:'error',run_id:this.run_id,reason,...summarizeEpisodeMetrics([])};this.onChange(this.state);}
  async load(context){
    const records=await this.fetchJson('/api/recordings');
    if(!Array.isArray(records))throw Error('历史记录列表格式无效');
    const upperBound=r=>Number.isFinite(r.ended?.unix_ns)?r.ended.unix_ns:Infinity;
    const candidates=records.filter(r=>r.run_id===context.run_id&&typeof r.id==='string')
      .sort((a,b)=>upperBound(b)-upperBound(a)||b.id.localeCompare(a.id));
    const episodes=[];
    for(const record of candidates){
      if(this.desired.run_id!==context.run_id)return summarizeEpisodeMetrics([]);
      if(record.complete_episodes===0)continue;
      const eligible=episodes.filter(metricEligibleEpisode).sort((a,b)=>b.finalized.unix_ns-a.finalized.unix_ns);
      if(eligible.length>=10&&upperBound(record)<eligible[9].finalized.unix_ns)break;
      const detail=await this.fetchJson(query('/api/recording',{id:record.id}));
      if(!Array.isArray(detail?.episodes))throw Error('episode 记录格式无效');
      for(const episode of detail.episodes)episodes.push({...episode,capture_id:record.id});
    }
    return summarizeEpisodeMetrics(episodes);
  }
}
function renderEpisodeHistory(metric){
  if(!$('recent-intervention-rate'))return;
  const ready=metric.phase==='ready',count=metric.episode_count;
  const pct=value=>value===null?'—':(value*100).toFixed(1)+'%';
  text('recent-intervention-rate',ready?pct(metric.intervention_rate):'—');
  text('recent-success-rate',ready?pct(metric.success_rate):'—');
  const waiting=metric.phase==='loading'?'正在读取本轮记录':metric.phase==='error'?'历史统计暂不可用':metric.reason||'尚无已完成记录';
  text('recent-intervention-detail',ready&&count?(metric.intervention_rate===null?'最近 '+count+' 条中有接管数据缺失':metric.human_steps+' / '+metric.steps+' 步 · 最近 '+count+' 条'):waiting);
  text('recent-success-detail',ready&&count?metric.success_count+' / '+count+' 条成功':waiting);
  text('episode-metrics-scope',metric.phase==='error'?'历史统计暂不可用：'+metric.reason:
    '本轮最近 10 条已完成且人工标注的 episode；接管率按有效步数汇总。');
}
const episodeMetricHistory=new EpisodeMetricHistory(get,renderEpisodeHistory);
function renderEpisodeMetrics(value){
  if(!$('current-intervention-rate'))return;
  const current=currentEpisodeMetric(value);
  text('current-intervention-rate',current.rate===null?'—':(current.rate*100).toFixed(1)+'%');
  text('current-intervention-detail',current.rate===null?current.reason:current.human_steps+' / '+current.steps+' 步由人工接管');
  episodeMetricHistory.update(value);
}

async function refresh(){
  try{
    status=await get('/api/status');const actor=status.actor,cfg=status.effective_config||status.config;
    renderEpisodeMetrics(status);
    const launchBusy=['checking','waiting_learner','waiting_actor','stopping_learner','starting'].includes(status.launch.phase);
    const controllerBusy=controllerIsBusy(),busy=launchBusy||controllerBusy||learnerPending;
    const launchDegraded=['fault','degraded'].includes(status.launch.phase);
    text('connection',status.offline?'离线预览 · 未连接机器人':'本机控制台');$('connection').classList.add('ready');
    text('phase-title',actor?(phases[actor.phase]||actor.phase):launchBusy?(phases[status.launch.phase]||'等待 Learner 初始化'):launchDegraded?'任务已降级':status.launch.phase==='stopped'?'任务已停止':'准备开始');
    text('phase-detail',actor?.error||(actor?.phase==='resetting'?resetDetail(actor.reset):'')||status.launch.error||actor?.prompt||'选择训练、独立评估或人工示教。');
    text('disk-free',status.free_gib.toFixed(1)+' GiB');text('disk-reserve','保留 '+cfg.min_free_gib+' GiB');text('storage-path',status.data_dir);
    text('step-count',actor?.step??'—');text('step-limit',actor?.phase==='legacy_running'?'旧入口配置':'/ '+cfg.max_episode_steps+' 步');text('episodes-count',(actor?.completed_episodes||0)+' 条');
    const learner=status.processes.find(x=>x.role==='learner'),device=status.processes.find(x=>x.role!=='learner');
    const recoverable=!!device&&actor?.recoverable===true&&actor.phase==='waiting_reset';
    const controllerFaultContext=!device&&actor?.phase==='fault'&&/FR3 控制器|底层控制服务/.test(actor.error||'');
    const controllerRecovered=status.controller?.phase==='ready'&&controllerFaultContext;
    if(controllerRecovered){text('phase-title','底层服务已恢复');text('phase-detail','上次 Actor 启动失败的记录已保留。请重新启动 Actor，再确认复位。');}
    else if(controllerFaultContext&&status.controller?.phase==='failed'){text('phase-title','底层服务需要处理');text('phase-detail',status.controller.error||status.controller.message);}
    const learnerPolicy=status.learner_policy||{};
    const learnerRuntime=status.learner_runtime||{};
    const candidate=status.resume_candidate,learnerView=learnerDisplay(learner,device,learnerRuntime,learnerPolicy,candidate);
    text('learner-status',learnerView.label);text('learner-detail',learnerView.detail);
    if(learner&&status.launch.phase==='stopping_learner'){text('phase-title',learnerView.label);text('phase-detail',learnerView.detail);}
    else if(recoverable){text('phase-title','反馈中断，等待确认复位');text('phase-detail',actorStateError(actor)+' Actor 仍在运行；底层状态恢复后，请确认复位。已有模型和数据保留。');}
    else if(learner&&!device&&['paused','saving_checkpoint'].includes(learnerRuntime.phase)){text('phase-title',learnerView.label);text('phase-detail',learnerView.detail);}
    text('recording-status',recoverable?'等待复位 · 持续记录':actor?.phase==='fault'?'记录已中断':status.recording_id?'随任务持续记录':'等待启动');
    $('start').disabled=status.offline||busy||!!device;
    const labels=!status.offline&&!controllerBusy&&['collecting','awaiting_label'].includes(actor?.phase);
    $('label-fail').disabled=$('label-success').disabled=!labels;
    $('continue').hidden=!['waiting_reset','paused','operator_prompt'].includes(actor?.phase);
    $('continue').disabled=status.offline||controllerBusy||!actor?.gate_id;
    text('continue',actor?.phase==='waiting_reset'?'复位并开始':'确认并继续');
    $('pause').disabled=status.offline||controllerBusy||actor?.phase!=='collecting';
    $('stop').disabled=status.offline||controllerBusy||!(launchBusy||(device&&actor?.phase!=='legacy_running'));
    $('start-learner').disabled=status.offline||!!learner||!!device||busy;
    $('resume-learner').disabled=status.offline||!!learner||!!device||busy||!candidate?.checkpoint;
    $('resume-learner').title=candidate?'从 step '+candidate.step+' 续训':'暂无可续训的已保存 checkpoint';
    $('learner-activity').disabled=status.offline||!learner||busy||!learnerIdentity(learnerRuntime)||!['training','paused'].includes(learnerRuntime.phase);
    text('learner-activity',learnerRuntime.manual_paused?'继续 Learner':'手动暂停 Learner');
    text('learner-resume-detail',learner?'Actor 未采集时自动暂停更新，恢复采集后继续原 step。手动暂停后需点击“继续 Learner”。':candidate?'续训将恢复 step '+candidate.step+' 的模型与 optimizer；“启动 Learner”会新建训练。':'暂无可续训的已保存 checkpoint；“启动 Learner”会新建训练。');
    $('start-actor').disabled=status.offline||!learner||!!device||busy||learnerRuntime.stop_requested||['fault','closing','stopped'].includes(learnerRuntime.phase);
    $('stop-learner').disabled=status.offline||!learner||!!device||busy;
    const grip=status.gripper,readback=grip?.after||grip?.before;
    const receiptPending=['sending','queued'].includes(grip?.verification)&&Date.now()*1e6-(grip.unix_ns||0)<10e9;
    renderController({device,launchBusy,gripperPending:receiptPending});
    const gripIdle=!device||(['waiting_reset','paused'].includes(actor?.phase)&&actor?.gripper_available&&actor?.gate_id);
    for(const id of ['gripper-open','gripper-close','gripper-refresh'])$(id).disabled=status.offline||busy||gripperBusy||receiptPending||!gripIdle;
    text('gripper-width',Number.isFinite(readback?.width_mm)?readback.width_mm.toFixed(1)+' mm · 服务器读数':'尚未读取开口');
    text('gripper-detail',grip?.message||(readback?(readback.freshness==='fresh'?'已读取传感器开口；请确认插头夹持情况。':'读数时效未确认；不能据此判断夹爪已经到位。'):'请在 Actor 停止、暂停或等待复位时调整夹爪。'));
    text('runtime-details',(status.processes.map(p=>p.role+' · PID '+p.pid+'\n'+(p.checkpoint||'')).join('\n')||'尚未启动')+(learnerRuntime.checkpoint_saved?'\n已保存 checkpoint：'+learnerRuntime.checkpoint_path:'')+(candidate?'\n可续训 step '+candidate.step+'：'+candidate.checkpoint:''));
    if(status.offline)notice('离线预览不会启动或控制 FR3。合成记录仅用于检查回放和导出。');
    else if(controllerRecovered)notice('底层状态已恢复更新，可以重新启动 Actor。');
    else if(controllerFaultContext&&status.controller?.phase==='failed')notice(status.controller.error||status.controller.message,true);
    else if(recoverable)notice('本条采集因反馈中断而暂停，Actor 进程保留。处理底层状态后，请确认复位。');
    else if(actor?.error||status.launch.error)notice(actor?.error||status.launch.error,true);
    for(const cam of Object.keys(names)){
      const img=$('preview-'+cam),empty=img.parentElement.querySelector('.camera-empty');
      if(status.recording_id){img.onload=()=>{img.hidden=false;empty.hidden=true;};img.onerror=()=>{img.hidden=true;empty.hidden=false;};img.src=query('/api/preview',{camera:cam,t:Date.now()});}
      else{img.hidden=true;empty.hidden=false;}
    }
    text('wrist-state',status.recording_id?'录制中':'未连接');text('side-state',status.recording_id?'录制中':'未连接');settings(cfg);
    if(document.querySelector('.logs').open){const logs=await get('/api/logs');text('logs-output',logs.map(x=>x.role+'\n'+x.text).join('\n\n')||'暂无统一入口日志；旧进程请查看原终端。');}
  }catch(e){if($('current-intervention-rate')){text('current-intervention-rate','—');text('current-intervention-detail','连接中断，等待状态更新');episodeMetricHistory.invalidate('连接中断，等待状态更新');}text('connection','连接中断');text('controller-state','状态无法确认');text('controller-detail',e.message);$('controller-controls').dataset.phase='unknown';$('controller-recover').disabled=$('controller-refresh').disabled=$('resume-learner').disabled=$('learner-activity').disabled=true;notice(e.message,true);}
}
async function loadRecordings(){
  try{
    const records=await get('/api/recordings');const content=$('replay-content');content.className='';
    if(!records.length){content.className='empty-state';content.textContent='还没有录制记录。新的任务会自动保存在数据目录。';return;}
    const list=node('div',undefined,'recording-list');
    for(const record of records){const b=node('button',undefined,'recording-card');b.append(node('strong',(record.synthetic?'合成验证 · ':'')+({train:'在线训练',eval:'独立评估',collect:'人工示教'}[record.mode]||record.mode)));b.append(node('span',new Date(record.started.unix_ns/1e6).toLocaleString('zh-CN')+' · '+record.complete_episodes+' 条有效 · '+record.successes+' 条成功','muted'));b.append(node('small',record.run_id,'muted'));b.onclick=()=>openRecording(record.id);list.append(b);}
    content.replaceChildren(list,node('div',undefined,'recording-detail'));content.lastChild.id='recording-detail';
    if(replayId&&records.some(x=>x.id===replayId))await openRecording(replayId);
  }catch(e){notice(e.message,true);}
}
function frameAt(frames,seconds){let lo=0,hi=frames.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].time_seconds<=seconds)lo=mid;else hi=mid-1;}return frames[lo];}
function seekCamera(cam,seconds,play){
  const frames=timelines[cam],video=players[cam];if(!frames?.length)return;
  const frame=frameAt(frames,seconds),first=frames.find(x=>x.segment===frame.segment);
  const position=(frame.frame-first.frame)/frame.fps;
  if(segments[cam]!==frame.segment){segments[cam]=frame.segment;video.src=query('/api/video',{id:replayId,camera:cam,file:String(frame.segment).padStart(6,'0')+'.mp4'});video.onloadedmetadata=()=>{video.currentTime=Math.min(position,video.duration||position);if(play)video.play().catch(()=>{});};}
  else if(Math.abs(video.currentTime-position)>.12)video.currentTime=position;
  if(play&&video.paused&&video.readyState>=2)video.play().catch(()=>{});
  if(!play)video.pause();
}
function seekAll(seconds,play=false){syncing=true;for(const cam of Object.keys(players))seekCamera(cam,seconds,play);$('replay-seek').value=seconds;text('replay-time',formatTime(seconds));syncing=false;}
async function openRecording(id){
  for(const v of Object.values(players))v.pause();replayId=id;players={};segments={};timelines={};
  try{
    detail=await get(query('/api/recording',{id}));const box=$('recording-detail');box.replaceChildren();
    box.append(node('h2','全程回放'),node('p','视频包含等待、整理和复位。点击 episode 可定位到有效任务段。','muted'));
    const grid=node('div',undefined,'camera-grid');
    for(const cam of Object.keys(names)){
      if(!detail.videos[cam]?.segments.length)continue;
      timelines[cam]=await get(query('/api/timeline',{id,camera:cam}));const figure=node('figure',undefined,'replay-camera'),video=document.createElement('video');video.controls=Object.keys(players).length===0;video.muted=true;video.playsInline=true;video.preload='metadata';players[cam]=video;figure.append(video,node('figcaption',names[cam]));grid.append(figure);
    }box.append(grid);
    const controls=node('div',undefined,'replay-controls'),slider=document.createElement('input');slider.id='replay-seek';slider.type='range';slider.min='0';slider.max=String(Math.max(0,...Object.values(timelines).map(a=>a.at(-1)?.time_seconds||0)));slider.step='.1';slider.value='0';slider.setAttribute('aria-label','完整视频时间轴');const timeLabel=node('span','0:00');timeLabel.id='replay-time';controls.append(slider,timeLabel);box.append(controls);slider.oninput=()=>seekAll(Number(slider.value));
    const master=Object.keys(players)[0];if(master){
      const video=players[master];
      video.ontimeupdate=()=>{if(syncing)return;const rows=timelines[master],first=rows.find(x=>x.segment===segments[master]);if(!first)return;const frame=rows[Math.min(rows.length-1,first.frame+Math.floor(video.currentTime*first.fps))];const seconds=frame.time_seconds;slider.value=seconds;timeLabel.textContent=formatTime(seconds);for(const cam of Object.keys(players))if(cam!==master)seekCamera(cam,seconds,!video.paused);};
      video.onplay=()=>{const seconds=Number(slider.value);for(const cam of Object.keys(players))if(cam!==master)seekCamera(cam,seconds,true);};
      video.onpause=()=>{for(const cam of Object.keys(players))if(cam!==master)players[cam].pause();};
      video.onended=()=>{const next=timelines[master].find(x=>x.segment===segments[master]+1);if(next)seekAll(next.time_seconds,true);};
      seekAll(0);
    }else box.append(node('p','此记录没有可播放的视频片段。','muted'));
    const exportControls=node('div',undefined,'export-controls');
    for(const [id,label,options]of [['outcome','结果',[['all','全部结果'],['success','成功'],['failure','失败']]],['source','动作来源',[['all','全部来源'],['human','人工'],['policy','策略']]],['format','格式',[['npz','通用 NPZ'],['serl','SERL pickle']]]]){
      const field=node('label',label);const select=document.createElement('select');select.id='export-'+id;for(const[value,title]of options){const option=node('option',title);option.value=value;select.append(option);}field.append(select);exportControls.append(field);
    }
    const exportButton=node('button','导出有效数据','primary');exportControls.append(exportButton);box.append(exportControls);
    const table=node('table',undefined,'episode-table'),header=table.createTHead().insertRow();for(const label of ['选择','Episode','结果','步数','人工动作','时间'])header.append(node('th',label));
    const body=table.createTBody();for(const episode of detail.episodes){const row=body.insertRow();const check=document.createElement('input');check.type='checkbox';check.value=episode.id;check.className='episode-select';check.disabled=!episode.complete;check.setAttribute('aria-label','选择 episode '+episode.id);row.insertCell().append(check);const jump=node('button',episode.id,'text-button');jump.onclick=()=>seekAll(episode.start_seconds);row.insertCell().append(jump);row.insertCell().textContent=episode.complete?(episode.outcome?'成功':'失败'):'未完成';row.insertCell().textContent=episode.steps;row.insertCell().textContent=episode.human_steps??'—';row.insertCell().textContent=formatStamp(episode.start_seconds)+'–'+formatStamp(episode.end_seconds);}
    box.append(table,node('p','不勾选时导出所有符合筛选条件的有效 episode。按动作来源筛选保留原始步索引，不把间隔拼成连续轨迹。','muted'));
    const result=node('div',undefined,'export-result');box.append(result);
    exportButton.onclick=async()=>{exportButton.disabled=true;result.textContent='正在整理导出文件…';try{const chosen=[...box.querySelectorAll('.episode-select:checked')].map(x=>x.value);const data={id:replayId,outcome:$('export-outcome').value,source:$('export-source').value,format:$('export-format').value,episode_ids:chosen.length?chosen:null};const output=await post('/api/export',data);const link=node('a','下载 '+output.episodes+' 条 / '+output.steps+' 步');link.href=query('/api/download',{file:output.file});link.download=output.file;result.replaceChildren(link);}catch(e){result.textContent=e.message;}finally{exportButton.disabled=false;}};
  }catch(e){$('recording-detail').textContent=e.message;}
}
refresh();setInterval(refresh,1500);
