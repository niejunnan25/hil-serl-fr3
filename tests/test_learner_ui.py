"""Run the shipped JS against a small DOM/fetch harness; no browser or device I/O."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


WEB = Path(__file__).resolve().parents[1] / "hilserl" / "web"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for the UI execution harness")

HARNESS = r"""
const fs=require('fs'),vm=require('vm');
const input=JSON.parse(fs.readFileSync(0,'utf8')),root=process.argv[1],posts=[];
class Element{
  constructor(id=''){
    this.id=id;this.textContent='';this.hidden=false;this.disabled=false;this.dataset={};this.children=[];this.value='train';this.listeners={};
    const classes=new Set(id==='workspace'?['active']:[]);
    this.classList={add:x=>classes.add(x),contains:x=>classes.has(x),toggle:(x,yes)=>yes?classes.add(x):classes.delete(x)};
    this.parentElement={querySelector:()=>new Element()};
  }
  addEventListener(name,fn){this.listeners[name]=fn;}
  setAttribute(name,value){this[name]=value;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  insertRow(){const row=new Element();this.append(row);return row;}
  insertCell(){const cell=new Element();this.append(cell);return cell;}
  get lastChild(){return this.children.at(-1);}
}
const elements=new Map();for(const match of fs.readFileSync(root+'/index.html','utf8').matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element(match[1]));
const document={getElementById:id=>{if(!elements.has(id))throw Error('Missing DOM id '+id);return elements.get(id);},
  createElement:tag=>new Element(),querySelectorAll:()=>[],querySelector:()=>({open:false}),addEventListener:()=>{}};
const context=vm.createContext({document,URLSearchParams,Date,Math,console,setInterval:()=>{},
  fetch:async(path,options)=>{
    if(path==='/api/status')return {ok:true,json:async()=>input.status};
    if(options?.method==='POST'){posts.push({path,data:JSON.parse(options.body)});return {ok:true,json:async()=>({})};}
    throw Error('Unexpected fetch '+path);
  }});
vm.runInContext(fs.readFileSync(root+'/app.js','utf8'),context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));await context.refresh();
  if(input.action){const button=elements.get(input.action);if(button.onclick)await button.onclick();else if(button.listeners.click)await button.listeners.click();}
  const output={posts};for(const id of ['learner-status','learner-detail','learner-resume-detail','phase-title','phase-detail','recording-status','notice','start-learner','resume-learner','learner-activity','start-actor','stop-learner','stop']){
    const item=elements.get(id);output[id]={text:item.textContent,disabled:item.disabled,hidden:item.hidden,error:item.classList.contains('error'),title:item.title};
  }
  process.stdout.write(JSON.stringify(output));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""


def run_ui(status, action=None):
    result = subprocess.run([shutil.which("node"), "-e", HARNESS, str(WEB)],
                            input=json.dumps({"status": status, "action": action}), text=True,
                            capture_output=True, check=True, timeout=10)
    return json.loads(result.stdout)


@pytest.fixture
def status():
    return dict(processes=[], actor=None, learner_runtime=None, learner_policy={},
                resume_candidate={"checkpoint": "/runs/current/checkpoints/checkpoint_25974", "step": 25974, "run_dir": "/runs/current"},
                launch={"phase": "stopped"}, controller={"phase": "unknown"}, offline=False,
                config={"max_episode_steps": 190, "min_free_gib": 8, "control_hz": 10, "video_segment_seconds": 60},
                data_dir="/runs", free_gib=70, recording_id=None, token="local-test-token")


def active(status, *, phase="paused", manual=False, actor_phase="collecting"):
    status["launch"] = {"phase": "running"}
    status["processes"] = [{"role": "learner", "pid": 123}, {"role": "train", "pid": 456}]
    status["actor"] = dict(phase=actor_phase, pid=456, start_time="actor-start", attempt_id="actor-one", gate_id="gate")
    status["learner_runtime"] = dict(pid=123, start_time="learner-start", attempt_id="learner-one", phase=phase,
                                     manual_paused=manual, step=25974, pause_reason="等待在线数据", stop_requested=False)
    status["learner_policy"] = {"state": phase, "reason": "等待在线数据"}
    return status


def test_resume_button_dispatches_explicit_checkpoint_and_displays_step(status):
    page = run_ui(status, "resume-learner")
    assert page["posts"] == [{"path": "/api/start", "data": {"mode": "train", "learner_only": True, "resume_checkpoint": status["resume_candidate"]["checkpoint"]}}]
    assert not page["resume-learner"]["disabled"]
    assert "25974" in page["learner-resume-detail"]["text"]


def test_plain_start_never_sends_implicit_resume(status):
    page = run_ui(status, "start-learner")
    assert page["posts"] == [{"path": "/api/start", "data": {"mode": "train", "learner_only": True}}]


@pytest.mark.parametrize("manual,command,label", [(False, "pause", "手动暂停 Learner"), (True, "resume", "继续 Learner")])
def test_activity_is_available_with_live_actor_and_sends_identity_fence(status, manual, command, label):
    page = run_ui(active(status, manual=manual), "learner-activity")
    assert not page["learner-activity"]["disabled"]
    assert page["learner-activity"]["text"] == label
    assert page["posts"] == [{"path": "/api/learner-activity", "data": {"command": command, "expected": {"pid": 123, "start_time": "learner-start", "attempt_id": "learner-one"}}}]
    assert "25974" in page["learner-detail"]["text"]
    assert page["stop-learner"]["disabled"]  # Actor need not stop for manual pause.


def test_pause_checkpoint_save_is_not_presented_as_stopping(status):
    active(status, phase="saving_checkpoint")
    page = run_ui(status)
    assert page["learner-status"]["text"] == "暂停保存 checkpoint"
    assert "保存完成后保持暂停" in page["learner-detail"]["text"]
    assert page["learner-activity"]["disabled"]
    status["learner_runtime"]["stop_requested"] = True
    assert run_ui(status)["learner-status"]["text"] == "停止保存 checkpoint"


def test_disconnected_actor_is_never_reported_as_training(status):
    active(status, phase="training")
    status["processes"] = status["processes"][:1]
    status["actor"] = {"phase": "stopped"}
    page = run_ui(status)
    assert page["learner-status"]["text"] == "进程运行中 · 等待 Actor"
    assert "等待 Learner 确认暂停" in page["learner-detail"]["text"]


def test_recoverable_feedback_error_keeps_actor_controls_and_nonfatal_notice(status):
    active(status, actor_phase="waiting_reset")
    status["actor"].update(recoverable=True, error="503 robot state stale", robot_state_error="503 robot state stale")
    page = run_ui(status)
    assert page["phase-title"]["text"] == "反馈中断，等待确认复位"
    assert "Actor 仍在运行" in page["phase-detail"]["text"]
    assert page["notice"]["error"] is False
    assert page["stop"]["disabled"] is False
    assert page["start-actor"]["disabled"] is True


@pytest.mark.parametrize("condition", ["no_candidate", "offline", "live_learner", "recovering"])
def test_resume_button_is_fenced_by_current_state(status, condition):
    if condition == "no_candidate":
        status["resume_candidate"] = None
    elif condition == "offline":
        status["offline"] = True
    elif condition == "live_learner":
        active(status)
    else:
        status["controller"]["phase"] = "recovering"
    page = run_ui(status, "resume-learner")
    assert page["resume-learner"]["disabled"] is True
    assert page["posts"] == []


def test_activity_requires_current_learner_identity(status):
    active(status)
    status["learner_runtime"]["attempt_id"] = None
    page = run_ui(status, "learner-activity")
    assert page["learner-activity"]["disabled"] is True
    assert page["posts"] == []


@pytest.mark.parametrize("error,structured,expected", [
    ("503 controller feedback stale", {"type": "RobotStateUnavailable", "message": "structured fallback", "status_code": 503}, "503 controller feedback stale"),
    (None, {"type": "RobotStateUnavailable", "message": "503 structured feedback stale", "status_code": 503}, "503 structured feedback stale"),
    (None, {"type": "RobotStateUnavailable", "status_code": 503}, "机器人状态反馈暂不可用。"),
])
def test_structured_actor_state_error_is_rendered_as_human_text(status, error, structured, expected):
    active(status, actor_phase="waiting_reset")
    status["actor"].update(recoverable=True, error=error, robot_state_error=structured)
    page = run_ui(status)
    assert expected in page["phase-detail"]["text"]
    assert "[object Object]" not in page["phase-detail"]["text"]
    assert page["notice"]["error"] is False
