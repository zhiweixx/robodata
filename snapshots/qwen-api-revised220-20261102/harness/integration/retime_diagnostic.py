"""Independent free-GPU diagnostic; never resumes or rewrites paid pilot data."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
from collections.abc import Mapping
import numpy as np
from cap_harness.artifacts import RunRecorder
from aspire_glue import build_api,execute_program,implementation_provenance
from robodojo_adapter import RoboDojoAdapter


def plain(value):
    if isinstance(value,Mapping):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    return value


def run_retime_diagnostic(task_env):
    if task_env.task_name!='general_pickup' or int(task_env.env_seeds[0])!=0:
        raise ValueError('retiming diagnostic requires independent general_pickup layout0')
    root=Path(os.environ['ASPIRE_OUTPUT_DIR']).resolve();root.mkdir(parents=True,exist_ok=True)
    recorder=RunRecorder(output_root=root/'artifacts',benchmark='RoboDojo',suite='retime-diagnostic',
        task_id=0,task_name=task_env.task_name,seed=0,max_steps=int(task_env.step_lim),
        camera_width=640,camera_height=480,control_frequency_hz=25,capture_videos=False,step_video=False,
        providers={'planner':'RoboDojo-native-CuRobo','mode':'free-retime-diagnostic'})
    adapter=RoboDojoAdapter(task_env,recorder=recorder)
    report={'scope':'independent free regression of previously failed Cartesian targets',
        'ok':False,'model_requests':0,'artifact_dir':str(recorder.root),
        'implementation_provenance':implementation_provenance(),'native_plan_audits':[]}
    original_result=adapter._native_result
    def audited_result(raw,arm,start):
        index=len(report['native_plan_audits'])
        name=f'native_plan_{index:02d}'
        audit={'index':index,'arm':arm,'native_status':str(raw.get('status'))}
        report['native_plan_audits'].append(audit)
        arrays={'measured_start':start.joint_positions[arm]}
        if raw.get('position') is not None:
            arrays['native_positions']=np.asarray(raw['position'],dtype=float)
            audit['native_first_point_gap_rad']=float(np.max(np.abs(arrays['native_positions'][0]-arrays['measured_start'])))
        dt=float(adapter._manager.planner[adapter._robots[arm].robot_name].dt)
        arrays['native_dt_s']=np.array(dt)
        try:
            result=original_result(raw,arm,start)
            audit['ok']=bool(result.get('ok'));audit['diagnostics']=result.get('diagnostics')
            if result.get('joint_positions') is not None:
                arrays['control_positions']=result['joint_positions']
            return result
        except Exception as exc:
            audit['error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            np.savez_compressed(root/(name+'.npz'),**arrays)
            (root/(name+'.json')).write_text(json.dumps(audit,indent=2,default=str))
    adapter._native_result=audited_result
    started=time.monotonic()
    try:
        report['calibration']=adapter.calibration_sanity()
        recorder.on_reset(adapter.get_observation(),{'diagnostic':'retiming','layout_id':0})
        api=build_api(adapter,recorder=recorder)
        outcome=execute_program(api,Path(__file__).with_name('retime_smoke_program.py'),recorder)
        official=adapter.official_status()
        report.update(ok=outcome.ok,program_result=plain(outcome.result),
            program_error=None if outcome.error is None else str(outcome.error),official=official)
        recorder.finalize(program_ok=outcome.ok,task_success=official['official_success'],
            protocol_success=outcome.ok,termination_reason='program_completed' if outcome.ok else 'program_error',
            program_result=outcome,error=outcome.error)
        return report
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        recorder.close_incomplete(exc);raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        (root/'retime_report.json').write_text(json.dumps(report,indent=2,default=str))
        official=adapter.official_status()
        task_env.success[0]=bool(official['official_success']);task_env.end_flag[0]=True
        adapter.close()
